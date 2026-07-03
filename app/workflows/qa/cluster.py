"""Unsupervised root-cause grouping of model_1 errors (QA stage 6).

Runs *after* model_2 scoring. Takes the rows where ``IF_ERROR=True`` (model_1's
hard label disagreed with the imported ``IS_OWN_RESTAURANT`` label, or the
label was null) and groups them into error **archetypes**.

Why cluster in SHAP space (not raw features)
--------------------------------------------
Clustering the raw feature matrix of the error subset mostly recovers the
natural data distribution (region, cuisine, ...), not the *error modes*. So we
cluster each error row by its model_2 **grouped SHAP vector** — the per-row,
signed SHAP contribution of every source column to "this looks like an error".
Two rows land in the same cluster when model_2 flags them *for the same
reason*, which is far closer to "same root cause" than raw-feature proximity.

The grouping (embedding PCs summed, signed, into their source column) is shared
byte-for-byte with the SHAP stage (`workflows.qa.shap_eval`).

Lineage cross-tabs
------------------
SHAP says *which input* is implicated; it cannot, on its own, say *which
pipeline stage* (scraping / setup / ETL) produced the bad value. So every
cluster is cross-tabbed against **every column** in the scored parquet (minus
model outputs / target / near-unique ids; see `CLUSTER_LINEAGE_EXCLUDE`) with an
enrichment score: a cluster that is 80% drawn from one execution that is only 5%
of all errors is a strong "look here" signal for the QA person. Continuous
numeric columns are quantile-binned, datetimes bucket by day, and categoricals
are used as-is.

From archetypes to value-resolved groups
----------------------------------------
A SHAP archetype tells us *which columns* drive an error mode, but a single
archetype can still mix many concrete culprits (Mexico vs Colombia, exec A vs
exec B). So each archetype is **subdivided by the value-combination of its
main-driver columns** — the columns whose ``total_enrichment`` (sum of their
over-represented values' enrichment) clears ``CLUSTER_DRIVER_IMPORTANCE_LEVEL``.
The result is one group per concrete pattern: every ``main_driver`` column is
pinned to a single value (one country + one parent_app + one execution_id, ...).
A value/sub-group is only kept when it holds at least ``CLUSTER_MIN_GROUP_ROWS``
error rows, which drops near-unique noise and bounds the group count.

Outputs land under ``data/ml/model_2/clusters/<scored_stem>/``:
  * cluster_report.json         — THE headline. A ``groups`` list (biggest
                                   first); each group carries its size/share,
                                   error-direction mix, ``main_drivers`` (each
                                   column -> its column ``total_enrichment`` plus
                                   the single pinned ``value`` and that value's
                                   ``value_enrichment``) and ``not_important``
                                   (the model's other suspected columns that were
                                   not over-represented here).
  * cluster_assignments.parquet — row-level detail: row_index, archetype
                                   (ERROR_CLUSTER), final ERROR_GROUP, error
                                   direction, proba + lineage columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from bootstrap import ensure_app_on_path
from common.config.columns import IF_ERROR_COL, TARGET_COL
from common.pipeline_timing import step as pipeline_step
from workflows.qa.config import (
    CLUSTER_CROSSTAB_TOP_N,
    CLUSTER_DESCRIBE_UNDER,
    CLUSTER_DIR,
    CLUSTER_DRIVER_EXCLUDE,
    CLUSTER_DRIVER_IMPORTANCE_LEVEL,
    CLUSTER_K,
    CLUSTER_K_RANGE,
    CLUSTER_LINEAGE_COLS,
    CLUSTER_LINEAGE_EXCLUDE,
    CLUSTER_MAX_GROUPS,
    CLUSTER_MAX_SPLIT_COLS,
    CLUSTER_MIN_GROUP_ROWS,
    CLUSTER_MIN_ROWS,
    CLUSTER_NUMERIC_BINS,
    MODEL_1_PRED_COL,
    MODEL_2_PROBA_COL,
    SHAP_CHUNK_SIZE,
)
from workflows.qa.shap_eval import (
    _build_feature_groups,
    _shap_values_positive_class,
)

ensure_app_on_path(__file__)

ASSIGNMENTS_PARQUET = "cluster_assignments.parquet"
REPORT_JSON = "cluster_report.json"
# How many top (by |mean signed SHAP|) source columns describe an archetype
# (used to scope the model's "reasons" and the not_important list).
TOP_GROUPS_PER_CLUSTER = 6
# Max distinct values shown in a node's `descriptions` breakdown (e.g. brand
# arrays under BRAND_COUNT); the remainder is summed into an "(other)" entry.
DESCRIBE_TOP_N = 20
CLUSTER_COL = "ERROR_CLUSTER"
DIRECTION_COL = "ERROR_DIRECTION"

# Numeric dtypes that get quantile-binned (rather than cross-tabbed value-by-
# value) when used as a lineage column.
_NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)

# Meaning-first names for the direction of each model_1 error, plus a plain-
# language glossary embedded in the report so it reads standalone.
DIR_OWN_NOT_CAPTURED = "own_not_captured"
DIR_UNEXPECTED_OWN = "unexpected_own_flag"
DIR_LABEL_MISSING = "label_missing"
DIR_NO_DISAGREEMENT = "no_disagreement"
DIRECTION_GLOSSARY = {
    DIR_OWN_NOT_CAPTURED: (
        "model_1 predicted IS_OWN_RESTAURANT=True but the imported data says "
        "False — looks like a genuine own-restaurant the input data failed to "
        "capture (missed own flag upstream)."
    ),
    DIR_UNEXPECTED_OWN: (
        "model_1 predicted IS_OWN_RESTAURANT=False but the imported data flags "
        "it as True — an own flag the model does not expect (likely an "
        "over-matched / wrongly-set own flag upstream)."
    ),
    DIR_LABEL_MISSING: (
        "the imported IS_OWN_RESTAURANT label was null — a completeness gap "
        "(value dropped somewhere in scraping/setup/ETL), not a wrong value."
    ),
    DIR_NO_DISAGREEMENT: (
        "no disagreement between prediction and label (should not appear inside "
        "the IF_ERROR subset)."
    ),
}


@dataclass
class ClusterResult:
    out_dir: Path
    n_errors: int
    n_clusters: int
    k_selection: str
    skipped: bool
    reason: str | None = None


def _error_direction_expr() -> pl.Expr:
    """Signed direction of each model_1 error (for interpretable clusters).

    A null ground truth is its own class (a completeness problem), while the two
    disagreement directions usually trace to different stages.
    """
    pred = pl.col(MODEL_1_PRED_COL).cast(pl.Boolean)
    truth = pl.col(TARGET_COL).cast(pl.Boolean)
    return (
        pl.when(truth.is_null())
        .then(pl.lit(DIR_LABEL_MISSING))
        .when(pred & ~truth)
        .then(pl.lit(DIR_OWN_NOT_CAPTURED))
        .when(~pred & truth)
        .then(pl.lit(DIR_UNEXPECTED_OWN))
        .otherwise(pl.lit(DIR_NO_DISAGREEMENT))  # should not occur in IF_ERROR
        .alias(DIRECTION_COL)
    )


def _grouped_shap_for_subset(
    classifier,
    X_sub: np.ndarray,
    onehot: np.ndarray,
    chunk_size: int,
    timer=None,
) -> np.ndarray:
    """Per-row signed grouped SHAP for the error subset, computed in chunks."""
    import shap

    explainer = shap.TreeExplainer(classifier)
    n_groups = onehot.shape[1]
    out = np.empty((X_sub.shape[0], n_groups), dtype=np.float64)
    n_total = X_sub.shape[0]
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    with pipeline_step(
        "qa.6a.shap_values",
        f"SHAP values over IF_ERROR subset ({n_total:,} rows, {n_chunks} chunks)",
        "shap.TreeExplainer.shap_values",
        timer=timer,
    ):
        for start in range(0, n_total, chunk_size):
            sl = slice(start, start + chunk_size)
            sv = _shap_values_positive_class(explainer, X_sub[sl])
            out[sl] = sv @ onehot
    return out


def _select_k(
    features: np.ndarray,
    *,
    k_fixed: int | None,
    k_range: tuple[int, int],
    random_state: int,
) -> tuple[np.ndarray, int, str]:
    """Fit KMeans, choosing k by silhouette unless one is pinned.

    Returns (labels, k, selection_description).
    """
    from sklearn.cluster import KMeans

    n = features.shape[0]
    if k_fixed is not None:
        k = max(1, min(int(k_fixed), n))
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
            features
        )
        return labels, k, f"fixed_k={k}"

    from sklearn.metrics import silhouette_score

    lo, hi = k_range
    hi = min(hi, n - 1)
    if hi < lo or n < 3:
        # Too few points to choose between ks — one cluster.
        return np.zeros(n, dtype=np.int64), 1, "single_cluster (too few rows)"

    best_labels: np.ndarray | None = None
    best_k = lo
    best_score = -np.inf
    scores: dict[int, float] = {}
    for k in range(lo, hi + 1):
        labels = KMeans(
            n_clusters=k, random_state=random_state, n_init=10
        ).fit_predict(features)
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(features, labels))
        except Exception:  # noqa: BLE001
            continue
        scores[k] = score
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    if best_labels is None:
        return np.zeros(n, dtype=np.int64), 1, "single_cluster (silhouette failed)"
    score_txt = ", ".join(f"k={k}:{s:.4f}" for k, s in sorted(scores.items()))
    return best_labels, best_k, f"silhouette best_k={best_k} ({score_txt})"


def _crosstab_value_expr(df: pl.DataFrame, value_col: str, n_bins: int) -> pl.Expr:
    """Build the categorical 'value' expression for a lineage column.

    - datetimes/dates -> day buckets,
    - continuous numerics (distinct > n_bins) -> equal-frequency quantile bins,
    - everything else (booleans, strings, low-cardinality numerics) -> as-is.
    """
    dtype = df.schema[value_col]
    col = pl.col(value_col)
    if dtype in (pl.Datetime, pl.Date):
        return col.dt.date().cast(pl.Utf8).fill_null("<null>")
    if dtype in _NUMERIC_DTYPES and df.get_column(value_col).n_unique() > n_bins:
        try:
            return (
                col.qcut(n_bins, allow_duplicates=True)
                .cast(pl.Utf8)
                .fill_null("<null>")
            )
        except Exception:  # noqa: BLE001  fall back to raw values if qcut fails
            pass
    return col.cast(pl.Utf8).fill_null("<null>")


def _crosstab_enrichment(
    df: pl.DataFrame,
    value_col: str,
    *,
    top_n: int,
    n_bins: int,
) -> pl.DataFrame:
    """Long-format cluster x value counts with cluster/overall share enrichment.

    For each cluster we keep only the `top_n` most frequent values so the file
    stays small even for high-cardinality columns (EXECUTION_ID). A high
    `cluster_share` paired with a low `overall_share` is the "look here" signal.
    """
    work = df.select(
        pl.col(CLUSTER_COL),
        _crosstab_value_expr(df, value_col, n_bins).alias("value"),
    )
    overall_n = work.height
    overall = (
        work.group_by("value")
        .agg(pl.len().alias("overall_count"))
        .with_columns((pl.col("overall_count") / overall_n).alias("overall_share"))
    )
    per_cluster = (
        work.group_by([CLUSTER_COL, "value"]).agg(pl.len().alias("count"))
    )
    cluster_sizes = work.group_by(CLUSTER_COL).agg(pl.len().alias("cluster_size"))
    out = (
        per_cluster.join(cluster_sizes, on=CLUSTER_COL)
        .join(overall, on="value")
        .with_columns(
            (pl.col("count") / pl.col("cluster_size")).alias("cluster_share")
        )
        .with_columns(
            (pl.col("cluster_share") / pl.col("overall_share")).alias("enrichment")
        )
        .sort([CLUSTER_COL, "count"], descending=[False, True])
    )
    # Keep top_n values per cluster.
    out = out.with_columns(
        pl.col("count").rank("ordinal", descending=True).over(CLUSTER_COL).alias("_rk")
    )
    out = out.filter(pl.col("_rk") <= top_n).drop("_rk")
    return out.select(
        CLUSTER_COL,
        "value",
        "count",
        "cluster_size",
        "cluster_share",
        "overall_count",
        "overall_share",
        "enrichment",
    )


def _resolve_lineage_cols(
    schema_names: list[str],
    configured: list[str] | None,
    exclude: list[str],
) -> list[str]:
    """Lineage columns to cross-tab: explicit list, or every column minus excludes.

    `None`/empty `configured` means "profile each cluster across all available
    fields" (every column in the scored parquet except `exclude` and the
    helper/label columns). An explicit list restricts it (still filtered to
    columns that are actually present).
    """
    helper = {IF_ERROR_COL, CLUSTER_COL, DIRECTION_COL, "row_index"}
    if configured:
        return [c for c in configured if c in schema_names]
    excl = set(exclude) | helper
    return [c for c in schema_names if c not in excl]


def run_error_clustering(
    result,
    scored_path: Path,
    *,
    cluster_dir: Path = CLUSTER_DIR,
    k: int | None = CLUSTER_K,
    k_range: tuple[int, int] = CLUSTER_K_RANGE,
    min_rows: int = CLUSTER_MIN_ROWS,
    lineage_cols: list[str] | None = CLUSTER_LINEAGE_COLS,
    lineage_exclude: list[str] = CLUSTER_LINEAGE_EXCLUDE,
    numeric_bins: int = CLUSTER_NUMERIC_BINS,
    importance_level: float = CLUSTER_DRIVER_IMPORTANCE_LEVEL,
    driver_exclude: list[str] = CLUSTER_DRIVER_EXCLUDE,
    describe_under: dict[str, str] = CLUSTER_DESCRIBE_UNDER,
    min_group_rows: int = CLUSTER_MIN_GROUP_ROWS,
    max_split_cols: int = CLUSTER_MAX_SPLIT_COLS,
    max_groups: int = CLUSTER_MAX_GROUPS,
    crosstab_top_n: int = CLUSTER_CROSSTAB_TOP_N,
    chunk_size: int = SHAP_CHUNK_SIZE,
    random_state: int = 42,
    timer=None,
) -> ClusterResult:
    """Cluster the IF_ERROR=True rows in model_2 grouped-SHAP space.

    `result` is the `Model2ScoreResult` from the model_2 stage (carries the
    fitted classifier, feature matrix `X`, and feature column names). The error
    subset is selected from `scored_path`'s `IF_ERROR` column (aligned row-for-
    row with `result.X`).
    """
    scored_path = Path(scored_path)
    stem = scored_path.name.removesuffix(".parquet").removesuffix(".scored")
    out_dir = Path(cluster_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = list(result.feature_columns)
    X = result.X

    # Feature groups map every model_2 feature back to its source column (embedding
    # PCs -> base text column). The SHAP-driver group names double as the parquet
    # columns we look up exact culprit values from.
    group_names, onehot, _ = _build_feature_groups(feature_columns)

    # ---- Select the IF_ERROR=True subset (aligned to X by row order) --------
    available = pl.read_parquet_schema(scored_path)
    schema_names = list(available)
    lineage_cols = _resolve_lineage_cols(schema_names, lineage_cols, lineage_exclude)
    # Source columns behind the SHAP drivers, so each top driver can report its
    # exact value(s). Columns in `lineage_exclude` are barred from the report
    # entirely — never a cross-tab dimension AND never a driver (even though some,
    # e.g. CUISINES / FOOD_CATEGORIES, are model_2 feature sources).
    _exclude_set = set(lineage_exclude or [])
    driver_cols = [g for g in group_names if g in available and g not in _exclude_set]
    # Always read the columns needed to derive the error direction + proba, even
    # if they are excluded from the lineage cross-tabs.
    meta_cols = [IF_ERROR_COL]
    for c in [
        MODEL_1_PRED_COL,
        TARGET_COL,
        MODEL_2_PROBA_COL,
        *lineage_cols,
        *driver_cols,
    ]:
        if c in available and c not in meta_cols:
            meta_cols.append(c)
    meta = pl.read_parquet(scored_path, columns=meta_cols)
    if meta.height != X.shape[0]:
        raise ValueError(
            f"Row mismatch: scored parquet has {meta.height:,} rows but model_2 "
            f"feature matrix has {X.shape[0]:,}. Re-run the model_2 stage."
        )

    err_mask = meta.get_column(IF_ERROR_COL).cast(pl.Boolean).fill_null(False).to_numpy()
    err_idx = np.flatnonzero(err_mask)
    n_errors = int(err_idx.size)

    if n_errors < min_rows:
        reason = (
            f"only {n_errors} IF_ERROR=True rows (< CLUSTER_MIN_ROWS={min_rows}); "
            "skipping clustering."
        )
        (out_dir / REPORT_JSON).write_text(
            json.dumps(
                {
                    "scored_parquet": str(scored_path),
                    "n_errors": n_errors,
                    "min_rows": int(min_rows),
                    "skipped": True,
                    "reason": reason,
                    "groups": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[qa/cluster] {reason}", flush=True)
        return ClusterResult(
            out_dir=out_dir,
            n_errors=n_errors,
            n_clusters=0,
            k_selection="skipped",
            skipped=True,
            reason=reason,
        )

    # ---- Grouped SHAP signature for every error row -------------------------
    X_sub = X[err_idx]
    sv_group = _grouped_shap_for_subset(
        result.classifier, X_sub, onehot, chunk_size, timer=timer
    )

    # Standardise so no single high-variance column dominates the distance.
    from sklearn.preprocessing import StandardScaler

    sv_scaled = StandardScaler().fit_transform(sv_group)

    with pipeline_step(
        "qa.6b.kmeans",
        f"KMeans over {n_errors:,} error rows in SHAP space",
        "sklearn.cluster.KMeans",
        timer=timer,
    ):
        labels, n_clusters, k_selection = _select_k(
            sv_scaled, k_fixed=k, k_range=k_range, random_state=random_state
        )

    # ---- Assemble the per-row assignment table ------------------------------
    # `err_idx` is ascending (np.flatnonzero), so filtering `meta` by the same
    # mask preserves the row order that `labels`/`sv_group` are in.
    sub_meta = meta.with_columns(
        pl.Series("row_index", np.arange(meta.height, dtype=np.int64))
    ).filter(pl.Series(err_mask))
    if MODEL_1_PRED_COL in sub_meta.columns and TARGET_COL in sub_meta.columns:
        sub_meta = sub_meta.with_columns(_error_direction_expr())
    else:
        sub_meta = sub_meta.with_columns(pl.lit("unknown").alias(DIRECTION_COL))

    enriched = sub_meta.with_columns(
        pl.Series(CLUSTER_COL, labels.astype(np.int64))
    )

    # ---- Cluster SHAP signature (mean signed grouped SHAP) ------------------
    # Used only to scope WHICH source columns are the model's "reasons" per
    # archetype (so we know which columns to consider as drivers / report as
    # not_important). The groups themselves are defined by value enrichment.
    n_feat_groups = len(group_names)
    signature = np.zeros((n_clusters, n_feat_groups), dtype=np.float64)
    arche_sizes = np.zeros(n_clusters, dtype=np.int64)
    for c in range(n_clusters):
        m = labels == c
        arche_sizes[c] = int(m.sum())
        if arche_sizes[c]:
            signature[c] = sv_group[m].mean(axis=0)

    # Cross-tab every lineage column AND every SHAP-driver source column present.
    present_lineage = [c for c in lineage_cols if c in enriched.columns]
    crosstab_cols = list(
        dict.fromkeys(
            [c for c in (*present_lineage, *driver_cols) if c in enriched.columns]
        )
    )
    crosstabs = {
        col: _crosstab_enrichment(
            enriched, col, top_n=crosstab_top_n, n_bins=numeric_bins
        )
        for col in crosstab_cols
    }
    numeric_cols = {c for c in crosstab_cols if available.get(c) in _NUMERIC_DTYPES}

    # Pre-bin every candidate column to its cross-tab "value" string so the
    # value-splitting groups by the same buckets the enrichment was computed on.
    valued = enriched.with_columns(
        [
            _crosstab_value_expr(enriched, col, numeric_bins).alias(_val_col(col))
            for col in crosstab_cols
        ]
    )

    # ---- Build the value-resolved groups + report --------------------------
    report, group_assignments = _build_groups_report(
        scored_path=scored_path,
        backend=result.manifest.get("backend"),
        n_errors=n_errors,
        n_clusters=n_clusters,
        arche_sizes=arche_sizes,
        signature=signature,
        group_names=group_names,
        valued=valued,
        crosstabs=crosstabs,
        crosstab_cols=crosstab_cols,
        numeric_cols=numeric_cols,
        importance_level=importance_level,
        driver_exclude=driver_exclude,
        describe_under=describe_under,
        min_group_rows=min_group_rows,
        max_split_cols=max_split_cols,
        max_groups=max_groups,
        k_selection=k_selection,
    )
    (out_dir / REPORT_JSON).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ---- Row-level assignments (archetype + final group id) ----------------
    keep = ["row_index", CLUSTER_COL, DIRECTION_COL]
    if MODEL_2_PROBA_COL in enriched.columns:
        keep.append(MODEL_2_PROBA_COL)
    keep += [c for c in lineage_cols if c in enriched.columns]
    assignments_out = enriched.select(keep).join(
        group_assignments, on="row_index", how="left"
    )
    assignments_out.write_parquet(out_dir / ASSIGNMENTS_PARQUET)

    n_final_groups = len(report["groups"])
    print(
        f"[qa/cluster] {n_errors:,} IF_ERROR rows -> {n_clusters} archetype(s) -> "
        f"{n_final_groups} value-resolved group(s) -> {out_dir / REPORT_JSON} "
        f"({k_selection}).",
        flush=True,
    )
    return ClusterResult(
        out_dir=out_dir,
        n_errors=n_errors,
        n_clusters=int(n_clusters),
        k_selection=k_selection,
        skipped=False,
    )


def _val_col(col: str) -> str:
    """Name of the pre-binned cross-tab value column for `col`."""
    return f"__val::{col}"


def _archetype_column_stats(
    crosstab: pl.DataFrame | None,
    cluster: int,
    min_group_rows: int,
) -> tuple[dict, dict, list[dict]]:
    """For one column in one archetype: total_enrichment stats, value->enrichment
    map, and the over-represented (enrichment>1, count>=min) values, richest first.

    total_enrichment is reported as ``{"max", "mean", "sum"}`` over the column's
    over-represented *and substantial* values (the substantiality gate drops
    near-unique noise whose values are tiny). The ``sum`` doubles as the column's
    main-driver gate (see ``_build_groups_report``).
    """
    if crosstab is None:
        return {"max": 0.0, "mean": 0.0, "sum": 0.0}, {}, []
    sub = crosstab.filter(pl.col(CLUSTER_COL) == cluster)
    enr_map: dict = {}
    over: list[dict] = []
    for r in sub.iter_rows(named=True):
        enr = float(r["enrichment"])
        enr_map[r["value"]] = enr
        if enr > 1.0 and int(r["count"]) >= min_group_rows:
            over.append(
                {"value": r["value"], "enrichment": enr, "count": int(r["count"])}
            )
    over.sort(key=lambda d: d["enrichment"], reverse=True)
    enrs = [d["enrichment"] for d in over]
    stats = {
        "max": round(float(max(enrs)), 4) if enrs else 0.0,
        "mean": round(float(sum(enrs) / len(enrs)), 4) if enrs else 0.0,
        "sum": round(float(sum(enrs)), 4),
    }
    return stats, enr_map, over


def _build_groups_report(
    *,
    scored_path: Path,
    backend,
    n_errors: int,
    n_clusters: int,
    arche_sizes: np.ndarray,
    signature: np.ndarray,
    group_names: list[str],
    valued: pl.DataFrame,
    crosstabs: dict[str, pl.DataFrame],
    crosstab_cols: list[str],
    numeric_cols: set[str],
    importance_level: float,
    driver_exclude: list[str],
    describe_under: dict[str, str],
    min_group_rows: int,
    max_split_cols: int,
    max_groups: int,
    k_selection: str,
) -> tuple[dict, pl.DataFrame]:
    """Subdivide each SHAP archetype into value-resolved groups.

    A group resolves every main-driver column to a single value. Returns the
    JSON-ready report and a (row_index -> group id) DataFrame for the
    row-level assignment parquet.
    """
    driver_exclude_set = set(driver_exclude or [])
    groups: list[dict] = []

    for c in range(n_clusters):
        if arche_sizes[c] == 0:
            continue
        rows_c = valued.filter(pl.col(CLUSTER_COL) == c)

        # SHAP "reasons" for this archetype: the top source columns by |SHAP|.
        order = np.argsort(-np.abs(signature[c]))
        shap_driver_cols = [
            group_names[i]
            for i in order[:TOP_GROUPS_PER_CLUSTER]
            if group_names[i] in crosstabs
        ]

        # Per-column stats for every candidate column at archetype level.
        col_stats: dict[str, dict] = {}
        col_total: dict[str, float] = {}  # the "sum" stat: drives the gate + sort
        col_enr_map: dict[str, dict] = {}
        for col in crosstab_cols:
            stats, enr_map, _ = _archetype_column_stats(
                crosstabs.get(col), c, min_group_rows
            )
            col_stats[col] = stats
            col_total[col] = stats["sum"]
            col_enr_map[col] = enr_map

        # Order (and therefore tree-branch) by each column's strongest single
        # value (total_enrichment.max), so the most concentrated driver sits at
        # the root. The main/not-important gate still uses the sum (equivalent:
        # sum>1 iff the column has any over-represented value).
        main_cols = sorted(
            [
                col
                for col in crosstab_cols
                if col_total[col] > importance_level
                and col not in driver_exclude_set
            ],
            key=lambda col: col_stats[col]["max"],
            reverse=True,
        )
        split_cols = main_cols[:max_split_cols]
        # not_important: SHAP-reason columns that did NOT become main drivers.
        not_important_cols = [
            col
            for col in shap_driver_cols
            if col_total.get(col, 0.0) <= importance_level
            and col not in driver_exclude_set
        ]

        archetype_groups = _split_archetype(
            rows_c=rows_c,
            archetype=c,
            split_cols=split_cols,
            main_cols=main_cols,
            not_important_cols=not_important_cols,
            col_stats=col_stats,
            col_enr_map=col_enr_map,
            numeric_cols=numeric_cols,
            min_group_rows=min_group_rows,
            n_errors=n_errors,
        )
        groups.extend(archetype_groups)

    # Keep the largest groups, renumber 1..N (biggest first), and remap rows.
    groups.sort(key=lambda d: d["size"], reverse=True)
    groups = groups[:max_groups]

    final_row_to_group: dict[int, int] = {}
    for gid, g in enumerate(groups, start=1):
        g["group"] = gid
        for ri in g.pop("_members"):
            final_row_to_group[int(ri)] = gid
    # Reorder keys so 'group' leads each object.
    ordered_groups = [
        {"group": g["group"], **{k: v for k, v in g.items() if k != "group"}}
        for g in groups
    ]

    n_assigned = len(final_row_to_group)
    group_assignments = pl.DataFrame(
        {
            "row_index": list(final_row_to_group.keys()),
            "ERROR_GROUP": list(final_row_to_group.values()),
        },
        schema={"row_index": pl.Int64, "ERROR_GROUP": pl.Int64},
    )

    # Per-group value->count breakdown for each "describe-only" column (e.g. the
    # raw RESTAURANT_BRAND_NAMES arrays), to hang under its driver node.
    describe_per_group = _describe_counts_per_group(
        valued, final_row_to_group, describe_under
    )

    report = {
        "scored_parquet": str(scored_path),
        "model_2_backend": backend,
        "n_errors": int(n_errors),
        "n_archetypes": int(n_clusters),
        "n_groups": len(ordered_groups),
        "n_rows_grouped": int(n_assigned),
        "n_rows_ungrouped": int(n_errors - n_assigned),
        "k_selection": k_selection,
        "min_group_rows": int(min_group_rows),
        "importance_level": float(importance_level),
        "error_direction_glossary": DIRECTION_GLOSSARY,
        "reading_guide": (
            "'groups' is a tree: it branches first by the dominant error direction "
            "(its 'error_direction_mix' shows the aggregated mix beneath it), then "
            "by the 1st 'main_driver' (column=value), then the 2nd, and so on — "
            "shared prefixes are merged, so each path down the tree spells out one "
            "error pattern. A node's 'total_enrichment' is its column's strength in "
            "the parent archetype as {max, mean, sum} over its over-represented "
            "values' enrichment; 'value_enrichment' is how over-represented that "
            "node's specific value is vs all errors (>1 = over-represented). A "
            "branch is cut at the first driver whose 'value_enrichment' falls below "
            f"the importance level ({importance_level:g}): that weak value and "
            "everything under it are hidden, so a group terminates at its last "
            "strong driver. Some "
            "driver nodes carry a 'descriptions' breakdown — value->count of a "
            "linked raw column over the node's rows (e.g. the actual brand-list "
            "arrays under a BRAND_COUNT node). A leaf (no 'drivers') is a final "
            "group: it carries its 'group' id, 'archetype', and 'not_important' "
            "(the model's other suspected columns that were not over-represented). "
            "Every node has a 'size'/'share'. Follow the highest 'value_enrichment' "
            "branches first."
        ),
        "groups": _groups_to_tree(
            ordered_groups,
            n_errors,
            describe_under,
            describe_per_group,
            importance_level,
        ),
    }
    return report, group_assignments


def _describe_counts_per_group(
    valued: pl.DataFrame,
    row_to_group: dict[int, int],
    describe_under: dict[str, str],
) -> dict[int, dict[str, dict]]:
    """For each describe column, count its values within each final group.

    Returns ``{group_id: {describe_col: {value: count}}}``. Counts use the same
    pre-binned cross-tab value (so a raw brand-list string is counted verbatim).
    """
    out: dict[int, dict[str, dict]] = {}
    if not describe_under or not row_to_group:
        return out
    gmap = pl.DataFrame(
        {
            "row_index": list(row_to_group.keys()),
            "__g": list(row_to_group.values()),
        },
        schema={"row_index": pl.Int64, "__g": pl.Int64},
    )
    joined = valued.join(gmap, on="row_index", how="inner")
    for describe_col in dict.fromkeys(describe_under):
        vc = _val_col(describe_col)
        if vc not in joined.columns:
            continue
        agg = joined.group_by(["__g", vc]).agg(pl.len().alias("n"))
        for r in agg.iter_rows(named=True):
            out.setdefault(int(r["__g"]), {}).setdefault(describe_col, {})[
                r[vc]
            ] = int(r["n"])
    return out


def _groups_to_tree(
    groups: list[dict],
    n_errors: int,
    describe_under: dict[str, str] | None = None,
    describe_per_group: dict[int, dict[str, dict]] | None = None,
    importance_level: float = 0.0,
) -> list[dict]:
    """Collapse the flat group list into a tree (trie) for compactness.

    Branch order: error direction -> 1st main driver (column=value) -> 2nd main
    driver -> ... . Groups sharing a prefix share branches; each leaf carries the
    final group id / archetype / not_important. Driver nodes named by a column in
    ``describe_under`` additionally get a ``descriptions`` breakdown of the linked
    raw column (e.g. the actual brand-list arrays under a BRAND_COUNT node).

    Each group's path is truncated at the first driver whose ``value_enrichment``
    is below ``importance_level`` — that driver (a non-important value) and every
    deeper branch are dropped, so the group terminates at its last strong driver.
    """
    describe_per_group = describe_per_group or {}
    # driver column -> the raw columns described beneath it.
    driver_to_describe: dict[str, list[str]] = {}
    for describe_col, driver_col in (describe_under or {}).items():
        driver_to_describe.setdefault(driver_col, []).append(describe_col)

    # Trie node: {"_size", "_mix", "_children", "_leaves", "_driver", "_describe"}.
    def new_node() -> dict:
        return {"_size": 0, "_mix": {}, "_children": {}, "_leaves": []}

    roots: dict[str, dict] = {}
    for g in groups:
        mix = g["error_direction_mix"]
        # Group by the *dominant* direction only — a small secondary direction
        # must not split two otherwise-identical own_not_captured branches.
        sig = _dominant_direction(mix)
        dnode = roots.setdefault(sig, new_node())
        dnode["_size"] += g["size"]
        for label, n in mix.items():
            dnode["_mix"][label] = dnode["_mix"].get(label, 0) + n

        node = dnode
        for drv in g["main_drivers"]:
            # Stop at the first non-important value: hide it and all lower branches.
            if drv["value_enrichment"] < importance_level:
                break
            key = (drv["column"], str(drv["value"]))
            child = node["_children"].get(key)
            if child is None:
                child = new_node()
                child["_driver"] = drv
                node["_children"][key] = child
            child["_size"] += g["size"]
            # Accumulate the described raw-column breakdown under its driver node.
            if drv["column"] in driver_to_describe:
                gdesc = describe_per_group.get(g["group"], {})
                node_desc = child.setdefault("_describe", {})
                for dcol in driver_to_describe[drv["column"]]:
                    counts = gdesc.get(dcol)
                    if not counts:
                        continue
                    acc = node_desc.setdefault(dcol, {})
                    for val, cnt in counts.items():
                        acc[val] = acc.get(val, 0) + cnt
            node = child
        node["_leaves"].append(g)

    return [
        _dir_node_to_json(dnode, n_errors)
        for _, dnode in sorted(
            roots.items(), key=lambda kv: kv[1]["_size"], reverse=True
        )
    ]


def _dominant_direction(mix: dict[str, int]) -> str:
    """The single most common error direction (ties broken by label name)."""
    if not mix:
        return "unknown"
    return max(mix.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _dir_node_to_json(node: dict, n_errors: int) -> dict:
    out = {
        "error_direction_mix": dict(
            sorted(node["_mix"].items(), key=lambda kv: kv[1], reverse=True)
        ),
        "size": int(node["_size"]),
        "share": round(float(node["_size"] / max(n_errors, 1)), 4),
    }
    _attach_children_and_leaves(out, node, n_errors)
    return out


def _driver_node_to_json(node: dict, n_errors: int) -> dict:
    drv = node["_driver"]
    out = {
        "column": drv["column"],
        "value": drv["value"],
        "total_enrichment": drv["total_enrichment"],
        "value_enrichment": drv["value_enrichment"],
        "size": int(node["_size"]),
        "share": round(float(node["_size"] / max(n_errors, 1)), 4),
    }
    if node.get("_describe"):
        out["descriptions"] = {
            dcol: _top_counts(counts, DESCRIBE_TOP_N)
            for dcol, counts in node["_describe"].items()
        }
    _attach_children_and_leaves(out, node, n_errors)
    return out


def _top_counts(counts: dict, top_n: int) -> dict:
    """value->count, most frequent first, capped at top_n (+ an '(other)' sum)."""
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    head = ordered[:top_n]
    out = {str(v): int(c) for v, c in head}
    rest = sum(c for _, c in ordered[top_n:])
    if rest:
        out["(other)"] = int(rest)
    return out


def _attach_children_and_leaves(out: dict, node: dict, n_errors: int) -> None:
    """Add nested 'drivers' (branches) and, for terminating groups, the leaf
    fields (group id / archetype / not_important)."""
    if node["_children"]:
        out["drivers"] = [
            _driver_node_to_json(child, n_errors)
            for _, child in sorted(
                node["_children"].items(),
                key=lambda kv: kv[1]["_size"],
                reverse=True,
            )
        ]
    leaves = node["_leaves"]
    if len(leaves) == 1 and not node["_children"]:
        g = leaves[0]
        out["group"] = g["group"]
        out["archetype"] = g["archetype"]
        if g["not_important"]:
            out["not_important"] = g["not_important"]
    elif leaves:
        # Rare: a group's path is a prefix of another's (mixed depths) — list them.
        out["groups"] = [
            {
                "group": g["group"],
                "archetype": g["archetype"],
                "size": int(g["size"]),
                "not_important": g["not_important"],
            }
            for g in leaves
        ]


def _split_archetype(
    *,
    rows_c: pl.DataFrame,
    archetype: int,
    split_cols: list[str],
    main_cols: list[str],
    not_important_cols: list[str],
    col_stats: dict[str, dict],
    col_enr_map: dict[str, dict],
    numeric_cols: set[str],
    min_group_rows: int,
    n_errors: int,
) -> list[dict]:
    """Split one archetype into value-resolved groups (>= min_group_rows each).

    ERROR_DIRECTION is always the first grouping key, so every emitted group is
    direction-pure (a group is never a mix of own_not_captured / label_missing /
    unexpected_own_flag). The main-driver values further subdivide within a
    direction.
    """
    zero_stats = {"max": 0.0, "mean": 0.0, "sum": 0.0}
    # not_important entries are archetype-level (constant across the split groups):
    # each suspected-but-flat column with its most prevalent value.
    not_important = []
    for col in not_important_cols:
        top = _archetype_top_value(rows_c, col)
        if top is None:
            continue
        not_important.append(
            {
                "column": col,
                "total_enrichment": col_stats.get(col, zero_stats),
                "value": top,
                "value_enrichment": round(
                    float(col_enr_map.get(col, {}).get(top, 0.0)), 4
                ),
            }
        )

    val_cols = [_val_col(col) for col in split_cols]
    grouped = (
        rows_c.group_by([DIRECTION_COL, *val_cols])
        .agg(
            pl.len().alias("size"),
            pl.col("row_index").alias("_members"),
        )
        .sort("size", descending=True)
    )
    # When there ARE main drivers, drop value-combos below the floor (noise);
    # with no driver to split on, keep each direction so its rows aren't lost.
    if val_cols:
        grouped = grouped.filter(pl.col("size") >= min_group_rows)

    out: list[dict] = []
    for r in grouped.iter_rows(named=True):
        direction = r[DIRECTION_COL]
        main_drivers = []
        for col in main_cols:
            # Value for split columns comes from the group key; non-split main
            # columns (beyond max_split_cols) fall back to their archetype top.
            if col in split_cols:
                value = r[_val_col(col)]
            else:
                value = _archetype_top_value(rows_c, col)
            if value is None:
                continue
            main_drivers.append(
                {
                    "column": col,
                    "total_enrichment": col_stats.get(col, zero_stats),
                    "value": value,
                    "value_enrichment": round(
                        float(col_enr_map.get(col, {}).get(value, 0.0)), 4
                    ),
                }
            )
        out.append(
            _make_group(
                archetype=archetype,
                size=int(r["size"]),
                main_drivers=main_drivers,
                not_important=not_important,
                direction_mix={direction: int(r["size"])},
                n_errors=n_errors,
                members=list(r["_members"]),
            )
        )
    return out


def _make_group(
    *,
    archetype: int,
    size: int,
    main_drivers: list[dict],
    not_important: list[dict],
    direction_mix: dict,
    n_errors: int,
    members: list,
) -> dict:
    return {
        "archetype": int(archetype),
        "size": int(size),
        "share": round(float(size / max(n_errors, 1)), 4),
        "error_direction_mix": direction_mix,
        "main_drivers": main_drivers,
        "not_important": not_important,
        "_members": members,
    }


def _archetype_top_value(rows_c: pl.DataFrame, col: str):
    """Most prevalent pre-binned value of `col` within an archetype (or None)."""
    vc = _val_col(col)
    if vc not in rows_c.columns:
        return None
    agg = (
        rows_c.group_by(vc)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    if agg.height == 0:
        return None
    return agg.get_column(vc)[0]
