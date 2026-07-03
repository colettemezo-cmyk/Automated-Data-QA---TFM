"""Stages 3a and 3b: heatmap plots driven off the parquet + per-column
embedding sidecars produced by `features.embeddings`.

Four outputs:

* `plot_ownership_heatmap`     - one IS_OWN_RESTAURANT-ratio heatmap
                                  (parent_app x country).
* `plot_correlation_heatmaps`  - one Pearson-correlation heatmap per
                                  (parent_app, country) subset, with
                                  PC1 scalars standing in for each
                                  text column. Reads `*_EMB_PC1.parquet`
                                  sidecars from disk.
* `plot_strategy_correlation_heatmaps` - same idea, but driven off a
                                  pre-built strategy-specific `model_input.parquet`
                                  (i.e. uses ALL of that strategy's reduction
                                  features, not just PC1). Designed for the
                                  `scripts/compare_reductions.py` outputs.
* `plot_strategy_block_correlation_heatmaps` - unified block view: each
                                  text column collapses back into a single
                                  row/col via the first canonical correlation
                                  of its K PCs against every other block.
                                  Answers "how strongly does FRANCHISE
                                  correlate with IS_OWN_RESTAURANT?" in
                                  one cell regardless of K. Same 17x17
                                  layout for every strategy, so heatmaps
                                  diff visually side-by-side.

All aggregation/pivot/correlation math stays inside polars so that
even the 8.2M-row subsets never round-trip through a pandas-style
contiguous float64 buffer (which is what used to OOM us).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from bootstrap import ensure_app_on_path
from common.config.columns import (
    CORR_EXCLUDE_COLS,
    EMBED_COLS,
    ML_TABULAR_COLS,
    TARGET_COL,
)
from workflows.training.config import EMBED_DIR, MIN_ROWS_FOR_CORR, PARQUET_PATH

ensure_app_on_path(__file__)


# ============================================================================
# Stage 3a: own-restaurant ratio heatmap (parent_app x country)
# ============================================================================
def plot_ownership_heatmap(
    parquet_path: Path = PARQUET_PATH,
    output_path: Path = Path("figures/ownership_ratio.png"),
    close_after_save: bool = True,
) -> None:
    """Heatmap of `IS_OWN_RESTAURANT` ratio per (PARENT_APP_NAME, COUNTRY).

    All aggregation and pivoting happens in polars (no pandas
    roundtrip); only the final 2D numeric grid is passed to seaborn as
    a numpy array.
    """
    dataset = pl.read_parquet(parquet_path)

    # Aggregate to one row per (app, country). Streaming engine keeps
    # memory bounded regardless of how big the underlying parquet is.
    ratio_df = (
        dataset.lazy()
        .group_by(["PARENT_APP_NAME", "APP_GOOGLE_COUNTRY_NAME"])
        .agg([
            pl.col("IS_OWN_RESTAURANT").cast(pl.Float64).mean().alias("own_ratio"),
            pl.len().alias("n_rows"),
        ])
        .sort(["PARENT_APP_NAME", "APP_GOOGLE_COUNTRY_NAME"])
        .collect(engine="streaming")
    )
    print(ratio_df)

    # Pivot in polars - rows = PARENT_APP_NAME, cols = APP_GOOGLE_COUNTRY_NAME.
    pivot = ratio_df.pivot(
        values="own_ratio",
        index="PARENT_APP_NAME",
        on="APP_GOOGLE_COUNTRY_NAME",
    ).sort("PARENT_APP_NAME")

    # Pull out labels + numeric grid for seaborn. Each column is
    # Float64 already so `to_numpy` is a cheap view.
    row_labels = pivot["PARENT_APP_NAME"].to_list()
    col_labels = [c for c in pivot.columns if c != "PARENT_APP_NAME"]
    grid = pivot.select(col_labels).to_numpy()  # (n_apps, n_countries)

    fig_h, ax_h = plt.subplots(
        figsize=(
            max(6, 0.7 * len(col_labels) + 4),
            max(4, 0.5 * len(row_labels) + 2),
        )
    )
    sns.heatmap(
        grid,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="viridis",
        cbar_kws={"label": "is_own_restaurant ratio"},
        ax=ax_h,
    )
    ax_h.set_title("Own-restaurant ratio by parent app x country")
    ax_h.set_xlabel("Country")
    ax_h.set_ylabel("Parent app")
    fig_h.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig_h.savefig(output_path, dpi=120, bbox_inches="tight")
    if close_after_save:
        plt.close(fig_h)


# ============================================================================
# Stage 3b: per-(app, country) correlation heatmaps with embeddings
# ============================================================================
def _to_numeric_for_corr(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce a polars dataframe so `pl.corr` can consume every column.

      * pl.Array / pl.List  -> drop (vector columns aren't scalar-correlatable)
      * Boolean             -> cast to Float32 (polars.corr requires numeric)
      * Other numeric       -> cast to Float32 (halves the working memory vs Float64;
                               pearson is well-conditioned, float32 precision is plenty)
      * Datetime-family     -> cast to Int64 (nanoseconds) then Float32
      * Anything else (str) -> dense rank-encode then Float32

    Why Float32 everywhere? The crash we're fixing here was pandas
    calling `.to_numpy(dtype=float64)` and allocating 1.16 GB in one
    shot for the iFood/Brazil subset. Going Float32-native halves the
    peak. (Note: polars-side pearson aggregates into Float64 internally
    anyway, so we lose nothing precision-wise.)

    In stage 3b we drop the original text columns BEFORE calling this,
    so the rank-encoding branch should rarely fire. It's kept as a
    safety net.
    """
    cast_exprs = []
    drop_cols = []
    for name, dtype in df.schema.items():
        if isinstance(dtype, (pl.List, pl.Array)):
            drop_cols.append(name)
        elif dtype == pl.Boolean:
            cast_exprs.append(pl.col(name).cast(pl.Float32).alias(name))
        elif dtype.is_numeric():
            cast_exprs.append(pl.col(name).cast(pl.Float32).alias(name))
        elif dtype in (pl.Datetime, pl.Date, pl.Time, pl.Duration):
            cast_exprs.append(pl.col(name).cast(pl.Int64).cast(pl.Float32).alias(name))
        else:
            cast_exprs.append(pl.col(name).rank("dense").cast(pl.Float32).alias(name))
    out = df
    if drop_cols:
        out = out.drop(drop_cols)
    if cast_exprs:
        out = out.with_columns(cast_exprs)
    return out


def _polars_corr_matrix(df: pl.DataFrame, cols: list) -> np.ndarray:
    """Pearson correlation matrix computed entirely inside polars.

    For each upper-triangle pair (i, j) we emit a
    `pl.corr(cols[i], cols[j])` aggregation. All 190 aggregations (for
    19 cols) live in a single `select(...)` and polars schedules them
    together, streaming over the columns. Memory usage is O(n_cols^2) -
    independent of n_rows.

    Why this exists: `pandas.DataFrame.corr()` calls
    `to_numpy(dtype=float64)` which interleaves every column into one
    contiguous Float64 buffer. At 8.2M rows x 19 cols that's 1.16 GB
    allocated in one shot, on top of everything else - which is exactly
    what OOMed the previous run. polars never materialises that buffer.
    """
    n = len(cols)
    aggs = []
    pair_indices = []
    for i in range(n):
        for j in range(i, n):
            name = f"__corr_{i}_{j}"
            pair_indices.append((i, j, name))
            aggs.append(pl.corr(cols[i], cols[j], method="pearson").alias(name))
    row = df.lazy().select(aggs).collect(engine="streaming").row(0, named=True)
    corr = np.full((n, n), np.nan, dtype=np.float64)
    for i, j, name in pair_indices:
        v = row[name]
        v = float("nan") if v is None else float(v)
        corr[i, j] = v
        corr[j, i] = v
    return corr


def plot_correlation_heatmaps(
    parquet_path: Path = PARQUET_PATH,
    embed_dir: Path = EMBED_DIR,
    text_cols: list = None,
    target_col: str = TARGET_COL,
    min_rows: int = MIN_ROWS_FOR_CORR,
    output_dir: Path = Path("figures/correlations"),
    close_after_save: bool = True,
) -> None:
    """Pearson correlation heatmaps per (PARENT_APP_NAME, COUNTRY).

    The original text columns listed in EMBED_COLS are REMOVED from
    the correlation matrix. In their place we use each column's PC1
    scalar (the projection of the embedding onto the column's first
    principal component) - read from the sidecars produced by Stage 2.

    Why PC1 and not the full embedding?
      Correlation needs scalar features. PC1 captures the direction of
      maximum variance in the embedding distribution, weighted by row
      frequency. It's the "best single number" summary of a text
      column's semantics under a linear/variance lens.

    Alternative scalars you could swap in here:
      * dCor (distance correlation) - non-linear, works directly on
        the full embedding vector. Needs `dcor` package. Heaviest
        option.
      * Canonical correlation of (embedding, target_indicator) -
        supervised, captures the linear direction most aligned with
        the target. Needs sklearn.
      * Linear-probe AUC - train a tiny logistic regression on the
        embedding, use AUC as the cell. Most informative but most code.
    """
    text_cols = text_cols if text_cols is not None else EMBED_COLS
    dataset = pl.read_parquet(parquet_path)
    text_cols_present = [c for c in text_cols if c in dataset.columns]

    # Attach each column's PC1 scalar sidecar as a new column.
    missing = []
    pc1_series_list = []
    for col in text_cols_present:
        pc1_path = embed_dir / f"{col}_EMB_PC1.parquet"
        if not pc1_path.exists():
            missing.append(col)
            continue
        pc1_series = pl.read_parquet(pc1_path).get_column(f"{col}_EMB_PC1")
        if len(pc1_series) != dataset.height:
            raise ValueError(
                f"PC1 sidecar for {col} has {len(pc1_series)} rows but "
                f"dataset has {dataset.height}. Re-run embed_text_columns(force=True)."
            )
        pc1_series_list.append(pc1_series)
    if missing:
        print(
            f"[stage3b] missing PC1 sidecars for {missing}. "
            f"Run embed_text_columns() first. These columns will be excluded."
        )
    dataset = dataset.with_columns(pc1_series_list)

    # Drop the originals. We keep PARENT_APP_NAME / APP_GOOGLE_COUNTRY_NAME
    # *temporarily* for the group-by, then drop them per-subset.
    grouping_cols = ["PARENT_APP_NAME", "APP_GOOGLE_COUNTRY_NAME"]
    # Always-exclude list: columns whose semantic content is captured
    # by derived features (e.g. EXECUTION_ID -> EXECUTION_ROW_COUNT,
    # PIPELINE_FLAG -> IS_ODM/IS_QCA). Leaving them in would just
    # rank-encode them into a noisy correlation column. See
    # `CORR_EXCLUDE_COLS` for the catalogue.
    cols_to_drop_in_corr = [
        c for c in text_cols_present + CORR_EXCLUDE_COLS if c not in grouping_cols
    ]

    combos = (
        dataset
        .select(grouping_cols)
        .unique()
        .sort(grouping_cols)
    )

    # Where to dump the per-combo PNGs. We always save (so you can
    # browse all 46 plots without piling up matplotlib figures in
    # memory), and optionally close each figure after saving.
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for parent_app, country in combos.iter_rows():
        subset_pl = dataset.filter(
            (pl.col("PARENT_APP_NAME") == parent_app)
            & (pl.col("APP_GOOGLE_COUNTRY_NAME") == country)
        )
        if subset_pl.height < min_rows:
            print(f"[skip corr] {parent_app} / {country}: only {subset_pl.height} rows")
            continue

        # Drop the text originals AND the grouping cols (which are
        # constant within this subset, so corr would be NaN for them).
        for_corr = subset_pl.drop(
            [c for c in cols_to_drop_in_corr + grouping_cols if c in subset_pl.columns]
        )
        for_corr = _to_numeric_for_corr(for_corr)

        # Target first so it lands at the top-left of the heatmap.
        cols_ordered = [target_col] + [c for c in for_corr.columns if c != target_col]

        # Polars-native Pearson matrix - no pandas, no float64
        # broadcast. This is the fix for the pandas to_numpy(float64)
        # OOM that was killing the iFood/Brazil subset (8.2M rows x
        # 19 cols).
        corr_np = _polars_corr_matrix(for_corr, cols_ordered)
        if corr_np.size == 0:
            continue

        n_cols = corr_np.shape[1]
        side = max(9, 0.55 * n_cols + 4)
        fig_c, ax_c = plt.subplots(figsize=(side, side))
        sns.heatmap(
            corr_np,
            xticklabels=cols_ordered,
            yticklabels=cols_ordered,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            vmin=-1,
            vmax=1,
            center=0,
            cmap="coolwarm",
            square=True,
            cbar_kws={"label": "Pearson correlation (text cols replaced by PC1)"},
            ax=ax_c,
        )
        ax_c.set_title(
            f"Correlation matrix - {parent_app} / {country}  (n={subset_pl.height})"
        )
        ax_c.tick_params(axis="x", rotation=45, labelsize=8)
        ax_c.tick_params(axis="y", labelsize=8)
        for label in ax_c.get_xticklabels():
            label.set_horizontalalignment("right")
        fig_c.tight_layout()

        # Save + (optionally) close. Saving sidesteps matplotlib's
        # "more than 20 figures opened" warning when you have ~46
        # combos. Sanitise the filename in case of slashes/spaces in
        # app/country names.
        def _safe(s: str) -> str:
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))
        if output_dir is not None:
            fig_path = output_dir / f"corr_{_safe(parent_app)}_{_safe(country)}.png"
            fig_c.savefig(fig_path, dpi=120, bbox_inches="tight")
        if close_after_save:
            plt.close(fig_c)


# ============================================================================
# Stage 3b (strategy-aware): correlations using a cached model_input.parquet
# ============================================================================
def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _exec_id_to_grouping(
    source_parquet: Path,
    grouping_cols: tuple[str, ...] = ("PARENT_APP_NAME", "APP_GOOGLE_COUNTRY_NAME"),
    group_col: str = "EXECUTION_ID",
) -> pl.DataFrame:
    """Many-to-one lookup `EXECUTION_ID -> (parent_app, country)`.

    Each `EXECUTION_ID` is a single crawl session for one (app, country),
    so we can recover grouping columns that the cached `model_input.parquet`
    deliberately drops (they're forbidden as features). Lazy-streamed so it
    works on the 11M-row corpus without materialising it.
    """
    cols = [group_col, *grouping_cols]
    return (
        pl.scan_parquet(source_parquet)
        .select(cols)
        .unique(subset=[group_col], keep="first")
        .collect(engine="streaming")
    )


def plot_strategy_correlation_heatmaps(
    cache_dir: Path,
    source_parquet: Path,
    output_dir: Path,
    *,
    target_col: str = TARGET_COL,
    grouping_cols: tuple[str, ...] = ("PARENT_APP_NAME", "APP_GOOGLE_COUNTRY_NAME"),
    min_rows: int = MIN_ROWS_FOR_CORR,
    top_combos: int | None = None,
    annot_threshold: int = 25,
    close_after_save: bool = True,
) -> dict[str, int]:
    """Per-(parent_app, country) Pearson heatmaps from a strategy's cache.

    Why this exists separate from `plot_correlation_heatmaps`:
      The non-strategy version reads `*_EMB_PC1.parquet` sidecars from
      `EMBED_DIR`, which only encode K=1 per column. With the new
      strategy-aware reduction (`top5`, `adaptive_*`, `raw`) the projected
      feature count varies per strategy AND per text column. The cached
      `model_input.parquet` already has those features fanned out into
      named columns, so we read them straight from the cache and never
      re-project.

    Output:
      `<output_dir>/corr_<parent_app>_<country>.png` for every combo with
      at least `min_rows` rows. When `top_combos` is set we only render
      the `top_combos` most-populous combos (keeps figure counts sane on
      the 130+ unique pairs).

    Annotation policy:
      Per-cell numeric labels are only useful up to ~25 features; beyond
      that they overlap. Above the threshold we drop labels and rely on
      the colormap pattern to surface block structure.

    Returns:
      `{"combos_seen": int, "combos_plotted": int, "combos_skipped": int}`
      so the driver script can print a one-line per-strategy summary.
    """
    cache_dir = Path(cache_dir)
    source_parquet = Path(source_parquet)
    output_dir = Path(output_dir)

    meta_path = cache_dir / "meta.json"
    model_path = cache_dir / "model_input.parquet"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"strategy cache incomplete at {cache_dir}: need meta.json + "
            f"model_input.parquet (run scripts/compare_reductions.py first)"
        )
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols: list[str] = list(meta.get("feature_columns", []))
    if not feature_cols:
        raise ValueError(f"meta.json at {meta_path} has no feature_columns")

    df = pl.read_parquet(model_path)
    if target_col not in df.columns:
        raise ValueError(
            f"target column {target_col!r} not in {model_path}; got {df.columns}"
        )

    # Attach grouping columns via EXECUTION_ID -> (app, country) lookup.
    lookup = _exec_id_to_grouping(source_parquet, grouping_cols=grouping_cols)
    missing_lookup = [c for c in grouping_cols if c not in lookup.columns]
    if missing_lookup:
        raise ValueError(
            f"source parquet {source_parquet} is missing grouping columns "
            f"{missing_lookup}; cannot subset by (app, country)"
        )
    df = df.join(lookup, on="EXECUTION_ID", how="left")

    combos = (
        df.group_by(list(grouping_cols))
        .agg(pl.len().alias("n_rows"))
        .filter(pl.col("n_rows") >= min_rows)
        .sort("n_rows", descending=True)
    )
    if top_combos is not None:
        combos = combos.head(top_combos)
    combos_seen = combos.height

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    skipped = 0

    # Order cols so target lands top-left and tabular features come before
    # the reduction features (matches the original version_zero layout).
    ordered_features = [target_col] + [c for c in feature_cols if c != target_col]
    n_feat = len(ordered_features)
    annotate = n_feat <= annot_threshold

    for row in combos.iter_rows(named=True):
        keys = {c: row[c] for c in grouping_cols}
        filter_expr = None
        for c, v in keys.items():
            cond = pl.col(c) == v
            filter_expr = cond if filter_expr is None else filter_expr & cond
        subset = df.filter(filter_expr).select(ordered_features)
        if subset.height < min_rows:
            skipped += 1
            continue

        numeric = _to_numeric_for_corr(subset)
        cols_ordered = [target_col] + [c for c in numeric.columns if c != target_col]
        corr_np = _polars_corr_matrix(numeric, cols_ordered)
        if corr_np.size == 0:
            skipped += 1
            continue

        n_cols = corr_np.shape[1]
        side = max(9, 0.40 * n_cols + 4) if not annotate else max(9, 0.55 * n_cols + 4)
        fig, ax = plt.subplots(figsize=(side, side))
        sns.heatmap(
            corr_np,
            xticklabels=cols_ordered,
            yticklabels=cols_ordered,
            annot=annotate,
            fmt=".2f" if annotate else "",
            annot_kws={"size": 7} if annotate else None,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="coolwarm",
            square=True,
            cbar_kws={"label": "Pearson correlation"},
            ax=ax,
        )
        title_keys = " / ".join(str(row[c]) for c in grouping_cols)
        ax.set_title(
            f"Correlation - {title_keys}  (n={row['n_rows']:,}, p={n_feat})"
        )
        tick_size = 8 if annotate else max(4, 8 - n_feat // 30)
        ax.tick_params(axis="x", rotation=45, labelsize=tick_size)
        ax.tick_params(axis="y", labelsize=tick_size)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        fig.tight_layout()

        suffix = "_".join(_safe_name(row[c]) for c in grouping_cols)
        fig.savefig(output_dir / f"corr_{suffix}.png", dpi=120, bbox_inches="tight")
        if close_after_save:
            plt.close(fig)
        plotted += 1

    return {
        "combos_seen": int(combos_seen),
        "combos_plotted": int(plotted),
        "combos_skipped": int(skipped),
    }


# ============================================================================
# Stage 3b (block view): collapse multi-PC reductions back to one row/col
# ============================================================================
_PC_FEATURE_RE = re.compile(r"^(?P<col>.+?)_EMB_PC\d+$")
_RAW_FEATURE_RE = re.compile(r"^(?P<col>.+?)_EMB_DIM\d{3}$")


def _block_for_feature(name: str) -> str:
    """Map a feature column name to its parent block name.

      `<COL>_EMB_PC<i>` or `<COL>_EMB_DIM<i>`  -> `<COL>`
      anything else (tabular scalar)           -> the feature name itself

    Used to reverse the strategy-aware fan-out: every K-PC reduction of
    a text column collapses back to that column's identity, so a 64-PC
    `FRANCHISE` block in `adaptive_0.95` shows up as one row/col labelled
    `FRANCHISE` in the heatmap.
    """
    m = _PC_FEATURE_RE.match(name) or _RAW_FEATURE_RE.match(name)
    return m.group("col") if m else name


def _ordered_blocks_from_features(
    feature_cols: list[str],
    tabular_order: list[str] = ML_TABULAR_COLS,
    text_order: list[str] = EMBED_COLS,
) -> list[tuple[str, list[str]]]:
    """Group + order feature columns into (block_name, [feature_names]).

    Ordering: tabular scalars first (in `tabular_order`), then text-column
    blocks (in `text_order`). Anything unrecognised lands at the tail to
    avoid silently dropping features.
    """
    by_block: dict[str, list[str]] = {}
    for name in feature_cols:
        by_block.setdefault(_block_for_feature(name), []).append(name)

    ordered: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for block in tabular_order:
        if block in by_block:
            ordered.append((block, by_block[block]))
            seen.add(block)
    for block in text_order:
        if block in by_block:
            ordered.append((block, by_block[block]))
            seen.add(block)
    for block, members in by_block.items():
        if block not in seen:
            ordered.append((block, members))
    return ordered


def _whiten_block(matrix: np.ndarray) -> np.ndarray | None:
    """Centre + economy-QR a (n, k) block into an orthonormal (n, r) basis.

    Returns None if the block has zero rank in this subset (constant column,
    all-NaN, or n < 2). Callers treat None as "no association possible" and
    drop the block from the displayed matrix.

    Numerical hardening (vs the initial version):
      * Force float64 — the legacy `_to_numeric_for_corr` ran in float32 to
        halve memory on the 8M-row global matrix. For per-(app, country)
        block CCA the matrices are tiny and float32 datetime nanoseconds
        get crushed to 7 significant digits, which spuriously collinearised
        EXECUTION_ROW_COUNT vs START_EXECUTION_DATETIME at r=1.0.
      * Drop NaN-only columns before centering (otherwise the column mean
        propagates NaN through the whole block).
      * Truncate Q to numerical rank using the standard `R`-pivot tolerance,
        scaled by `max(n, k) * eps * max(|diag(R)|)`.
    """
    if matrix.size == 0:
        return None
    arr = np.asarray(matrix, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return None
    # If any column is fully non-finite, drop it (mean would be NaN and
    # contaminate the rest). Don't impute — the caller already filtered
    # NaN rows globally, so a fully NaN column here means the feature is
    # genuinely undefined within this subset.
    col_has_data = finite_mask.any(axis=0)
    if not col_has_data.all():
        arr = arr[:, col_has_data]
        if arr.shape[1] == 0:
            return None
    means = arr.mean(axis=0)
    if not np.isfinite(means).all():
        return None
    centred = arr - means
    col_std = centred.std(axis=0)
    # Two-tier "is this column actually varying?" filter:
    #   * absolute: drop true constants (std == 0).
    #   * relative: drop columns whose standard deviation is below the
    #     float32 ULP of the column's magnitude. Without this, a
    #     "logically constant" column (e.g. APP_GOOGLE_COUNTRY_NAME_EMB_PC1
    #     within a single-country subset) that picked up ~1e-8 of float32
    #     projection noise sneaks through and trivially correlates with
    #     ANY other near-constant column at r=1.
    col_max_abs = np.abs(arr).max(axis=0)
    rel_std = col_std / np.maximum(col_max_abs, 1e-300)
    keep = (
        np.isfinite(col_std)
        & (col_std > 1e-12)
        & (rel_std > 1e-7)
    )
    if not keep.any():
        return None
    centred = centred[:, keep]
    if centred.shape[0] < 2:
        return None
    q, r = np.linalg.qr(centred, mode="reduced")
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        return None
    # LAPACK convention for numerical rank: tolerance scales with the
    # largest pivot and the float64 machine epsilon * max(n, k).
    eps = np.finfo(np.float64).eps
    tol = float(max(centred.shape)) * eps * float(diag.max())
    rank_mask = diag > max(tol, 1e-10)
    if not rank_mask.any():
        return None
    return q[:, rank_mask]


def _first_canonical_corr(qx: np.ndarray, qy: np.ndarray) -> float:
    """Largest canonical correlation between two pre-whitened bases."""
    if qx is None or qy is None or qx.size == 0 or qy.size == 0:
        return 0.0
    sv = np.linalg.svd(qx.T @ qy, compute_uv=False)
    if sv.size == 0:
        return 0.0
    return float(min(1.0, max(0.0, sv[0])))


def _coerce_block_matrix_float64(
    df_subset: pl.DataFrame, cols: list[str]
) -> np.ndarray:
    """Convert `df_subset[cols]` to a float64 numpy matrix safe for QR.

    Boolean -> 0/1, numeric -> float64 cast, datetime -> milliseconds since
    the subset's minimum (THEN cast to float64 — staying inside int64 for
    the subtraction keeps nanosecond precision; the divide-by-1e6 puts us
    in millisecond units that comfortably fit float64 mantissa). Anything
    non-numeric (shouldn't appear here since the cache holds only numerics
    + EXECUTION_ID + TARGET) gets dense-rank-encoded as a last resort.

    All math runs in float64 — the previous float32 path crushed
    nanosecond datetimes to 7 significant digits, which made tiny
    centred residuals collinearise with EXECUTION_ROW_COUNT at r=1.0.
    """
    exprs: list[pl.Expr] = []
    for name in cols:
        dtype = df_subset.schema[name]
        if dtype == pl.Boolean:
            exprs.append(pl.col(name).cast(pl.Float64).alias(name))
        elif dtype.is_numeric():
            exprs.append(pl.col(name).cast(pl.Float64).alias(name))
        elif dtype in (pl.Datetime, pl.Date, pl.Time, pl.Duration):
            ms_since_min = (
                (pl.col(name).cast(pl.Int64) - pl.col(name).cast(pl.Int64).min())
                .cast(pl.Float64)
                / 1.0e6
            )
            exprs.append(ms_since_min.alias(name))
        else:
            exprs.append(pl.col(name).rank("dense").cast(pl.Float64).alias(name))
    return df_subset.select(exprs).to_numpy()


def _block_correlation_matrix(
    df_subset: pl.DataFrame,
    blocks: list[tuple[str, list[str]]],
    target_col: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute the canonical-correlation matrix for ACTIVE blocks only.

    The returned matrix excludes degenerate blocks (constant in subset,
    all-NaN, etc.), so its diagonal is always 1.0 for every visible
    row/col. The first row/col is the target (`names[0] == target_col`).

    Cells where the joint observation count cannot support `K_a + K_b`
    free parameters are emitted as `np.nan` (the joint design matrix is
    rank-deficient, which would force a spurious canonical correlation
    of 1.0). The bound is computed lazily: a cheap upper bound from
    per-block uniqueness short-circuits most cells, and the exact joint
    uniqueness is only counted when the cheap bound is borderline.

    Returns `(matrix, names, sign_against_target)`:
      * `matrix[i, j]`: first canonical correlation in `[0, 1]` between
        block i and block j. `np.nan` for overfit cells.
      * `names`: human-readable block names of ONLY the active blocks.
      * `sign_against_target[i]`: sign of the strongest individual
        feature-vs-target Pearson r in block i (used when `signed=True`
        in the caller). +1 for positive, -1 for negative, 0 when no signal.
    """
    # 1. Build the per-block column list, including the target as its own
    #    1-D block at position 0 so it shares the same NaN-drop, casting,
    #    and whitening pipeline as everything else.
    target_block = (target_col, [target_col])
    indexed_blocks: list[tuple[str, list[str]]] = [target_block, *blocks]
    needed_cols = [m for _, members in indexed_blocks for m in members]
    # Deduplicate while preserving order (target_col may also appear later).
    seen: set[str] = set()
    unique_cols: list[str] = []
    for c in needed_cols:
        if c not in seen and c in df_subset.columns:
            seen.add(c)
            unique_cols.append(c)

    # 2. Cast to float64 with datetime-safe scaling, then drop any rows
    #    that contain a NaN in ANY of the selected columns. Whitening per
    #    block would otherwise see different effective row counts per
    #    block which makes pairwise CCA undefined.
    numeric_np = _coerce_block_matrix_float64(df_subset, unique_cols)
    valid_mask = np.isfinite(numeric_np).all(axis=1)
    if not valid_mask.any():
        return (
            np.zeros((0, 0), dtype=np.float64),
            [],
            np.zeros(0, dtype=np.int8),
        )
    numeric_np = numeric_np[valid_mask, :]
    n_effective = int(numeric_np.shape[0])
    col_to_idx = {c: i for i, c in enumerate(unique_cols)}

    # 3. Whiten each block independently; drop blocks that come back None.
    active_names: list[str] = []
    active_matrices: list[np.ndarray] = []
    active_whitened: list[np.ndarray] = []
    active_K: list[int] = []
    for block_name, members in indexed_blocks:
        idxs = [col_to_idx[m] for m in members if m in col_to_idx]
        if not idxs:
            continue
        mat = numeric_np[:, idxs]
        q = _whiten_block(mat)
        if q is None:
            continue
        active_names.append(block_name)
        active_matrices.append(mat)
        active_whitened.append(q)
        active_K.append(q.shape[1])

    if not active_whitened:
        return (
            np.zeros((0, 0), dtype=np.float64),
            [],
            np.zeros(0, dtype=np.int8),
        )

    # 4. Build the matrix on active blocks only; diagonal is therefore
    #    always 1.0 by construction. Overfit cells become NaN so the
    #    caller can mask them visually.
    n = len(active_whitened)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)

    # Per-block effective sample size = number of unique rows when the
    # block's columns are read together. For text embeddings that repeat
    # across rows (e.g. "Pizza Hut" appearing in 800 sections), this is
    # MUCH smaller than `n_effective` and bounds the legitimate dimensionality
    # of the joint regression / CCA fit.
    #
    # Float-rounding before np.unique coalesces float32 ULP noise so
    # that logically-identical rows count as one. 6 significant digits
    # is well above float32 ULP and well below any real per-row variation
    # the embedding pipeline would produce.
    def _unique_rows(mat: np.ndarray) -> int:
        if mat.size == 0:
            return 0
        scale = float(np.abs(mat).max())
        if scale <= 0:
            return 1
        rounded = np.round(mat / (scale * 1.0e-6)).astype(np.int64)
        if rounded.ndim == 1 or rounded.shape[1] == 1:
            return int(np.unique(rounded.ravel()).size)
        return int(np.unique(rounded, axis=0).shape[0])

    n_unique_per_block = [_unique_rows(mat) for mat in active_matrices]

    for i in range(n):
        for j in range(i + 1, n):
            k_total = active_K[i] + active_K[j]
            # Absolute floor: any regression / CCA fit needs at least
            # `K_total + 1` distinct joint observations, otherwise the
            # design matrix is rank-deficient and r is algebraically
            # pegged at 1.0. For 1-D-vs-1-D we additionally require >= 3
            # so Pearson r isn't trivially determined by 2 collinear
            # points (the geometric degeneracy of EXECUTION_ROW_COUNT vs
            # START_EXECUTION_DATETIME in a subset with only 2 executions).
            required = max(k_total + 1, 3)

            # Cheap upper bound: joint uniqueness can't exceed the
            # smaller of n_effective or the product of per-block
            # uniques. If even this upper bound is below `required`,
            # short-circuit to NaN without an expensive np.unique call.
            n_eff_ub = min(
                n_effective,
                n_unique_per_block[i] * n_unique_per_block[j],
            )
            if n_eff_ub < required:
                continue  # leave as NaN

            # The min-per-block bound is tighter than the product, but
            # can drastically understate joint uniqueness when one side
            # is binary against a rich continuous block (e.g. binary
            # IS_OWN_RESTAURANT vs 64-D FRANCHISE in iFood/Brazil:
            # min=2, but the joint has up to 2*450=900 distinct tuples).
            # Only resort to the exact joint uniqueness check when the
            # cheap min-check would otherwise mask the cell.
            n_eff_min = min(n_unique_per_block[i], n_unique_per_block[j])
            if n_eff_min < required:
                joint = np.column_stack(
                    [active_matrices[i], active_matrices[j]]
                )
                n_eff_joint = _unique_rows(joint)
                if n_eff_joint < required:
                    continue  # leave as NaN

            cc = _first_canonical_corr(active_whitened[i], active_whitened[j])
            matrix[i, j] = cc
            matrix[j, i] = cc

    # 5. Per-block sign vs target — uses the (NaN-filtered) target column
    #    so signs agree with what the matrix actually computed.
    target_idx = active_names.index(target_col) if target_col in active_names else None
    sign_against_target = np.zeros(n, dtype=np.int8)
    if target_idx is not None:
        target_col_vec = active_matrices[target_idx][:, 0]
        target_c = target_col_vec - target_col_vec.mean()
        tn = float(np.linalg.norm(target_c))
        if tn > 1e-12:
            sign_against_target[target_idx] = 1
            for k, mat in enumerate(active_matrices):
                if k == target_idx:
                    continue
                best_r = 0.0
                for jcol in range(mat.shape[1]):
                    col = mat[:, jcol] - mat[:, jcol].mean()
                    norm = float(np.linalg.norm(col))
                    if norm <= 1e-12:
                        continue
                    r = float((col @ target_c) / (norm * tn))
                    if abs(r) > abs(best_r):
                        best_r = r
                sign_against_target[k] = (
                    1 if best_r > 0 else (-1 if best_r < 0 else 0)
                )
    return matrix, active_names, sign_against_target


def plot_strategy_block_correlation_heatmaps(
    cache_dir: Path,
    source_parquet: Path,
    output_dir: Path,
    *,
    target_col: str = TARGET_COL,
    grouping_cols: tuple[str, ...] = ("PARENT_APP_NAME", "APP_GOOGLE_COUNTRY_NAME"),
    min_rows: int = MIN_ROWS_FOR_CORR,
    top_combos: int | None = None,
    signed: bool = False,
    close_after_save: bool = True,
) -> dict[str, int]:
    """Per-(parent_app, country) BLOCK correlation heatmaps for one strategy.

    Each text column's K reduction PCs collapse back into one row/column
    via the first canonical correlation between that block and every other
    block. Result is a 17 x 17 matrix (1 target + 9 tabular + 7 text)
    regardless of strategy, so heatmaps line up side-by-side and you can
    visually answer "did adaptive_0.95 actually pick up more
    FRANCHISE <-> target signal than pc1?".

    Math (justification for the choice):
      For two centred matrices X (n, p) and Y (n, q), the first canonical
      correlation is the supremum over (a, b) of corr(Xa, Yb) — i.e. the
      strongest possible linear association between any 1-D projection of
      X and any 1-D projection of Y. We compute it via economy QR of each
      block followed by SVD of Q_x^T Q_y; the top singular value is the
      answer. For K=1 blocks this reduces to |Pearson r|, so the `pc1`
      strategy's block heatmap is exactly the version_zero heatmap in
      absolute value.

    Sign handling:
      Canonical correlation is non-negative by construction. With
      `signed=False` (default) cells live in [0, 1] with a sequential
      colormap. With `signed=True` we attach the sign of the strongest
      individual feature-vs-target Pearson r within each block — useful
      for keeping the +/- semantics of the version_zero heatmap.

    Outputs:
      `<output_dir>/block_corr_<parent_app>_<country>.png` for every
      (app, country) combo with at least `min_rows` rows.

    Returns:
      `{"combos_seen": int, "combos_plotted": int, "combos_skipped": int}`.
    """
    cache_dir = Path(cache_dir)
    source_parquet = Path(source_parquet)
    output_dir = Path(output_dir)

    meta_path = cache_dir / "meta.json"
    model_path = cache_dir / "model_input.parquet"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"strategy cache incomplete at {cache_dir}: need meta.json + "
            f"model_input.parquet (run scripts/compare_reductions.py first)"
        )
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols: list[str] = list(meta.get("feature_columns", []))
    if not feature_cols:
        raise ValueError(f"meta.json at {meta_path} has no feature_columns")

    blocks = _ordered_blocks_from_features(feature_cols)

    df = pl.read_parquet(model_path)
    if target_col not in df.columns:
        raise ValueError(
            f"target column {target_col!r} not in {model_path}; got {df.columns}"
        )

    lookup = _exec_id_to_grouping(source_parquet, grouping_cols=grouping_cols)
    missing_lookup = [c for c in grouping_cols if c not in lookup.columns]
    if missing_lookup:
        raise ValueError(
            f"source parquet {source_parquet} is missing grouping columns "
            f"{missing_lookup}; cannot subset by (app, country)"
        )
    df = df.join(lookup, on="EXECUTION_ID", how="left")

    combos = (
        df.group_by(list(grouping_cols))
        .agg(pl.len().alias("n_rows"))
        .filter(pl.col("n_rows") >= min_rows)
        .sort("n_rows", descending=True)
    )
    if top_combos is not None:
        combos = combos.head(top_combos)
    combos_seen = combos.height

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    skipped = 0

    for row in combos.iter_rows(named=True):
        filter_expr = None
        for c in grouping_cols:
            cond = pl.col(c) == row[c]
            filter_expr = cond if filter_expr is None else filter_expr & cond
        subset_cols = [target_col, *feature_cols]
        subset = df.filter(filter_expr).select(subset_cols)
        if subset.height < min_rows:
            skipped += 1
            continue

        matrix, names, signs = _block_correlation_matrix(
            subset, blocks, target_col=target_col
        )
        if matrix.size == 0 or target_col not in names:
            skipped += 1
            continue

        # Move the target to position 0 for the heatmap so the top row
        # is always the "feature -> target" strip the user reads first.
        # _block_correlation_matrix returns active blocks in whatever
        # order they survived the degeneracy drop, so re-order here.
        target_idx = names.index(target_col)
        order = [target_idx] + [i for i in range(len(names)) if i != target_idx]
        matrix = matrix[np.ix_(order, order)]
        names = [names[i] for i in order]
        signs = signs[order]

        if signed:
            # Apply target-sign to the top row + left column; everything
            # else stays unsigned (canonical correlation has no native
            # sign for multi-D vs multi-D).
            display = matrix.copy()
            for k in range(len(names)):
                s = float(signs[k])
                if s == 0.0:
                    continue
                display[0, k] *= s
                display[k, 0] *= s
            vmin, vmax, center, cmap = -1.0, 1.0, 0.0, "coolwarm"
            cbar_label = "First canonical correlation (signed vs target)"
        else:
            display = matrix
            vmin, vmax, center, cmap = 0.0, 1.0, None, "viridis"
            cbar_label = "First canonical correlation"

        # Mask NaN cells (overfit-guarded) so they render as the cmap
        # bad colour rather than getting clipped to 0.
        masked = np.ma.masked_invalid(display)

        n = display.shape[0]
        side = max(8, 0.55 * n + 4)
        fig, ax = plt.subplots(figsize=(side, side))
        sns.heatmap(
            masked,
            xticklabels=names,
            yticklabels=names,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            vmin=vmin,
            vmax=vmax,
            center=center,
            cmap=cmap,
            square=True,
            cbar_kws={"label": cbar_label},
            ax=ax,
        )
        title_keys = " / ".join(str(row[c]) for c in grouping_cols)
        n_features = len(feature_cols)
        n_total_blocks = len(blocks) + 1
        dropped = n_total_blocks - n
        title = (
            f"Block correlation - {title_keys}  "
            f"(n={row['n_rows']:,}, blocks={n}/{n_total_blocks}, "
            f"features={n_features})"
        )
        if dropped:
            title += f"  [{dropped} constant-in-subset block(s) hidden]"
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        fig.tight_layout()

        suffix = "_".join(_safe_name(row[c]) for c in grouping_cols)
        fig.savefig(
            output_dir / f"block_corr_{suffix}.png", dpi=120, bbox_inches="tight"
        )
        if close_after_save:
            plt.close(fig)
        plotted += 1

    return {
        "combos_seen": int(combos_seen),
        "combos_plotted": int(plotted),
        "combos_skipped": int(skipped),
    }
