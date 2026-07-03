"""SHAP explanations for model_2 (the IF_ERROR classifier).

Runs *after* model_2 scoring in the QA pipeline. By default it explains the
**whole** scored dataset: SHAP values are computed in row-chunks (so peak
memory is bounded no matter how large the file is) and aggregated as we go.

Grouped by source column
------------------------
Each embedding column is reduced to several principal components
(`FRANCHISE_EMB_PC1..PC5`, ...). An individual PC carries no business meaning,
so the *headline* importance is reported per **source column**: tabular columns
map to themselves, and all of a column's PCs are collapsed into one number.

Because SHAP values are additive, the contribution of a column on a given row
is the **sum of the signed SHAP values** of its PCs for that row; we then take
the mean absolute of that per-row sum. (Summing the per-PC |SHAP| instead would
double-count PCs whose effects partly cancel.)

Outputs land under `data/ml/model_2/shap/<scored_stem>/`:
  * shap_grouped_importance.csv        — feature_group, mean_abs_shap (desc).
                                          THE headline file (per source column).
  * shap_grouped_bar.png               — same, as a bar chart.
  * shap_feature_importance.csv        — per-PC breakdown (detailed), for when
                                          you want to inspect a single component.
  * shap_summary_beeswarm.png          — per-PC beeswarm over a capped subset
                                          (a scatter can't legibly draw millions
                                          of points; importances above are exact).
  * shap_flagged_top_contributors.parquet — for every row model_2 flagged
                                          (PRED_IF_ERROR=1), the top-k source
                                          columns driving the flag (grouped).
  * shap_meta.json                     — counts, threshold, config.

Pass `sample_size` (CLI `--shap-sample N`) to instead run on N random rows for
a fast approximate summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bootstrap import ensure_app_on_path
from common.pipeline_timing import step as pipeline_step
from workflows.qa.config import SHAP_BEESWARM_MAX, SHAP_CHUNK_SIZE, SHAP_DIR
from workflows.qa.model_2 import Model2ScoreResult

ensure_app_on_path(__file__)

TOP_K_PER_ROW = 8
FLAGGED_PARQUET = "shap_flagged_top_contributors.parquet"
# Marker that identifies a reduced embedding feature (e.g. FRANCHISE_EMB_PC2).
# Everything before it is the source column the feature belongs to.
_EMB_MARKER = "_EMB_"


@dataclass
class ShapResult:
    out_dir: Path
    n_processed: int
    n_flagged: int
    n_beeswarm: int


def _group_name(feature: str) -> str:
    """Map a feature to its source column (embedding PCs -> base column)."""
    if _EMB_MARKER in feature:
        return feature.split(_EMB_MARKER)[0]
    return feature


def _build_feature_groups(
    feature_columns: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Group features by source column.

    Returns
    -------
    group_names : list[str]
        Unique source columns, in first-seen order.
    onehot : (n_features, n_groups) float32
        Membership matrix; `shap_values @ onehot` sums PCs into their column.
    single_feat_idx : (n_groups,) int64
        For groups backed by exactly one feature (all tabular columns), the
        index of that feature; -1 for multi-feature (embedding) groups. Lets us
        attach a meaningful raw value to single-feature groups only.
    """
    group_names: list[str] = []
    group_index: dict[str, int] = {}
    assign: list[int] = []
    for f in feature_columns:
        g = _group_name(f)
        if g not in group_index:
            group_index[g] = len(group_names)
            group_names.append(g)
        assign.append(group_index[g])

    n_features = len(feature_columns)
    n_groups = len(group_names)
    onehot = np.zeros((n_features, n_groups), dtype=np.float32)
    onehot[np.arange(n_features), assign] = 1.0

    single_feat_idx = np.full(n_groups, -1, dtype=np.int64)
    for gi in range(n_groups):
        members = np.flatnonzero(onehot[:, gi] > 0)
        if members.size == 1:
            single_feat_idx[gi] = int(members[0])
    return group_names, onehot, single_feat_idx


def _shap_values_positive_class(explainer, X: np.ndarray) -> np.ndarray:
    """Return a (n_rows, n_features) SHAP matrix for the positive class.

    `shap.TreeExplainer.shap_values` returns slightly different shapes across
    versions/backends (a single 2D array for XGBoost binary, a list of two
    arrays for some LightGBM builds, or a 3D array). Normalise them all.
    """
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def run_shap_evaluation(
    result: Model2ScoreResult,
    scored_path: Path,
    *,
    shap_dir: Path = SHAP_DIR,
    sample_size: int | None = None,
    chunk_size: int = SHAP_CHUNK_SIZE,
    beeswarm_max: int = SHAP_BEESWARM_MAX,
    random_state: int = 42,
    timer=None,
) -> ShapResult:
    """Compute + persist SHAP explanations for a model_2-scored parquet.

    sample_size=None -> explain the whole dataset (exact global importance and
    every flagged row). A positive value runs on that many random rows instead.
    """
    import shap

    scored_path = Path(scored_path)
    stem = scored_path.name.removesuffix(".parquet").removesuffix(".scored")
    out_dir = Path(shap_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    X = result.X
    feature_columns = result.feature_columns
    n_features = len(feature_columns)
    rng = np.random.default_rng(random_state)

    group_names, onehot, single_feat_idx = _build_feature_groups(feature_columns)
    n_groups = len(group_names)
    group_arr = np.asarray(group_names, dtype=object)

    if sample_size is not None and sample_size < X.shape[0]:
        idx = np.sort(rng.choice(X.shape[0], size=int(sample_size), replace=False))
        mode = "sample"
    else:
        idx = np.arange(X.shape[0])
        mode = "full"

    explainer = shap.TreeExplainer(result.classifier)

    sum_abs_feat = np.zeros(n_features, dtype=np.float64)
    sum_abs_group = np.zeros(n_groups, dtype=np.float64)
    n_processed = 0
    n_flagged = 0
    bee_shap: list[np.ndarray] = []
    bee_feat: list[np.ndarray] = []
    n_beeswarm = 0
    k = min(TOP_K_PER_ROW, n_groups)
    pred = result.pred.astype(bool)
    proba = result.proba

    import pyarrow as pa
    import pyarrow.parquet as pq

    flagged_path = out_dir / FLAGGED_PARQUET
    flagged_schema = pa.schema(
        [
            ("row_index", pa.int64()),
            ("proba_if_error", pa.float32()),
            ("rank", pa.int16()),
            ("feature_group", pa.string()),
            ("shap_value", pa.float32()),
            ("feature_value", pa.float32()),
        ]
    )
    writer = pq.ParquetWriter(flagged_path, flagged_schema)

    n_total = idx.size
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    try:
        with pipeline_step(
            "qa.5a.shap_values",
            f"SHAP values over {mode} dataset ({n_total:,} rows, {n_chunks} chunks)",
            "shap.TreeExplainer.shap_values",
            timer=timer,
        ):
            for start in range(0, n_total, chunk_size):
                chunk_idx = idx[start : start + chunk_size]
                Xc = X[chunk_idx]
                sv = _shap_values_positive_class(explainer, Xc)
                # Per-row grouped (signed) SHAP: sum PCs into their column.
                sv_group = sv @ onehot

                sum_abs_feat += np.abs(sv).sum(axis=0)
                sum_abs_group += np.abs(sv_group).sum(axis=0)
                n_processed += Xc.shape[0]

                if n_beeswarm < beeswarm_max:
                    take = min(beeswarm_max - n_beeswarm, Xc.shape[0])
                    bee_shap.append(sv[:take].astype(np.float32, copy=False))
                    bee_feat.append(Xc[:take].astype(np.float32, copy=False))
                    n_beeswarm += take

                fl_local = np.flatnonzero(pred[chunk_idx])
                if fl_local.size:
                    _write_flagged_chunk(
                        writer,
                        flagged_schema,
                        sv_group[fl_local],
                        Xc[fl_local],
                        chunk_idx[fl_local],
                        proba,
                        group_arr,
                        single_feat_idx,
                        k,
                    )
                    n_flagged += int(fl_local.size)
    finally:
        writer.close()
    if n_flagged == 0:
        flagged_path.unlink(missing_ok=True)

    mean_abs_group = sum_abs_group / max(n_processed, 1)
    mean_abs_feat = sum_abs_feat / max(n_processed, 1)

    g_order = np.argsort(mean_abs_group)[::-1]
    (out_dir / "shap_grouped_importance.csv").write_text(
        "feature_group,mean_abs_shap\n"
        + "\n".join(
            f"{group_names[i]},{float(mean_abs_group[i]):.8f}" for i in g_order
        )
        + "\n",
        encoding="utf-8",
    )
    f_order = np.argsort(mean_abs_feat)[::-1]
    (out_dir / "shap_feature_importance.csv").write_text(
        "feature,feature_group,mean_abs_shap\n"
        + "\n".join(
            f"{feature_columns[i]},{_group_name(feature_columns[i])},"
            f"{float(mean_abs_feat[i]):.8f}"
            for i in f_order
        )
        + "\n",
        encoding="utf-8",
    )

    with pipeline_step(
        "qa.5b.shap_plots",
        "Render SHAP summary plots",
        "shap.summary_plot",
        timer=timer,
    ):
        _render_bar_plot(
            mean_abs_group,
            group_names,
            out_dir,
            fname="shap_grouped_bar.png",
            title="model_2 feature importance (grouped by source column)",
        )
        bee_shap_arr = np.vstack(bee_shap) if bee_shap else np.empty((0, n_features))
        bee_feat_arr = np.vstack(bee_feat) if bee_feat else np.empty((0, n_features))
        _render_beeswarm(bee_shap_arr, bee_feat_arr, feature_columns, out_dir)

    meta = {
        "scored_parquet": str(scored_path),
        "model_2_backend": result.manifest.get("backend"),
        "decision_threshold": result.decision_threshold,
        "mode": mode,
        "n_rows_total": int(X.shape[0]),
        "n_rows_processed": int(n_processed),
        "n_flagged": int(n_flagged),
        "n_beeswarm_sample": int(n_beeswarm),
        "top_k_per_flagged_row": k,
        "chunk_size": int(chunk_size),
        "beeswarm_max": int(beeswarm_max),
        "sample_size": sample_size,
        "random_state": int(random_state),
        "feature_columns": feature_columns,
        "feature_groups": group_names,
        "grouping": "embedding PCs summed (signed) into their source column",
        "flagged_parquet": FLAGGED_PARQUET if n_flagged else None,
    }
    (out_dir / "shap_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(
        f"[qa/shap] wrote SHAP explanations -> {out_dir} "
        f"(mode={mode}, processed {n_processed:,} rows, "
        f"flagged {n_flagged:,}, {n_groups} feature groups, "
        f"beeswarm sample {n_beeswarm:,}).",
        flush=True,
    )
    return ShapResult(
        out_dir=out_dir,
        n_processed=int(n_processed),
        n_flagged=int(n_flagged),
        n_beeswarm=int(n_beeswarm),
    )


def _write_flagged_chunk(
    writer,
    schema,
    sv_group: np.ndarray,
    x_fl: np.ndarray,
    global_rows: np.ndarray,
    proba: np.ndarray | None,
    group_arr: np.ndarray,
    single_feat_idx: np.ndarray,
    k: int,
) -> None:
    """Append top-k source-column contributors for a chunk's flagged rows.

    `sv_group` is already the per-row grouped (signed) SHAP matrix
    (n_flagged, n_groups). For single-feature (tabular) groups we attach the raw
    feature value; embedding groups get NaN since no single value is meaningful.
    """
    import pyarrow as pa

    m = sv_group.shape[0]
    order = np.argsort(-np.abs(sv_group), axis=1)[:, :k]
    rows_rep = np.repeat(global_rows.astype(np.int64), k)
    ranks = np.tile(np.arange(1, k + 1, dtype=np.int16), m)
    grp_idx = order.reshape(-1)
    grp_names = group_arr[grp_idx]
    shap_vals = np.take_along_axis(sv_group, order, axis=1).reshape(-1)

    if proba is not None:
        proba_rep = np.repeat(proba[global_rows].astype(np.float32), k)
    else:
        proba_rep = np.full(rows_rep.shape[0], np.nan, dtype=np.float32)

    # Raw value only for single-feature (tabular) groups; NaN otherwise.
    feat_vals = np.full(grp_idx.shape[0], np.nan, dtype=np.float32)
    sfi = single_feat_idx[grp_idx]
    row_of_entry = np.repeat(np.arange(m), k)
    valid = sfi >= 0
    if valid.any():
        feat_vals[valid] = x_fl[row_of_entry[valid], sfi[valid]]

    table = pa.table(
        {
            "row_index": pa.array(rows_rep, type=pa.int64()),
            "proba_if_error": pa.array(proba_rep, type=pa.float32()),
            "rank": pa.array(ranks, type=pa.int16()),
            "feature_group": pa.array(grp_names.astype(str), type=pa.string()),
            "shap_value": pa.array(shap_vals.astype(np.float32), type=pa.float32()),
            "feature_value": pa.array(feat_vals, type=pa.float32()),
        },
        schema=schema,
    )
    writer.write_table(table)


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # noqa: BLE001
        print(f"[qa/shap] WARNING: matplotlib unavailable ({exc}); skipping plot.")
        return None


def _render_bar_plot(
    mean_abs: np.ndarray,
    names: list[str],
    out_dir: Path,
    *,
    fname: str,
    title: str,
) -> None:
    plt = _matplotlib()
    if plt is None:
        return
    try:
        order = np.argsort(mean_abs)[::-1]
        labels = [names[i] for i in order]
        vals = mean_abs[order]
        fig_h = max(4.0, 0.36 * len(labels) + 1.5)
        plt.figure(figsize=(9, fig_h))
        y = np.arange(len(labels))[::-1]
        plt.barh(y, vals, color="#1f77b4")
        plt.yticks(y, labels, fontsize=9)
        plt.xlabel("mean |SHAP value| (exact over processed rows)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        print(f"[qa/shap] WARNING: {fname} render failed ({exc}).")
    finally:
        plt.close("all")


def _render_beeswarm(
    shap_values: np.ndarray,
    x_sample: np.ndarray,
    feature_columns: list[str],
    out_dir: Path,
) -> None:
    if shap_values.shape[0] == 0:
        return
    plt = _matplotlib()
    if plt is None:
        return
    try:
        import shap

        plt.figure()
        shap.summary_plot(
            shap_values,
            x_sample,
            feature_names=feature_columns,
            plot_type="dot",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            out_dir / "shap_summary_beeswarm.png", dpi=120, bbox_inches="tight"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[qa/shap] WARNING: beeswarm render failed ({exc}).")
    finally:
        plt.close("all")
