"""Build a model-input feature matrix from frozen reduction axes + tabular columns.

Used by:
  * `workflows.qa.inference_features` — scoring new datasets with model_1.
  * `workflows.training.model_2` — training the IF_ERROR classifier on QA
    scored output (same features as model_1, never the model_1 predictions).

Living in `app/common/` keeps training and QA from importing each other while
still sharing the exact same feature-extraction logic. This is critical: model_2
must see identical features to model_1 at inference, so any drift between the
two implementations would silently corrupt training.

Reduction layout (resolved by `common.features.reduction.load_reduction_artifacts`):

  * Preferred: `model_1/reduction/reduction_manifest.json` plus
    `{col}_axes.npy` (shape (k, dim)) and `{col}_mean.npy` (shape (dim,)).
  * Legacy fallback: `model_1/pc1/{col}_pc1_axis.npy` (shape (dim,)) plus
    `{col}_pc1_mean.npy`. Promoted to a synthetic K=1 strategy.

Raw passthrough strategies skip the matmul entirely and emit one feature per
embedding dimension (e.g. 768 features per embed column).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from bootstrap import ensure_app_on_path
from common.config.columns import ML_PC1_SUFFIX, ML_TABULAR_COLS
from common.config.embedding import EMBED_ROW_CHUNK
from common.features.reduction import (
    ReductionArtifacts,
    feature_names_for_column,
    load_reduction_artifacts,
)
from common.pipeline_timing import step as pipeline_step

ensure_app_on_path(__file__)


def _embedding_row_count(embed_dir: Path, col: str) -> int:
    import pyarrow.parquet as pq

    path = embed_dir / f"{col}_EMBEDDING.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding sidecar: {path}")
    return pq.ParquetFile(path).metadata.num_rows


def assert_embedding_row_counts(
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
) -> None:
    """Embeddings must be row-aligned with the parquet (same order, same length)."""
    parquet_path = Path(parquet_path)
    embed_dir = Path(embed_dir)
    n_parquet = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    mismatches: list[str] = []
    for col in embed_cols:
        n_emb = _embedding_row_count(embed_dir, col)
        if n_emb != n_parquet:
            mismatches.append(f"  {col}: embeddings {n_emb:,} vs parquet {n_parquet:,}")
    if mismatches:
        raise ValueError(
            "Embedding row count does not match parquet — stale or wrong embed_dir.\n"
            + "\n".join(mismatches)
            + f"\n  parquet: {parquet_path}\n  embed_dir: {embed_dir}\n"
        )


def cast_tabular(df: pl.DataFrame) -> pl.DataFrame:
    """Cast tabular columns to float32 (numeric, bool, and datetime alike)."""
    present = [c for c in ML_TABULAR_COLS if c in df.columns]
    out = df.select(present)
    exprs = []
    for name, dtype in out.schema.items():
        if dtype == pl.Boolean:
            exprs.append(pl.col(name).cast(pl.Float32).alias(name))
        elif dtype in (pl.Datetime, pl.Date):
            exprs.append(pl.col(name).cast(pl.Int64).cast(pl.Float32).alias(name))
        elif dtype.is_numeric():
            exprs.append(pl.col(name).cast(pl.Float32).alias(name))
    return out.with_columns(exprs) if exprs else out


def _resolve_artifacts(
    pc1_dir: Path,
    embed_cols: list[str],
) -> ReductionArtifacts:
    """Find the reduction artifacts from either the new or legacy layout.

    Callers historically passed the legacy `pc1_dir` (`model_1/pc1/`). The
    new layout lives in a sibling `reduction/` directory, so we resolve
    relative to the parent unless the argument already points at a
    `reduction/` folder.
    """
    pc1_dir = Path(pc1_dir)
    if pc1_dir.name == "reduction":
        model_dir = pc1_dir.parent
    else:
        model_dir = pc1_dir.parent
    return load_reduction_artifacts(model_dir, embed_cols)


def _project_reduced_column(
    embed_dir: Path,
    col: str,
    artifacts: ReductionArtifacts,
    n_rows: int,
    row_chunk: int,
) -> np.ndarray:
    """Stream-project one embedding sidecar into a (n_rows, k_col) float32 block."""
    import pyarrow.parquet as pq

    k = artifacts.k_per_col[col]
    out = np.empty((n_rows, k), dtype=np.float32)
    field_name = f"{col}_EMBEDDING"
    axes = artifacts.axes_per_col.get(col)
    mean = artifacts.mean_per_col[col]
    offset = 0
    for emb_batch in pq.ParquetFile(embed_dir / f"{col}_EMBEDDING.parquet").iter_batches(
        batch_size=row_chunk
    ):
        n = emb_batch.num_rows
        col_data = emb_batch.column(emb_batch.schema.get_field_index(field_name))
        flat = col_data.values.to_numpy(zero_copy_only=False).astype(np.float32)
        dim = (
            len(flat) // n if n
            else (axes.shape[1] if axes is not None else mean.shape[0])
        )
        vectors = flat.reshape(n, dim)
        if axes is None or artifacts.strategy.is_raw:
            scores = vectors.astype(np.float32, copy=False)
        else:
            scores = ((vectors - mean) @ axes.T).astype(np.float32)
        end = min(offset + n, n_rows)
        out[offset:end] = scores[: end - offset]
        offset += n
    if offset != n_rows:
        raise ValueError(
            f"Embedding row count {offset:,} != parquet rows {n_rows:,} for {col!r} "
            f"under {embed_dir}."
        )
    return out


def project_frozen_reduction_columns(
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
    model_dir: Path | None = None,
    *,
    artifacts: ReductionArtifacts | None = None,
    row_chunk: int = EMBED_ROW_CHUNK,
    timer=None,
    step_id: str = "inference.pc1.column",
) -> tuple[dict[str, np.ndarray], ReductionArtifacts]:
    """Project each embedding column onto its frozen reduction axes.

    Returns
    -------
    columns : dict[str, (n_rows,) float32]
        One entry per *output feature column* (1-indexed PC names or raw
        dims, depending on the strategy).
    artifacts : ReductionArtifacts
        The bundle that was loaded — caller can inspect the strategy + K.
    """
    assert_embedding_row_counts(parquet_path, embed_dir, embed_cols)
    n_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    if artifacts is None:
        if model_dir is None:
            raise ValueError(
                "project_frozen_reduction_columns requires either `model_dir` "
                "or pre-loaded `artifacts`."
            )
        artifacts = load_reduction_artifacts(Path(model_dir), embed_cols)

    columns: dict[str, np.ndarray] = {}
    for col_i, col in enumerate(embed_cols, start=1):
        names = feature_names_for_column(
            col, artifacts.k_per_col[col], is_raw=artifacts.strategy.is_raw
        )
        with pipeline_step(
            step_id,
            f"Project {col} ({col_i}/{len(embed_cols)}) "
            f"[{artifacts.strategy.name}, k={artifacts.k_per_col[col]}]",
            f"common.features.inference_matrix.project_frozen_reduction_columns({col!r})",
            timer=timer,
        ):
            scores = _project_reduced_column(
                embed_dir, col, artifacts, n_rows, row_chunk
            )
        if scores.shape[1] != len(names):
            raise ValueError(
                f"projection k={scores.shape[1]} != feature_names_for_column={len(names)} "
                f"for {col!r}; manifest may be inconsistent with axes file."
            )
        for i, name in enumerate(names):
            columns[name] = scores[:, i]
    return columns, artifacts


def project_frozen_pc1_columns(
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
    pc1_dir: Path,
    row_chunk: int = EMBED_ROW_CHUNK,
    timer=None,
    step_id: str = "inference.pc1.column",
) -> dict[str, np.ndarray]:
    """Legacy entry point: PC1-only projection that returns `{col}_EMB_PC1` series.

    Internally routes through `project_frozen_reduction_columns`; works with
    either the new strategy layout (uses just the first PC of the saved
    axes) or the legacy `pc1/` axes layout.
    """
    artifacts = _resolve_artifacts(pc1_dir, embed_cols)
    columns, _ = project_frozen_reduction_columns(
        parquet_path,
        embed_dir,
        embed_cols,
        artifacts=artifacts,
        row_chunk=row_chunk,
        timer=timer,
        step_id=step_id,
    )
    # Caller expects `{col}_EMB_PC1` series. For PC1 strategy this is exactly
    # what was emitted. For higher-K strategies, expose just PC1 so the
    # legacy entry point keeps working — callers that need every K should
    # use `project_frozen_reduction_columns` directly.
    out: dict[str, np.ndarray] = {}
    for col in embed_cols:
        names = feature_names_for_column(
            col, artifacts.k_per_col[col], is_raw=artifacts.strategy.is_raw
        )
        if not names:
            continue
        legacy_name = f"{col}{ML_PC1_SUFFIX}"
        # Prefer the strategy's first PC; otherwise expose dim_001 from raw.
        out[legacy_name] = columns[names[0]]
    return out


def build_inference_feature_matrix(
    parquet_path: Path,
    embed_dir: Path,
    feature_columns: list[str],
    embed_cols: list[str],
    pc1_dir: Path,
    row_chunk: int = EMBED_ROW_CHUNK,
    timer=None,
    pc1_step_id: str = "inference.pc1.column",
) -> pl.DataFrame:
    """Tabular numerics + frozen reduction projections, in `feature_columns` order.

    `pc1_dir` is the legacy parameter name — it can point at either the
    new `model_1/reduction/` directory or the older `model_1/pc1/` one.
    The caller usually passes whatever was recorded in the model_1 manifest.
    """
    parquet_path = Path(parquet_path)
    tab_cols = [c for c in ML_TABULAR_COLS if c in pl.read_parquet_schema(parquet_path)]
    tab_df = pl.read_parquet(parquet_path, columns=tab_cols)
    tab_numeric = cast_tabular(tab_df)

    artifacts = _resolve_artifacts(pc1_dir, embed_cols)
    reduction_cols, _ = project_frozen_reduction_columns(
        parquet_path,
        embed_dir,
        embed_cols,
        artifacts=artifacts,
        row_chunk=row_chunk,
        timer=timer,
        step_id=pc1_step_id,
    )
    combined = pl.concat(
        [
            tab_numeric,
            pl.DataFrame({name: values for name, values in reduction_cols.items()}),
        ],
        how="horizontal",
    )
    missing = [c for c in feature_columns if c not in combined.columns]
    if missing:
        raise ValueError(
            "Inference feature matrix missing columns: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
            f"Reduction strategy={artifacts.strategy.name!r}, "
            f"per_col_k={artifacts.k_per_col}. "
            "If you trained model_1 with a different strategy, retrain or "
            "load the matching artifact."
        )
    return combined.select(feature_columns)
