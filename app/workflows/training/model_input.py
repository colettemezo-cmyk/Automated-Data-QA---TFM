"""Build and cache the leakage-safe model-input dataset (tabular + reduced embeddings).

The classifier never sees raw text / embed-source columns — only:
  * numeric tabular fields from `ML_TABULAR_COLS`
  * train-fitted reduced embedding projections, one or more columns per
    embed column, named according to the active `ReductionStrategy`
    (`{col}_EMB_PC{i}` for PCA strategies, `{col}_EMB_DIM{i:03d}` for raw).

Meta columns (`EXECUTION_ID`, `IS_OWN_RESTAURANT`, `is_train`, `is_test`) are
stored for splitting and evaluation but excluded from the feature matrix.

Caches under `data/ml/cache/<id>/` so iterative model work skips re-streaming
13M embedding rows when inputs are unchanged. The cache key includes the
reduction strategy so swapping strategies never silently reuses stale features.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bootstrap import ensure_app_on_path
from common.config.columns import (
    EMBED_COLS,
    ML_PC1_SUFFIX,
    ML_TABULAR_COLS,
    SPLIT_GROUP_COL,
    TARGET_COL,
)
from common.features.pca import first_pc_axis, top_k_pc_axes
from common.features.reduction import (
    ReductionStrategy,
    feature_names_for_column,
    parse_strategy,
    save_reduction_artifacts,
)
from common.pipeline_timing import step as pipeline_step
from common.storage.parquet_io import atomic_write_parquet
from workflows.training.config import (
    EMBED_DIR,
    ML_CACHE_DIR,
    ML_DIR,
    ML_MAX_EXECUTIONS,
    ML_ROW_CHUNK,
    PARQUET_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    REDUCTION_STRATEGY,
    TEST_SIZE,
)

ensure_app_on_path(__file__)


META_COLUMNS = [SPLIT_GROUP_COL, TARGET_COL, "is_train", "is_test"]


def _resolve_strategy(
    strategy: ReductionStrategy | str | None,
) -> ReductionStrategy:
    """Coerce CLI/string inputs into a ReductionStrategy."""
    if strategy is None:
        return parse_strategy(REDUCTION_STRATEGY)
    if isinstance(strategy, ReductionStrategy):
        return strategy
    return parse_strategy(strategy)


def ensure_project_root() -> Path:
    """Run pipeline paths from repo root regardless of caller cwd."""
    os.chdir(PROJECT_ROOT)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    ML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECT_ROOT


def _embed_cols_present(parquet_path: Path) -> list[str]:
    schema = pl.read_parquet_schema(parquet_path)
    return [c for c in EMBED_COLS if c in schema]


def _source_fingerprint(parquet_path: Path, embed_dir: Path, embed_cols: list[str]) -> str:
    parts = [str(parquet_path.resolve()), str(parquet_path.stat().st_mtime_ns)]
    for col in embed_cols:
        p = embed_dir / f"{col}_EMBEDDING.parquet"
        parts.append(str(p.resolve()))
        parts.append(str(p.stat().st_mtime_ns))
    blob = "|".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_dir(
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
    *,
    max_executions: int | None,
    test_size: float,
    random_state: int,
    strategy: ReductionStrategy | None = None,
) -> Path:
    src = _source_fingerprint(parquet_path, embed_dir, embed_cols)
    strategy = _resolve_strategy(strategy)
    tag = (
        f"src{src}_exec{max_executions or 'all'}_ts{test_size}_rs{random_state}"
        f"_red{strategy.name}"
    )
    return ML_CACHE_DIR / tag


def _pc1_axis_files_complete(pc1_save_dir: Path, embed_cols: list[str]) -> bool:
    """Legacy check: true when the pre-strategy PC1 axes layout is complete."""
    pc1_save_dir = Path(pc1_save_dir)
    return all(
        (pc1_save_dir / f"{col}_pc1_axis.npy").exists()
        and (pc1_save_dir / f"{col}_pc1_mean.npy").exists()
        for col in embed_cols
    )


def _reduction_axis_files_complete(
    reduction_save_dir: Path,
    embed_cols: list[str],
    strategy: ReductionStrategy,
) -> bool:
    """True when the strategy-aware reduction layout is complete for `strategy`."""
    from common.features.reduction import REDUCTION_MANIFEST

    reduction_save_dir = Path(reduction_save_dir)
    manifest_path = reduction_save_dir / REDUCTION_MANIFEST
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    saved = payload.get("strategy", {}).get("name")
    if saved != strategy.name:
        return False
    for col in embed_cols:
        if not (reduction_save_dir / f"{col}_mean.npy").exists():
            return False
        if not strategy.is_raw and not (
            reduction_save_dir / f"{col}_axes.npy"
        ).exists():
            return False
    return True


def _execution_masks(
    parquet_path: Path,
    *,
    test_size: float,
    random_state: int,
    max_executions: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_mask, test_mask, row_active) aligned to parquet rows."""
    from sklearn.model_selection import GroupShuffleSplit

    n_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    exec_ids = (
        pl.read_parquet(parquet_path, columns=[SPLIT_GROUP_COL])
        .to_series()
        .to_numpy()
    )
    row_active = np.ones(n_rows, dtype=bool)
    if max_executions is not None:
        unique_exec = np.unique(exec_ids)
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(
            unique_exec,
            size=min(max_executions, len(unique_exec)),
            replace=False,
        )
        row_active = np.isin(exec_ids, chosen)

    unique_ids = np.unique(exec_ids)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    tr_idx, te_idx = next(splitter.split(unique_ids, groups=unique_ids))
    train_set = set(unique_ids[tr_idx])
    test_set = set(unique_ids[te_idx])
    train_mask = np.isin(exec_ids, list(train_set))
    test_mask = np.isin(exec_ids, list(test_set))
    train_mask &= row_active
    test_mask &= row_active
    return train_mask, test_mask, row_active


def export_model_1_pc1_axes(
    parquet_path: Path = PARQUET_PATH,
    embed_dir: Path = EMBED_DIR,
    pc1_save_dir: Path | None = None,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    max_executions: int | None = None,
    row_chunk: int = ML_ROW_CHUNK,
    timer=None,
    strategy: ReductionStrategy | str | None = None,
    reduction_save_dir: Path | None = None,
) -> Path:
    """Fit train-only reduction axes and save under `model_1/reduction/`.

    `pc1_save_dir` is kept as an alias for backward compatibility with the
    legacy entry point in `scripts/export_model_1_pc1.py`. The actual
    persisted layout is the strategy-aware one (`reduction/`).
    """
    from workflows.training.model_1_export import model_1_reduction_dir

    ensure_project_root()
    parquet_path = Path(parquet_path)
    embed_dir = Path(embed_dir)
    strategy_obj = _resolve_strategy(strategy)
    save_dir = Path(
        reduction_save_dir
        or pc1_save_dir
        or model_1_reduction_dir()
    )
    embed_cols = _embed_cols_present(parquet_path)

    if _reduction_axis_files_complete(save_dir, embed_cols, strategy_obj):
        print(
            f"[model_input] reduction axes already present under {save_dir} "
            f"(strategy={strategy_obj.name!r})",
            flush=True,
        )
        return save_dir

    train_mask, test_mask, row_active = _execution_masks(
        parquet_path,
        test_size=test_size,
        random_state=random_state,
        max_executions=max_executions,
    )
    cache_dir = _cache_dir(
        parquet_path,
        embed_dir,
        embed_cols,
        max_executions=max_executions,
        test_size=test_size,
        random_state=random_state,
        strategy=strategy_obj,
    )
    print(
        f"[model_input] exporting reduction axes for {len(embed_cols)} columns "
        f"(strategy={strategy_obj.name!r}, train rows {int(train_mask.sum()):,}) "
        f"-> {save_dir}",
        flush=True,
    )
    with pipeline_step(
        "4.1.model_input.export_pc1",
        f"Export frozen reduction axes ({strategy_obj.name})",
        "workflows.training.model_input.export_model_1_pc1_axes",
        timer=timer,
    ):
        compute_reduction_columns(
            parquet_path=parquet_path,
            embed_dir=embed_dir,
            embed_cols=embed_cols,
            train_mask=train_mask,
            row_active=row_active,
            cache_dir=cache_dir,
            row_chunk=row_chunk,
            reduction_save_dir=save_dir,
            timer=timer,
            strategy=strategy_obj,
        )
    return save_dir


def _cast_tabular(df: pl.DataFrame) -> pl.DataFrame:
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


def _materialize_embedding_subset(
    parquet_path: Path,
    embed_dir: Path,
    col: str,
    row_indices: np.ndarray,
    out_path: Path,
    row_chunk: int = ML_ROW_CHUNK,
) -> None:
    """Copy embedding rows for `row_indices` into a compact sidecar (one-time cost)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out_path.exists():
        return

    emb_path = embed_dir / f"{col}_EMBEDDING.parquet"
    field_name = f"{col}_EMBEDDING"
    sorted_idx = np.sort(row_indices)
    ptr = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    writer = None
    dim = None

    pf = pq.ParquetFile(emb_path)
    offset = 0
    for emb_batch in pf.iter_batches(batch_size=row_chunk):
        n = emb_batch.num_rows
        col_data = emb_batch.column(emb_batch.schema.get_field_index(field_name))
        flat = col_data.values.to_numpy(zero_copy_only=False).astype(np.float32)
        if dim is None:
            dim = len(flat) // n if n else 768
        vectors = flat.reshape(n, dim)

        batch_rows: list[np.ndarray] = []
        while ptr < len(sorted_idx) and sorted_idx[ptr] < offset + n:
            local = int(sorted_idx[ptr] - offset)
            batch_rows.append(vectors[local])
            ptr += 1

        if batch_rows:
            chunk = np.stack(batch_rows, axis=0)
            fsl = pa.FixedSizeListArray.from_arrays(
                pa.array(chunk.reshape(-1), type=pa.float32()), dim
            )
            table = pa.Table.from_arrays([fsl], names=[field_name])
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
        offset += n

    if writer is None:
        raise ValueError(f"No embedding rows materialized for {col}")
    writer.close()
    tmp.replace(out_path)


def _stream_subset_embeddings(
    emb_subset_path: Path,
    col: str,
    row_chunk: int,
):
    import pyarrow.parquet as pq

    field_name = f"{col}_EMBEDDING"
    pf = pq.ParquetFile(emb_subset_path)
    offset = 0
    for batch in pf.iter_batches(batch_size=row_chunk):
        n = batch.num_rows
        col_data = batch.column(batch.schema.get_field_index(field_name))
        flat = col_data.values.to_numpy(zero_copy_only=False).astype(np.float32)
        dim = len(flat) // n if n else 768
        yield offset, flat.reshape(n, dim)
        offset += n


def _fit_reduction_on_subset(
    emb_subset_path: Path,
    train_mask: np.ndarray,
    col: str,
    row_chunk: int,
    strategy: ReductionStrategy,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int, float]:
    """Reduction fit from train rows of a compact embedding subset (row-weighted).

    Returns `(axes, mean, ratios, k, achieved_cumulative_variance)`. For raw
    mode `axes` is `None` and `mean` is zero — the projection is identity.
    """
    vectors_list: list[np.ndarray] = []
    offset = 0
    for batch_offset, vectors in _stream_subset_embeddings(
        emb_subset_path, col, row_chunk
    ):
        n = vectors.shape[0]
        use = train_mask[batch_offset : batch_offset + n]
        for i in range(n):
            if use[i]:
                vectors_list.append(vectors[i])
        offset += n

    if not vectors_list:
        dim = 768
        return (
            None if strategy.is_raw else np.zeros((0, dim), dtype=np.float32),
            np.zeros(dim, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            0,
            0.0,
        )
    uv = np.stack(vectors_list, axis=0)
    counts = np.ones(len(vectors_list), dtype=np.int64)
    return _fit_strategy_axes(uv, counts, strategy)


def _project_reduction_subset(
    emb_subset_path: Path,
    col: str,
    axes: np.ndarray | None,
    mean: np.ndarray,
    n_rows: int,
    row_chunk: int,
    k: int,
) -> np.ndarray:
    out = np.empty((n_rows, k), dtype=np.float32)
    offset = 0
    for batch_offset, vectors in _stream_subset_embeddings(
        emb_subset_path, col, row_chunk
    ):
        n = vectors.shape[0]
        out[batch_offset : batch_offset + n] = _project_batch(vectors, axes, mean)
        offset += n
    if offset != n_rows:
        raise ValueError(f"Subset embedding rows {offset} != expected {n_rows}")
    return out


def _fit_strategy_axes(
    unique_vectors: np.ndarray,
    counts: np.ndarray,
    strategy: ReductionStrategy,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int, float]:
    """Fit `strategy` on `(unique_vectors, counts)`; uniform return shape.

    Returns `(axes_or_None, mean, ratios, k, achieved_cumulative_variance)`.
    """
    if strategy.is_raw:
        dim = int(unique_vectors.shape[1])
        # Raw mode: no centering, no projection — passthrough at inference.
        # Mean is still computed (weighted) so QA / model_2 can recover it if
        # they ever want to switch to a centred passthrough.
        if counts.sum() > 0:
            counts_f = counts.astype(np.float64)
            mean = (
                (counts_f[:, None] * unique_vectors).sum(axis=0) / counts_f.sum()
            ).astype(np.float32)
        else:
            mean = np.zeros(dim, dtype=np.float32)
        return None, mean, np.zeros(0, dtype=np.float32), dim, 0.0

    stats = top_k_pc_axes(
        unique_vectors,
        counts,
        variance_target=strategy.variance_target,
        fixed_k=strategy.fixed_k,
        k_max=strategy.k_max,
        k_min=strategy.k_min,
    )
    return (
        stats["axes"],
        stats["mean"],
        stats["explained_variance_ratios"],
        int(stats["k"]),
        float(stats["cumulative_explained_variance"]),
    )


def _project_batch(
    vectors: np.ndarray,
    axes: np.ndarray | None,
    mean: np.ndarray,
) -> np.ndarray:
    """Project a (n, dim) batch using the fitted axes; raw mode = passthrough."""
    if axes is None or axes.size == 0:
        return vectors.astype(np.float32, copy=False)
    return ((vectors - mean) @ axes.T).astype(np.float32)


def _fit_reduction_full_stream(
    parquet_path: Path,
    embed_dir: Path,
    col: str,
    train_mask: np.ndarray,
    row_active: np.ndarray,
    row_chunk: int,
    strategy: ReductionStrategy,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int, float]:
    """Full-dataset streaming reduction fit (used when not subsampling)."""
    import pyarrow.parquet as pq

    emb_path = embed_dir / f"{col}_EMBEDDING.parquet"
    field_name = f"{col}_EMBEDDING"
    text_pf = pq.ParquetFile(parquet_path)
    emb_pf = pq.ParquetFile(emb_path)
    n_rows = train_mask.shape[0]
    text_to_idx: dict[str, int] = {}
    unique_vectors: list[np.ndarray] = []
    counts_list: list[int] = []
    offset = 0
    dim = 768

    for text_batch, emb_batch in zip(
        text_pf.iter_batches(columns=[col], batch_size=row_chunk),
        emb_pf.iter_batches(batch_size=row_chunk),
    ):
        n = emb_batch.num_rows
        texts = [
            v if v is not None else "MISSING"
            for v in text_batch.column(0).to_pylist()
        ]
        col_data = emb_batch.column(emb_batch.schema.get_field_index(field_name))
        flat = col_data.values.to_numpy(zero_copy_only=False).astype(np.float32)
        dim = len(flat) // n if n else dim
        vectors = flat.reshape(n, dim)
        slice_train = train_mask[offset : offset + n]
        slice_active = row_active[offset : offset + n]
        for i in range(n):
            if not (slice_train[i] and slice_active[i]):
                continue
            text = texts[i]
            if text in text_to_idx:
                counts_list[text_to_idx[text]] += 1
            else:
                text_to_idx[text] = len(unique_vectors)
                unique_vectors.append(vectors[i].copy())
                counts_list.append(1)
        offset += n

    if offset != n_rows:
        raise ValueError("Embedding row count mismatch")
    if not unique_vectors:
        return (
            None if strategy.is_raw else np.zeros((0, dim), dtype=np.float32),
            np.zeros(dim, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            0,
            0.0,
        )
    return _fit_strategy_axes(
        np.stack(unique_vectors), np.asarray(counts_list), strategy
    )


def _project_reduction_full_stream(
    embed_dir: Path,
    col: str,
    axes: np.ndarray | None,
    mean: np.ndarray,
    row_active: np.ndarray,
    row_chunk: int,
    k: int,
) -> np.ndarray:
    import pyarrow.parquet as pq

    field_name = f"{col}_EMBEDDING"
    parts: list[np.ndarray] = []
    offset = 0
    for emb_batch in pq.ParquetFile(embed_dir / f"{col}_EMBEDDING.parquet").iter_batches(
        batch_size=row_chunk
    ):
        n = emb_batch.num_rows
        col_data = emb_batch.column(emb_batch.schema.get_field_index(field_name))
        flat = col_data.values.to_numpy(zero_copy_only=False).astype(np.float32)
        dim = len(flat) // n if n else (axes.shape[1] if axes is not None else mean.shape[0])
        vectors = flat.reshape(n, dim)
        active = row_active[offset : offset + n]
        scores = _project_batch(vectors, axes, mean)
        if active.any():
            parts.append(scores[active])
        offset += n
    if not parts:
        return np.zeros((0, k), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _project_reduction_full_all(
    embed_dir: Path,
    col: str,
    axes: np.ndarray | None,
    mean: np.ndarray,
    n_rows: int,
    row_chunk: int,
    k: int,
) -> np.ndarray:
    import pyarrow.parquet as pq

    out = np.empty((n_rows, k), dtype=np.float32)
    field_name = f"{col}_EMBEDDING"
    offset = 0
    for emb_batch in pq.ParquetFile(embed_dir / f"{col}_EMBEDDING.parquet").iter_batches(
        batch_size=row_chunk
    ):
        n = emb_batch.num_rows
        col_data = emb_batch.column(emb_batch.schema.get_field_index(field_name))
        flat = col_data.values.to_numpy(zero_copy_only=False).astype(np.float32)
        dim = len(flat) // n if n else (axes.shape[1] if axes is not None else mean.shape[0])
        vectors = flat.reshape(n, dim)
        out[offset : offset + n] = _project_batch(vectors, axes, mean)
        offset += n
    return out


def compute_reduction_columns(
    *,
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
    train_mask: np.ndarray,
    row_active: np.ndarray,
    cache_dir: Path | None,
    row_chunk: int = ML_ROW_CHUNK,
    reduction_save_dir: Path | None = None,
    timer=None,
    strategy: ReductionStrategy | str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, dict]]:
    """Train-fitted reduction per embed column for `strategy`.

    Uses subset materialization when subsampling. If `reduction_save_dir` is
    set, persists the strategy-aware axes layout (`reduction/`) so QA and
    model_2 can re-project new data without refitting.

    Returns
    -------
    projected : dict[col -> (n_active, k_col) float32]
        The active-rows projection for each embed column.
    feature_names : dict[col -> list[str]]
        Per-column ordered feature names (`{col}_EMB_PC{i}` or
        `{col}_EMB_DIM{i:03d}` for raw).
    per_col_meta : dict[col -> {"k", "achieved_cumulative_variance",
                                "explained_variance_ratios", ...}]
        Per-column diagnostics that get written into the manifest.
    """
    strategy_obj = _resolve_strategy(strategy)
    n_active = int(row_active.sum())
    use_subset = n_active < row_active.shape[0]
    row_indices = np.flatnonzero(row_active) if use_subset else None

    projected: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    per_col_meta: dict[str, dict] = {}
    axes_per_col: dict[str, np.ndarray | None] = {}
    mean_per_col: dict[str, np.ndarray] = {}
    ratios_per_col: dict[str, np.ndarray] = {}

    emb_subset_dir = (cache_dir / "emb_subset") if cache_dir else None
    if use_subset and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    train_mask_active = train_mask[row_active] if use_subset else train_mask

    for col_i, col in enumerate(embed_cols, start=1):
        with pipeline_step(
            "4.1.model_input.pc1.column",
            f"Reduction column {col} ({col_i}/{len(embed_cols)}) [{strategy_obj.name}]",
            f"workflows.training.model_input.compute_reduction_columns({col!r})",
            timer=timer,
        ):
            if use_subset and emb_subset_dir is not None:
                emb_sub = emb_subset_dir / f"{col}_EMBEDDING.parquet"
                if not emb_sub.exists():
                    with pipeline_step(
                        "4.1.model_input.pc1.materialize",
                        f"Materialize emb subset {col}",
                        "workflows.training.model_input._materialize_embedding_subset",
                        timer=timer,
                    ):
                        _materialize_embedding_subset(
                            parquet_path, embed_dir, col, row_indices, emb_sub, row_chunk
                        )
                axes, mean, ratios, k, cum_var = _fit_reduction_on_subset(
                    emb_sub, train_mask_active, col, row_chunk, strategy_obj
                )
                scores = _project_reduction_subset(
                    emb_sub, col, axes, mean, n_active, row_chunk, k
                )
            else:
                axes, mean, ratios, k, cum_var = _fit_reduction_full_stream(
                    parquet_path, embed_dir, col, train_mask, row_active,
                    row_chunk, strategy_obj,
                )
                if n_active < row_active.shape[0]:
                    scores = _project_reduction_full_stream(
                        embed_dir, col, axes, mean, row_active, row_chunk, k
                    )
                else:
                    scores = _project_reduction_full_all(
                        embed_dir, col, axes, mean, row_active.shape[0], row_chunk, k
                    )

            names = feature_names_for_column(col, k, is_raw=strategy_obj.is_raw)
            projected[col] = scores
            feature_names[col] = names
            axes_per_col[col] = axes
            mean_per_col[col] = mean
            ratios_per_col[col] = ratios if ratios is not None else np.zeros(0, dtype=np.float32)
            per_col_meta[col] = {
                "k": int(k),
                "achieved_cumulative_variance": float(cum_var),
                "explained_variance_ratios": ratios.astype(np.float32).tolist() if ratios is not None else [],
                "variance_target": strategy_obj.variance_target,
                "k_max": strategy_obj.k_max,
                "n_features": len(names),
            }
            print(
                f"[model_input] {col}: k={k} cum_var={cum_var:.4f} "
                f"strategy={strategy_obj.name!r}",
                flush=True,
            )

    if reduction_save_dir is not None:
        # Accept either the model directory (back-compat) or the explicit
        # reduction/ subdirectory. `save_reduction_artifacts` writes directly
        # to whatever path we pass; normalise to `<model_dir>/reduction/`.
        save_path = Path(reduction_save_dir)
        from common.features.reduction import (
            REDUCTION_SUBDIR as _RSUB,
            LEGACY_PC1_SUBDIR as _PSUB,
        )
        if save_path.name == _PSUB:
            save_path = save_path.parent / _RSUB
        elif save_path.name != _RSUB:
            save_path = save_path / _RSUB
        save_reduction_artifacts(
            save_path,
            strategy_obj,
            embed_cols=embed_cols,
            axes_per_col=axes_per_col,
            mean_per_col=mean_per_col,
            k_per_col={col: per_col_meta[col]["k"] for col in embed_cols},
            achieved_cumvar_per_col={
                col: per_col_meta[col]["achieved_cumulative_variance"]
                for col in embed_cols
            },
            ratios_per_col=ratios_per_col,
        )

    return projected, feature_names, per_col_meta


def compute_pc1_columns(
    *,
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
    train_mask: np.ndarray,
    row_active: np.ndarray,
    cache_dir: Path | None,
    row_chunk: int = ML_ROW_CHUNK,
    pc1_save_dir: Path | None = None,
    timer=None,
) -> dict[str, np.ndarray]:
    """Legacy PC1-only shim. Calls `compute_reduction_columns` with strategy=pc1.

    Kept so existing callers (`scripts/export_model_1_pc1.py`) keep working.
    The returned dict maps `{col}_EMB_PC1 -> (n_active,) float32`, matching
    the legacy signature.
    """
    projected, _, _ = compute_reduction_columns(
        parquet_path=parquet_path,
        embed_dir=embed_dir,
        embed_cols=embed_cols,
        train_mask=train_mask,
        row_active=row_active,
        cache_dir=cache_dir,
        row_chunk=row_chunk,
        reduction_save_dir=pc1_save_dir,
        timer=timer,
        strategy=parse_strategy("pc1"),
    )
    return {f"{col}{ML_PC1_SUFFIX}": projected[col].reshape(-1) for col in embed_cols}


def build_model_input_dataset(
    parquet_path: Path = PARQUET_PATH,
    embed_dir: Path = EMBED_DIR,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    max_executions: int | None = ML_MAX_EXECUTIONS,
    row_chunk: int = ML_ROW_CHUNK,
    force: bool = False,
    pc1_save_dir: Path | None = None,
    timer=None,
    strategy: ReductionStrategy | str | None = None,
    reduction_save_dir: Path | None = None,
) -> tuple[Path, list[str]]:
    """Build `model_input.parquet` + cache; return path and feature column names.

    `pc1_save_dir` and `reduction_save_dir` are interchangeable aliases for
    the directory the new `reduction/` layout lives in; the legacy parameter
    name is kept so callers in `classifier.py` keep compiling. The cache is
    keyed by the reduction strategy so swapping strategies produces a fresh
    `model_input.parquet`.
    """
    ensure_project_root()
    parquet_path = Path(parquet_path)
    embed_dir = Path(embed_dir)
    strategy_obj = _resolve_strategy(strategy)

    embed_cols = _embed_cols_present(parquet_path)
    cache_dir = _cache_dir(
        parquet_path,
        embed_dir,
        embed_cols,
        max_executions=max_executions,
        test_size=test_size,
        random_state=random_state,
        strategy=strategy_obj,
    )
    out_path = cache_dir / "model_input.parquet"
    meta_path = cache_dir / "meta.json"

    save_dir = reduction_save_dir or pc1_save_dir

    if not force and out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with pipeline_step(
            "4.1.model_input.cache_hit",
            "Load cached model_input",
            f"workflows.training.model_input (cache {cache_dir.name})",
            predicted_sec=5.0,
            timer=timer,
        ):
            pass
        if save_dir is not None and not _reduction_axis_files_complete(
            save_dir, embed_cols, strategy_obj
        ):
            export_model_1_pc1_axes(
                parquet_path=parquet_path,
                embed_dir=embed_dir,
                reduction_save_dir=save_dir,
                test_size=test_size,
                random_state=random_state,
                max_executions=meta.get("max_executions", max_executions),
                row_chunk=row_chunk,
                timer=timer,
                strategy=strategy_obj,
            )
        return out_path, meta["feature_columns"]

    with pipeline_step(
        "4.1.model_input.build",
        f"Build model_input.parquet ({strategy_obj.name})",
        "workflows.training.model_input.build_model_input_dataset",
        timer=timer,
    ):
        return _build_model_input_inner(
            parquet_path=parquet_path,
            embed_dir=embed_dir,
            embed_cols=embed_cols,
            cache_dir=cache_dir,
            out_path=out_path,
            meta_path=meta_path,
            max_executions=max_executions,
            test_size=test_size,
            random_state=random_state,
            row_chunk=row_chunk,
            reduction_save_dir=save_dir,
            timer=timer,
            strategy=strategy_obj,
        )


def _build_model_input_inner(
    *,
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
    cache_dir: Path,
    out_path: Path,
    meta_path: Path,
    max_executions: int | None,
    test_size: float,
    random_state: int,
    row_chunk: int,
    reduction_save_dir: Path | None = None,
    timer=None,
    strategy: ReductionStrategy | str | None = None,
) -> tuple[Path, list[str]]:
    strategy_obj = _resolve_strategy(strategy)
    train_mask, test_mask, row_active = _execution_masks(
        parquet_path,
        test_size=test_size,
        random_state=random_state,
        max_executions=max_executions,
    )
    if max_executions is not None:
        print(
            f"[model_input] subsample {int(row_active.sum()):,} rows",
            flush=True,
        )

    tab_cols = [c for c in ML_TABULAR_COLS if c in pl.read_parquet_schema(parquet_path)]
    meta_cols = [SPLIT_GROUP_COL, TARGET_COL]
    load_cols = meta_cols + tab_cols

    with pipeline_step(
        "4.1.model_input.tabular",
        "Load tabular + target columns",
        "workflows.training.model_input._build_model_input_inner (tabular)",
        timer=timer,
    ):
        if row_active.all():
            tab_df = pl.read_parquet(parquet_path, columns=load_cols)
        else:
            exec_ids = (
                pl.read_parquet(parquet_path, columns=[SPLIT_GROUP_COL])
                .to_series()
                .to_numpy()
            )
            exec_filter = np.unique(exec_ids[row_active])
            tab_df = (
                pl.scan_parquet(parquet_path)
                .filter(pl.col(SPLIT_GROUP_COL).is_in(exec_filter.tolist()))
                .select(load_cols)
                .collect()
            )

    projected, feature_names, _per_col_meta = compute_reduction_columns(
        parquet_path=parquet_path,
        embed_dir=embed_dir,
        embed_cols=embed_cols,
        train_mask=train_mask,
        row_active=row_active,
        cache_dir=cache_dir,
        row_chunk=row_chunk,
        reduction_save_dir=reduction_save_dir,
        timer=timer,
        strategy=strategy_obj,
    )

    tab_numeric = _cast_tabular(tab_df)
    # Fan out per-column (n_active, k_col) matrices into named float32 series.
    reduction_series: list[pl.Series] = []
    for col in embed_cols:
        matrix = projected[col]
        names = feature_names[col]
        if matrix.shape[1] != len(names):
            raise ValueError(
                f"projection shape {matrix.shape} does not match "
                f"{len(names)} feature names for {col!r}"
            )
        for i, name in enumerate(names):
            reduction_series.append(pl.Series(name, matrix[:, i]))

    model_df = tab_numeric.with_columns(
        [
            tab_df.get_column(SPLIT_GROUP_COL),
            tab_df.get_column(TARGET_COL),
            pl.Series("is_train", train_mask[row_active]),
            pl.Series("is_test", test_mask[row_active]),
            *reduction_series,
        ]
    )

    reduction_feature_names = [s.name for s in reduction_series]
    feature_columns = tab_numeric.columns + reduction_feature_names
    cache_dir.mkdir(parents=True, exist_ok=True)
    with pipeline_step(
        "4.1.model_input.write",
        "Write model_input.parquet + meta.json",
        "common.storage.parquet_io.atomic_write_parquet",
        timer=timer,
    ):
        atomic_write_parquet(model_df, out_path)

    meta: dict[str, Any] = {
        "feature_columns": feature_columns,
        "meta_columns": META_COLUMNS,
        "excluded_from_features": list(
            set(EMBED_COLS) | {TARGET_COL, SPLIT_GROUP_COL, "is_train", "is_test"}
        ),
        "max_executions": max_executions,
        "test_size": test_size,
        "random_state": random_state,
        "n_rows": model_df.height,
        "source_parquet": str(parquet_path),
        "reduction_strategy": strategy_obj.name,
        "reduction": strategy_obj.to_manifest_dict(),
        "n_features": len(feature_columns),
        "n_reduction_features": len(reduction_feature_names),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[model_input] wrote {out_path} ({model_df.height:,} rows, "
        f"{len(feature_columns)} features, strategy={strategy_obj.name!r})"
    )
    return out_path, feature_columns
