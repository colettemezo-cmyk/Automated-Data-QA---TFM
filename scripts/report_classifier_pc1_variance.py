"""Report explained variance for train-fitted PC1 (classifier / model_1 features).

Uses the same fit as `workflows.training.model_input` (train executions only,
deduplicated text weighted by row count). Full training corpus by default.

Usage:
    python scripts/report_classifier_pc1_variance.py
    python scripts/report_classifier_pc1_variance.py --executions 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import numpy as np  # noqa: E402

from common.config.columns import EMBED_COLS, ML_PC1_SUFFIX  # noqa: E402
from common.features.pca import first_pc_fit_stats  # noqa: E402
from workflows.training.config import (  # noqa: E402
    EMBED_DIR,
    ML_DIR,
    ML_ROW_CHUNK,
    PARQUET_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from workflows.training.model_input import (  # noqa: E402
    _embed_cols_present,
    _execution_masks,
)  # noqa: E402


def _collect_train_unique_vectors(
    parquet_path: Path,
    embed_dir: Path,
    col: str,
    train_mask: np.ndarray,
    row_active: np.ndarray,
    row_chunk: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mirror `_fit_pc1_full_stream` but return stacked unique vectors + counts."""
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

    for text_batch, emb_batch in zip(
        text_pf.iter_batches(columns=[col], batch_size=row_chunk),
        emb_pf.iter_batches(batch_size=row_chunk),
    ):
        n = emb_batch.num_rows
        texts = [v if v is not None else "MISSING" for v in text_batch.column(0).to_pylist()]
        col_data = emb_batch.column(emb_batch.schema.get_field_index(field_name))
        flat = col_data.values.to_numpy(zero_copy_only=False).astype(np.float32)
        dim = len(flat) // n if n else 768
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
        return np.zeros((0, 768), dtype=np.float32), np.zeros(0, dtype=np.int64), 0
    return np.stack(unique_vectors), np.asarray(counts_list), int(train_mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explained variance report for classifier PC1 features."
    )
    parser.add_argument(
        "--executions",
        type=int,
        default=None,
        metavar="N",
        help="Subsample N executions (default: all, matching full model_input).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ML_DIR / "classifier_pc1_variance_report.json",
        help="JSON report path.",
    )
    args = parser.parse_args()

    parquet_path = PARQUET_PATH
    embed_dir = EMBED_DIR
    embed_cols = _embed_cols_present(parquet_path)
    train_mask, _test_mask, row_active = _execution_masks(
        parquet_path,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        max_executions=args.executions,
    )

    print(
        "Classifier PC1 explained variance (train-only fit, text-deduped, row-weighted)\n"
        f"  parquet: {parquet_path}\n"
        f"  train rows: {int(train_mask.sum()):,}  |  "
        f"executions cap: {args.executions or 'all'}\n"
        f"  split: test_size={TEST_SIZE} random_state={RANDOM_STATE}\n"
    )

    rows: list[dict] = []
    header = (
        f"{'column':<28} {'pc1_EV':>8} {'top3_sum':>9} "
        f"{'unique':>10} {'train_rows':>12}"
    )
    print(header)
    print("-" * len(header))

    for col in embed_cols:
        uv, counts, n_train = _collect_train_unique_vectors(
            parquet_path,
            embed_dir,
            col,
            train_mask,
            row_active,
            ML_ROW_CHUNK,
        )
        stats = first_pc_fit_stats(uv, counts, top_k=5)
        pc1 = stats["pc1_explained_variance_ratio"]
        top3 = sum(stats["top_explained_variance_ratios"][:3])
        print(
            f"{col + ML_PC1_SUFFIX:<28} {pc1:>8.4f} {top3:>9.4f} "
            f"{stats['n_unique_vectors']:>10,} {n_train:>12,}"
        )
        rows.append(
            {
                "embed_col": col,
                "feature_col": col + ML_PC1_SUFFIX,
                "pc1_explained_variance_ratio": pc1,
                "top5_explained_variance_ratios": stats["top_explained_variance_ratios"],
                "top5_cumulative_explained_variance": stats["top_k_cumulative_explained_variance"],
                "n_unique_texts": stats["n_unique_vectors"],
                "n_train_rows_weighted": stats["n_weighted_rows"],
                "embedding_dim": stats["embedding_dim"],
            }
        )

    report = {
        "description": "Train-only PC1 explained variance (classifier / model_1 features)",
        "parquet_path": str(parquet_path),
        "embed_dir": str(embed_dir),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "max_executions": args.executions,
        "n_train_rows": int(train_mask.sum()),
        "columns": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
