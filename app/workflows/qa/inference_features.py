"""QA workflow — build model_1 feature matrix (frozen PC1 from training).

The actual feature-extraction logic lives in
`common.features.inference_matrix` so the training workflow can reuse it
without importing QA code. This module wires the QA-specific defaults
(row chunk + step IDs) over that shared core.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bootstrap import ensure_app_on_path
from common.features.inference_matrix import (
    assert_embedding_row_counts as _assert_embedding_row_counts,
    build_inference_feature_matrix,
    project_frozen_pc1_columns as _project_frozen_pc1_columns,
)
from workflows.qa.config import ROW_CHUNK

ensure_app_on_path(__file__)


def assert_embedding_row_counts(
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
) -> None:
    _assert_embedding_row_counts(parquet_path, embed_dir, embed_cols)


def project_frozen_pc1_columns(
    parquet_path: Path,
    embed_dir: Path,
    embed_cols: list[str],
    pc1_dir: Path,
    row_chunk: int = ROW_CHUNK,
    timer=None,
):
    return _project_frozen_pc1_columns(
        parquet_path,
        embed_dir,
        embed_cols,
        pc1_dir,
        row_chunk=row_chunk,
        timer=timer,
        step_id="qa.pc1.column",
    )


def build_qa_feature_matrix(
    parquet_path: Path,
    embed_dir: Path,
    feature_columns: list[str],
    embed_cols: list[str],
    pc1_dir: Path,
    row_chunk: int = ROW_CHUNK,
    timer=None,
) -> pl.DataFrame:
    return build_inference_feature_matrix(
        parquet_path,
        embed_dir,
        feature_columns,
        embed_cols,
        pc1_dir,
        row_chunk=row_chunk,
        timer=timer,
        pc1_step_id="qa.pc1.column",
    )
