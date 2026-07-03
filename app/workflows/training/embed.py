"""Training workflow — stage 2: embeddings for the training corpus."""

from __future__ import annotations

from pathlib import Path

from bootstrap import ensure_app_on_path
from common.features.embeddings import embed_text_columns as _embed
from workflows.training.config import EMBED_COLS, EMBED_DIR, PARQUET_PATH

ensure_app_on_path(__file__)


def embed_training_columns(
    parquet_path: Path = PARQUET_PATH,
    embed_dir: Path = EMBED_DIR,
    *,
    force: bool = False,
    timer=None,
) -> None:
    _embed(
        parquet_path=parquet_path,
        embed_dir=embed_dir,
        cols=list(EMBED_COLS),
        force=force,
        timer=timer,
    )
