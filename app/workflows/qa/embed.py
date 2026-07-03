"""QA workflow — stage 2: embeddings (auto-refresh when stale or row-mismatched)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from bootstrap import ensure_app_on_path
from common.config.columns import EMBED_COLS
from common.features.embeddings import embed_text_columns as _embed

ensure_app_on_path(__file__)


def embed_cols_present(parquet_path: Path) -> list[str]:
    """`EMBED_COLS` that are actually present in the given parquet's schema."""
    schema = pl.read_parquet_schema(parquet_path)
    return [c for c in EMBED_COLS if c in schema]


def embeddings_need_refresh(
    parquet_path: Path,
    embed_dir: Path,
    *,
    cols: list[str] | None = None,
) -> bool:
    """True if sidecars are missing, outdated vs parquet, or wrong row count."""
    parquet_path = Path(parquet_path)
    embed_dir = Path(embed_dir)
    if not parquet_path.exists():
        return True
    cols = cols or embed_cols_present(parquet_path)
    if not cols:
        return False
    n_parquet = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    parquet_mtime = parquet_path.stat().st_mtime
    for col in cols:
        emb_path = embed_dir / f"{col}_EMBEDDING.parquet"
        pc1_path = embed_dir / f"{col}_EMB_PC1.parquet"
        if not emb_path.exists() or not pc1_path.exists():
            return True
        if emb_path.stat().st_mtime < parquet_mtime:
            return True
        if pq.ParquetFile(emb_path).metadata.num_rows != n_parquet:
            return True
    return False


def embed_qa_columns(
    parquet_path: Path,
    embed_dir: Path,
    *,
    force: bool = False,
    timer=None,
) -> None:
    """Embed QA text columns; re-embed automatically when sidecars are stale."""
    parquet_path = Path(parquet_path)
    embed_dir = Path(embed_dir)
    cols = embed_cols_present(parquet_path)
    if not cols:
        print("[qa/embed] no EMBED_COLS present in parquet; skipping embeddings.", flush=True)
        return

    auto_refresh = embeddings_need_refresh(parquet_path, embed_dir, cols=cols)
    if auto_refresh and not force:
        print(
            "[qa/embed] embeddings missing or out of date for this parquet — "
            f"building under {embed_dir}",
            flush=True,
        )
    elif force:
        print(f"[qa/embed] --force-embed: rebuilding under {embed_dir}", flush=True)

    _embed(
        parquet_path=parquet_path,
        embed_dir=embed_dir,
        cols=cols,
        force=force or auto_refresh,
        timer=timer,
    )
