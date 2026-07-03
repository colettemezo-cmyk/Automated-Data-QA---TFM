"""QA workflow — stage 1: CSV to parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bootstrap import ensure_app_on_path
from common.features.preprocess import add_derived_features, derive_extra_features
from common.storage.parquet_io import atomic_write_parquet, parquet_is_readable

ensure_app_on_path(__file__)


def build_parquet_from_csv(
    csv_path: Path,
    parquet_path: Path,
    force: bool = False,
) -> None:
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_up_to_date = (
        not force
        and parquet_path.exists()
        and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime
    )
    if parquet_up_to_date and not parquet_is_readable(parquet_path):
        print(f"[qa/ingest] {parquet_path} corrupted; re-parsing from CSV.")
        parquet_up_to_date = False

    if parquet_up_to_date:
        print(f"[qa/ingest] {parquet_path} is up to date, skipping CSV parse.")
        derive_extra_features(parquet_path)
        return

    print(f"[qa/ingest] reading {csv_path} ...")
    df = pl.read_csv(
        csv_path,
        schema_overrides={
            "IS_DATAGROUP_SECTION_RESTAURANT": pl.Boolean,
            "QTD_TOTAL_ITEMS": pl.Float64,
            "QTD_SECTION_ITEMS": pl.Float64,
            "IS_TOPTIER_RESTAURANT": pl.Boolean,
            "RESTAURANT_AVG_RATING": pl.Float64,
            "IS_OWN_RESTAURANT": pl.Boolean,
        },
    ).with_columns(
        pl.col("START_EXECUTION_DATETIME").str.strptime(pl.Datetime)
    )
    df = add_derived_features(df)
    atomic_write_parquet(df, parquet_path)
    print(f"[qa/ingest] wrote {parquet_path} ({df.height:,} rows, {df.width} cols).")
