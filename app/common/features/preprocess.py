"""Derived parquet columns (IS_ODM, EXECUTION_ROW_COUNT, etc.)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bootstrap import ensure_app_on_path
from common.storage.parquet_io import atomic_write_parquet

ensure_app_on_path(__file__)

DERIVED_FEATURE_COLS = [
    "IS_ODM",
    "IS_QCA",
    "EXECUTION_ROW_COUNT",
    "BRAND_COUNT",
]


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    if "IS_ODM" not in df.columns and "PIPELINE_FLAG" in df.columns:
        # The derived column is IS_ODM (project-wide rename), but raw input
        # data still tags ODM rows as PIPELINE_FLAG == "FSA". When upstream
        # finally switches the tag value from "FSA" to "ODM" in the source
        # CSV, update this literal accordingly.
        exprs.append(pl.col("PIPELINE_FLAG").eq("FSA").alias("IS_ODM"))
    if "IS_QCA" not in df.columns and "PIPELINE_FLAG" in df.columns:
        exprs.append(pl.col("PIPELINE_FLAG").eq("QCA").alias("IS_QCA"))
    if "EXECUTION_ROW_COUNT" not in df.columns and "EXECUTION_ID" in df.columns:
        exprs.append(
            pl.len().over("EXECUTION_ID").cast(pl.UInt32).alias("EXECUTION_ROW_COUNT")
        )
    if "BRAND_COUNT" not in df.columns and "RESTAURANT_BRAND_NAMES" in df.columns:
        # RESTAURANT_BRAND_NAMES is a JSON-array string, e.g.
        # '["Coca-Cola", "Fanta"]'. Each brand is wrapped in a pair of double
        # quotes, so the number of brands is (count of '"') // 2. This is
        # robust to commas inside brand names and to whitespace/newlines, and
        # yields 0 for empty arrays ("[]") or null.
        exprs.append(
            pl.col("RESTAURANT_BRAND_NAMES")
            .str.count_matches('"')
            .floordiv(2)
            .fill_null(0)
            .cast(pl.Int32)
            .alias("BRAND_COUNT")
        )
    if exprs:
        df = df.with_columns(exprs)
    return df


def derive_extra_features(parquet_path: Path) -> None:
    if not parquet_path.exists():
        return
    schema = pl.read_parquet_schema(parquet_path)
    missing = [c for c in DERIVED_FEATURE_COLS if c not in schema]
    if not missing:
        return
    print(f"[stage1.5] adding derived features {missing} to {parquet_path} ...")
    df = pl.read_parquet(parquet_path)
    df = add_derived_features(df)
    atomic_write_parquet(df, parquet_path)
    print(f"[stage1.5] wrote {parquet_path} (now {df.width} cols).")
