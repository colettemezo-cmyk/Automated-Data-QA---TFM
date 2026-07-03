"""Shared parquet I/O helpers used by every stage that writes parquet.

These exist to make in-place parquet rewrites safe against killed
processes, which used to leave half-written files behind and break
subsequent runs with `parquet: File out of specification: The file
must end with PAR1`.

NOTE: This package is named `storage` (not `io`) to avoid shadowing the
Python stdlib `io` module - that conflict is what bit us with the old
`connectors/snowflake.py` import.
"""

from pathlib import Path

import polars as pl

from bootstrap import ensure_app_on_path

ensure_app_on_path(__file__)


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a polars DataFrame to `path` atomically.

    We write to `<path>.tmp` first and then `Path.replace()` (which is
    atomic on the same filesystem on both POSIX and Windows). An
    interrupted write thus never leaves a half-written file at the
    final path - the worst outcome is a stale `.tmp` next to a
    still-good original, which the next successful write cleans up via
    `replace()`.

    This is the fix for the "File out of specification: The file must
    end with PAR1" failure we hit when an in-place rewrite of the
    parquet was killed mid-flight.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def parquet_is_readable(path: Path) -> bool:
    """Cheap probe: does this parquet have a valid footer / schema?

    `read_parquet_schema` only reads the file footer, not row data, so
    this is fast even for multi-GB parquets. Returns False on any
    error (truncated file, missing PAR1 magic, codec mismatch, ...)
    so callers can treat the file as if it didn't exist and re-parse
    from CSV.
    """
    try:
        pl.read_parquet_schema(path)
        return True
    except Exception:  # noqa: BLE001 - polars raises a few different types here
        return False
