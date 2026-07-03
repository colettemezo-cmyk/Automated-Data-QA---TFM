"""CSV logs that accumulate across runs for comparing reduction strategies.

Every model_1 training run, model_2 training run, and QA scoring run appends
one or more rows to a CSV under `data/ml/comparison/`. This gives a single
diff-friendly artifact for comparing strategies (`pc1`, `top5`,
`adaptive_0.90`, `adaptive_0.95`, `raw`) across runs along the four axes
the team cares about:

  * Training time           — `fit_seconds` / `predict_seconds` per backend.
  * Running time (QA)       — `qa_*_seconds` per stage in the QA log.
  * Precision               — accuracy / precision / recall / F1 / ROC-AUC.
  * Information content     — per-column K + cumulative explained variance,
                              plus an aggregate `mean_cumulative_variance`.

Logs (created on first write):
  * `data/ml/comparison/training_runs.csv`         — model_1 + model_2 trains.
  * `data/ml/comparison/qa_runs.csv`               — QA scoring runs.
  * `data/ml/comparison/reduction_per_column.csv`  — per-(run, column, strategy)
                                                     k + cumulative variance.

The writer is forward-compatible: if a future run adds new metric columns,
existing rows just leave those cells empty. We use `csv.DictWriter` with a
stable column order kept in `*_COLUMNS`. Unrecognised keys are dropped
(logged once) so a typo can't quietly produce garbage rows.
"""

from __future__ import annotations

import csv
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bootstrap import ensure_app_on_path

ensure_app_on_path(__file__)

from common.config.paths import PROJECT_ROOT  # noqa: E402

COMPARISON_DIR = PROJECT_ROOT / "data" / "ml" / "comparison"
TRAINING_LOG = COMPARISON_DIR / "training_runs.csv"
QA_LOG = COMPARISON_DIR / "qa_runs.csv"
PER_COLUMN_LOG = COMPARISON_DIR / "reduction_per_column.csv"

# Stable column orders. New metric columns may be appended at the END of the
# tuple without invalidating old rows; do NOT reorder, do NOT delete columns.
TRAINING_COLUMNS: tuple[str, ...] = (
    # Identity / context
    "timestamp_utc",
    "run_id",
    "run_name",                # "model_1" | "model_2" | "compare_reductions/<id>"
    "host",
    "git_sha",                 # short SHA if discoverable; else ""
    # Strategy
    "reduction_strategy",      # e.g. "pc1", "top5", "adaptive_0.90", "raw"
    "reduction_mode",          # "fixed_k" | "adaptive" | "raw"
    "variance_target",
    "k_max",
    "fixed_k",
    # Dataset
    "max_executions",
    "n_rows",
    "n_train_rows",
    "n_test_rows",
    "n_executions",
    "n_features",              # total fed to the classifier
    "n_reduction_features",    # subset coming from embedding reduction
    # Information content (aggregated across embed cols)
    "mean_k_per_column",
    "max_k_per_column",
    "total_k",
    "mean_cumulative_variance",
    "min_cumulative_variance",
    "max_cumulative_variance",
    "per_column_k_json",       # `{col: k}` as JSON string for downstream diff
    "per_column_cumvar_json",  # `{col: cum_var}` as JSON string
    # Model + metrics (one row per backend)
    "backend",
    "n_estimators",
    "fit_seconds",
    "predict_seconds",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    # Paths
    "model_dir",
    "model_input_parquet",
    "run_report_dir",
    "reduction_dir",
    # Notes
    "selected_as_canonical",   # "true" / "false" / ""
    "notes",
)

QA_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "run_id",
    "host",
    "git_sha",
    "input_csv",
    "input_parquet",
    "n_rows_scored",
    "model_dir",
    "reduction_strategy",
    "reduction_mode",
    "variance_target",
    "k_max",
    "n_features",
    "n_reduction_features",
    "mean_k_per_column",
    "total_k",
    # Per-stage timings — populated when the pipeline timer is available.
    "qa_parquet_seconds",
    "qa_embed_seconds",
    "qa_features_seconds",
    "qa_predict_seconds",
    "qa_write_seconds",
    "qa_total_seconds",
    # Quality metrics — populated when ground truth (`IS_OWN_RESTAURANT`) is
    # present in the input parquet, leaving the cells empty otherwise.
    "n_rows_with_ground_truth",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "n_if_error_true",
    # Paths
    "scored_parquet",
    "notes",
)

PER_COLUMN_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "run_id",
    "run_name",
    "reduction_strategy",
    "embed_col",
    "k",
    "n_features",
    "achieved_cumulative_variance",
    "variance_target",
    "k_max",
    "explained_variance_ratios_json",
)


def _ensure_dir() -> Path:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    return COMPARISON_DIR


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_host() -> str:
    try:
        return socket.gethostname() or os.environ.get("COMPUTERNAME", "")
    except Exception:  # noqa: BLE001
        return ""


def _git_short_sha() -> str:
    """Best-effort short SHA from .git/HEAD. Never raises."""
    try:
        head = (PROJECT_ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            sha_path = PROJECT_ROOT / ".git" / ref
            if sha_path.exists():
                return sha_path.read_text(encoding="utf-8").strip()[:12]
        return head[:12]
    except Exception:  # noqa: BLE001
        return ""


def _json_dump_safe(obj: Any) -> str:
    """Compact, deterministic JSON. Falls back to str() on non-serialisable."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:  # noqa: BLE001
        return str(obj)


def _coerce_row(columns: Sequence[str], row: Mapping[str, Any]) -> dict[str, Any]:
    """Project `row` onto `columns`; warn once per unknown key per process."""
    out = {col: row.get(col, "") for col in columns}
    unknown = sorted(set(row.keys()) - set(columns))
    if unknown:
        _warn_unknown_keys(tuple(unknown))
    return out


_WARNED_UNKNOWN: set[tuple[str, ...]] = set()


def _warn_unknown_keys(keys: tuple[str, ...]) -> None:
    if keys in _WARNED_UNKNOWN:
        return
    _WARNED_UNKNOWN.add(keys)
    print(
        f"[comparison_log] WARNING: dropping unknown CSV keys {list(keys)} "
        "(extend the *_COLUMNS tuple in app/common/comparison_log.py to keep them).",
        flush=True,
    )


def _append_csv_row(path: Path, columns: Sequence[str], row: Mapping[str, Any]) -> None:
    _ensure_dir()
    data = _coerce_row(columns, row)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        writer.writerow(data)


# ---------------------------------------------------------------------------
# Helpers to compute aggregates from per-column reduction metadata
# ---------------------------------------------------------------------------

def summarise_per_column_meta(
    per_column: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce `{col: {k, achieved_cumulative_variance, ...}}` to scalar aggregates.

    Returns the columns expected by the training/QA logs:
    `mean_k_per_column`, `max_k_per_column`, `total_k`,
    `mean_cumulative_variance`, `min_cumulative_variance`,
    `max_cumulative_variance`, plus JSON-encoded per-column k and cum_var.
    """
    if not per_column:
        return {
            "mean_k_per_column": "",
            "max_k_per_column": "",
            "total_k": "",
            "mean_cumulative_variance": "",
            "min_cumulative_variance": "",
            "max_cumulative_variance": "",
            "per_column_k_json": "{}",
            "per_column_cumvar_json": "{}",
        }
    ks = [int(v.get("k", 0)) for v in per_column.values()]
    cvs = [
        float(v.get("achieved_cumulative_variance", 0.0)) for v in per_column.values()
    ]
    mean_k = sum(ks) / max(len(ks), 1)
    mean_cv = sum(cvs) / max(len(cvs), 1)
    return {
        "mean_k_per_column": round(mean_k, 3),
        "max_k_per_column": max(ks) if ks else "",
        "total_k": sum(ks) if ks else "",
        "mean_cumulative_variance": round(mean_cv, 6),
        "min_cumulative_variance": round(min(cvs), 6) if cvs else "",
        "max_cumulative_variance": round(max(cvs), 6) if cvs else "",
        "per_column_k_json": _json_dump_safe(
            {col: int(meta.get("k", 0)) for col, meta in per_column.items()}
        ),
        "per_column_cumvar_json": _json_dump_safe(
            {
                col: round(float(meta.get("achieved_cumulative_variance", 0.0)), 6)
                for col, meta in per_column.items()
            }
        ),
    }


# ---------------------------------------------------------------------------
# Public append helpers
# ---------------------------------------------------------------------------

def append_training_row(
    *,
    run_id: str,
    run_name: str,
    reduction_strategy_dict: Mapping[str, Any],
    backend: str,
    metrics: Mapping[str, Any],
    per_column_meta: Mapping[str, Mapping[str, Any]] | None,
    fit_seconds: float,
    predict_seconds: float,
    n_estimators: int,
    n_train_rows: int,
    n_test_rows: int,
    n_features: int,
    n_reduction_features: int,
    n_rows: int | None = None,
    n_executions: int | None = None,
    max_executions: int | None = None,
    model_dir: Path | str | None = None,
    model_input_parquet: Path | str | None = None,
    run_report_dir: Path | str | None = None,
    reduction_dir: Path | str | None = None,
    selected_as_canonical: bool | None = None,
    notes: str = "",
) -> Path:
    """Append one row per (training run, backend) to `training_runs.csv`."""
    aggregates = summarise_per_column_meta(per_column_meta or {})
    row: dict[str, Any] = {
        "timestamp_utc": _utcnow_iso(),
        "run_id": run_id,
        "run_name": run_name,
        "host": _safe_host(),
        "git_sha": _git_short_sha(),
        "reduction_strategy": reduction_strategy_dict.get("name", ""),
        "reduction_mode": reduction_strategy_dict.get("mode", ""),
        "variance_target": reduction_strategy_dict.get("variance_target") or "",
        "k_max": reduction_strategy_dict.get("k_max") or "",
        "fixed_k": reduction_strategy_dict.get("fixed_k") or "",
        "max_executions": max_executions if max_executions is not None else "",
        "n_rows": n_rows if n_rows is not None else "",
        "n_train_rows": n_train_rows,
        "n_test_rows": n_test_rows,
        "n_executions": n_executions if n_executions is not None else "",
        "n_features": n_features,
        "n_reduction_features": n_reduction_features,
        "backend": backend,
        "n_estimators": n_estimators,
        "fit_seconds": round(float(fit_seconds), 3),
        "predict_seconds": round(float(predict_seconds), 3),
        "accuracy": _metric(metrics.get("accuracy")),
        "precision": _metric(metrics.get("precision")),
        "recall": _metric(metrics.get("recall")),
        "f1": _metric(metrics.get("f1")),
        "roc_auc": _metric(metrics.get("roc_auc")),
        "model_dir": str(model_dir) if model_dir else "",
        "model_input_parquet": str(model_input_parquet) if model_input_parquet else "",
        "run_report_dir": str(run_report_dir) if run_report_dir else "",
        "reduction_dir": str(reduction_dir) if reduction_dir else "",
        "selected_as_canonical": (
            ""
            if selected_as_canonical is None
            else ("true" if selected_as_canonical else "false")
        ),
        "notes": notes,
    }
    row.update(aggregates)
    _append_csv_row(TRAINING_LOG, TRAINING_COLUMNS, row)
    return TRAINING_LOG


def append_per_column_rows(
    *,
    run_id: str,
    run_name: str,
    reduction_strategy_dict: Mapping[str, Any],
    per_column_meta: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Append one row per (run, embed column) to `reduction_per_column.csv`."""
    if not per_column_meta:
        return PER_COLUMN_LOG
    ts = _utcnow_iso()
    strategy = reduction_strategy_dict.get("name", "")
    variance_target = reduction_strategy_dict.get("variance_target") or ""
    k_max = reduction_strategy_dict.get("k_max") or ""
    for col, meta in per_column_meta.items():
        row = {
            "timestamp_utc": ts,
            "run_id": run_id,
            "run_name": run_name,
            "reduction_strategy": strategy,
            "embed_col": col,
            "k": int(meta.get("k", 0)),
            "n_features": int(meta.get("n_features", meta.get("k", 0))),
            "achieved_cumulative_variance": round(
                float(meta.get("achieved_cumulative_variance", 0.0)), 6
            ),
            "variance_target": variance_target,
            "k_max": k_max,
            "explained_variance_ratios_json": _json_dump_safe(
                meta.get("explained_variance_ratios", [])
            ),
        }
        _append_csv_row(PER_COLUMN_LOG, PER_COLUMN_COLUMNS, row)
    return PER_COLUMN_LOG


def append_qa_row(
    *,
    run_id: str,
    input_csv: Path | str,
    input_parquet: Path | str,
    scored_parquet: Path | str,
    model_dir: Path | str,
    reduction_strategy_dict: Mapping[str, Any],
    n_rows_scored: int,
    n_features: int,
    n_reduction_features: int,
    per_column_meta: Mapping[str, Mapping[str, Any]] | None,
    stage_timings: Mapping[str, float] | None,
    metrics: Mapping[str, Any] | None = None,
    n_rows_with_ground_truth: int | None = None,
    n_if_error_true: int | None = None,
    notes: str = "",
) -> Path:
    """Append one row per QA run to `qa_runs.csv`."""
    aggregates = summarise_per_column_meta(per_column_meta or {})
    metrics = metrics or {}
    stage_timings = stage_timings or {}
    qa_total = (
        sum(float(v) for v in stage_timings.values() if v is not None)
        if stage_timings
        else ""
    )
    row: dict[str, Any] = {
        "timestamp_utc": _utcnow_iso(),
        "run_id": run_id,
        "host": _safe_host(),
        "git_sha": _git_short_sha(),
        "input_csv": str(input_csv),
        "input_parquet": str(input_parquet),
        "n_rows_scored": n_rows_scored,
        "model_dir": str(model_dir),
        "reduction_strategy": reduction_strategy_dict.get("name", ""),
        "reduction_mode": reduction_strategy_dict.get("mode", ""),
        "variance_target": reduction_strategy_dict.get("variance_target") or "",
        "k_max": reduction_strategy_dict.get("k_max") or "",
        "n_features": n_features,
        "n_reduction_features": n_reduction_features,
        "mean_k_per_column": aggregates["mean_k_per_column"],
        "total_k": aggregates["total_k"],
        "qa_parquet_seconds": _seconds(stage_timings.get("qa.1.parquet")),
        "qa_embed_seconds": _seconds(stage_timings.get("qa.2.embed")),
        "qa_features_seconds": _seconds(stage_timings.get("qa.3a.features")),
        "qa_predict_seconds": _seconds(stage_timings.get("qa.3b.predict")),
        "qa_write_seconds": _seconds(stage_timings.get("qa.3c.write")),
        "qa_total_seconds": _seconds(qa_total),
        "n_rows_with_ground_truth": (
            n_rows_with_ground_truth if n_rows_with_ground_truth is not None else ""
        ),
        "accuracy": _metric(metrics.get("accuracy")),
        "precision": _metric(metrics.get("precision")),
        "recall": _metric(metrics.get("recall")),
        "f1": _metric(metrics.get("f1")),
        "roc_auc": _metric(metrics.get("roc_auc")),
        "n_if_error_true": (
            n_if_error_true if n_if_error_true is not None else ""
        ),
        "scored_parquet": str(scored_parquet),
        "notes": notes,
    }
    _append_csv_row(QA_LOG, QA_COLUMNS, row)
    return QA_LOG


def _metric(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


def _seconds(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value


def timings_from_pipeline_timer(timer) -> dict[str, float]:
    """Extract `{step_id: actual_seconds}` from a finished `PipelineTimer`.

    Quietly returns `{}` when `timer` is None or the steps haven't run yet.
    The QA logger uses this to populate per-stage seconds without having
    to plumb explicit return values out of every pipeline_step block.
    """
    if timer is None or not getattr(timer, "steps", None):
        return {}
    out: dict[str, float] = {}
    for step in timer.steps:
        if step.end is None:
            continue
        # If a step_id fires more than once in a run, sum (matches the QA
        # workflow's behaviour for repeated per-column steps).
        out[step.step_id] = out.get(step.step_id, 0.0) + (step.end - step.start)
    return out
