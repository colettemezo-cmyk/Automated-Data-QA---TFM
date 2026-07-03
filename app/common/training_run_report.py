"""Structured documentation of model training/evaluation/comparison runs.

Every classifier-training run (`model_1`, `model_2`, ...) emits a `TrainingRunReport`
folder under `<runs_dir>/<run_id>_<run_name>/` containing:

  * `run_report.json` — machine-readable record of *everything* observed during the
    run: source data, split policy, feature list, per-model hyperparameters,
    fit/predict timings, full metrics, classification report, confusion matrix,
    top feature importances, and the chosen winning backend.
  * `run_report.md` — human-readable summary of the same content for quick diffing
    between runs.

Reports never overwrite each other (timestamp-prefixed), so we can compare exact
values across iterations later without re-running training.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bootstrap import ensure_app_on_path

ensure_app_on_path(__file__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class ModelRecord:
    name: str
    backend: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float | None]
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    fit_seconds: float
    predict_seconds: float
    n_train_rows: int
    n_test_rows: int
    feature_importances_top: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class TrainingRunReport:
    """Collects every piece of context for a single training run."""

    def __init__(self, run_name: str, runs_dir: Path) -> None:
        self.run_name = run_name
        self.run_id = _run_id_now()
        self.dir = Path(runs_dir) / f"{self.run_id}_{run_name}"
        self._started_perf = time.perf_counter()
        self._payload: dict[str, Any] = {
            "run_name": run_name,
            "run_id": self.run_id,
            "started_at_utc": _utcnow_iso(),
            "finished_at_utc": None,
            "elapsed_seconds": None,
            "config": {},
            "dataset": {},
            "split": {},
            "feature_columns": [],
            "leakage_guard": {
                "excluded_columns": [],
                "checked_columns": [],
            },
            "models": [],
            "comparison": {},
            "notes": [],
        }

    @property
    def report_path(self) -> Path:
        return self.dir / "run_report.json"

    @property
    def markdown_path(self) -> Path:
        return self.dir / "run_report.md"

    def record_config(self, **kwargs: Any) -> None:
        self._payload["config"].update(kwargs)

    def record_dataset(self, **kwargs: Any) -> None:
        self._payload["dataset"].update(kwargs)

    def record_split(self, **kwargs: Any) -> None:
        self._payload["split"].update(kwargs)

    def record_feature_columns(self, feature_columns: Iterable[str]) -> None:
        self._payload["feature_columns"] = list(feature_columns)

    def record_leakage_guard(
        self,
        *,
        excluded_columns: Iterable[str],
        checked_columns: Iterable[str],
    ) -> None:
        self._payload["leakage_guard"] = {
            "excluded_columns": list(excluded_columns),
            "checked_columns": list(checked_columns),
        }

    def record_model(
        self,
        *,
        name: str,
        backend: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float | None],
        classification_report: dict[str, Any],
        confusion_matrix: list[list[int]],
        fit_seconds: float,
        predict_seconds: float,
        n_train_rows: int,
        n_test_rows: int,
        feature_importances_top: list[dict[str, Any]] | None = None,
        notes: list[str] | None = None,
    ) -> ModelRecord:
        rec = ModelRecord(
            name=name,
            backend=backend,
            hyperparameters=dict(hyperparameters),
            metrics=dict(metrics),
            classification_report=classification_report,
            confusion_matrix=[list(map(int, row)) for row in confusion_matrix],
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            n_train_rows=int(n_train_rows),
            n_test_rows=int(n_test_rows),
            feature_importances_top=list(feature_importances_top or []),
            notes=list(notes or []),
        )
        self._payload["models"].append(rec.__dict__)
        return rec

    def record_comparison(
        self,
        *,
        primary_metric: str,
        ranking: list[dict[str, Any]],
        best_model: str,
        selected_backend: str | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self._payload["comparison"] = {
            "primary_metric": primary_metric,
            "ranking": ranking,
            "best_model": best_model,
            "selected_backend": selected_backend,
            "notes": list(notes or []),
        }

    def add_note(self, msg: str) -> None:
        self._payload["notes"].append(msg)

    def finalize_and_save(self) -> Path:
        elapsed = time.perf_counter() - self._started_perf
        self._payload["finished_at_utc"] = _utcnow_iso()
        self._payload["elapsed_seconds"] = round(elapsed, 3)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(self._payload, indent=2, default=_json_default),
            encoding="utf-8",
        )
        self.markdown_path.write_text(
            _render_markdown(self._payload), encoding="utf-8"
        )
        print(
            f"[training_run_report] wrote {self.report_path} (+ run_report.md)",
            flush=True,
        )
        return self.dir


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Training run report — {payload['run_name']}")
    lines.append("")
    lines.append(f"- **Run ID:** `{payload['run_id']}`")
    lines.append(f"- **Started (UTC):** {payload['started_at_utc']}")
    lines.append(f"- **Finished (UTC):** {payload['finished_at_utc']}")
    lines.append(f"- **Elapsed:** {payload['elapsed_seconds']} s")
    lines.append("")

    if payload.get("config"):
        lines.append("## Configuration")
        lines.append("")
        for k, v in payload["config"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    if payload.get("dataset"):
        lines.append("## Dataset")
        lines.append("")
        for k, v in payload["dataset"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    if payload.get("split"):
        lines.append("## Split")
        lines.append("")
        for k, v in payload["split"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    guard = payload.get("leakage_guard") or {}
    if guard.get("excluded_columns") or guard.get("checked_columns"):
        lines.append("## Leakage guard")
        lines.append("")
        if guard.get("excluded_columns"):
            lines.append("**Excluded from features:**")
            for col in guard["excluded_columns"]:
                lines.append(f"- `{col}`")
            lines.append("")
        if guard.get("checked_columns"):
            lines.append("**Asserted absent from feature matrix:**")
            for col in guard["checked_columns"]:
                lines.append(f"- `{col}`")
            lines.append("")

    if payload.get("feature_columns"):
        lines.append(
            f"## Features used ({len(payload['feature_columns'])})"
        )
        lines.append("")
        for col in payload["feature_columns"]:
            lines.append(f"- `{col}`")
        lines.append("")

    if payload.get("models"):
        lines.append("## Model comparison")
        lines.append("")
        lines.append(
            "| model | accuracy | precision | recall | f1 | roc_auc | fit_s | predict_s |"
        )
        lines.append(
            "|-------|---------:|----------:|-------:|----:|--------:|------:|----------:|"
        )
        for m in payload["models"]:
            metrics = m.get("metrics", {})
            lines.append(
                f"| `{m['name']}` "
                f"| {_fmt_metric(metrics.get('accuracy'))} "
                f"| {_fmt_metric(metrics.get('precision'))} "
                f"| {_fmt_metric(metrics.get('recall'))} "
                f"| {_fmt_metric(metrics.get('f1'))} "
                f"| {_fmt_metric(metrics.get('roc_auc'))} "
                f"| {m['fit_seconds']:.2f} "
                f"| {m['predict_seconds']:.2f} |"
            )
        lines.append("")

        cmp = payload.get("comparison") or {}
        if cmp:
            lines.append(
                f"**Best by `{cmp.get('primary_metric', '?')}`:** "
                f"`{cmp.get('best_model', '?')}`  "
            )
            if cmp.get("selected_backend"):
                lines.append(
                    f"**Backend persisted as model artifact:** "
                    f"`{cmp['selected_backend']}`"
                )
            lines.append("")

        for m in payload["models"]:
            lines.append(f"### `{m['name']}`")
            lines.append("")
            lines.append(f"- backend: `{m['backend']}`")
            lines.append(f"- train rows: `{m['n_train_rows']:,}`")
            lines.append(f"- test rows: `{m['n_test_rows']:,}`")
            lines.append(f"- fit time: `{m['fit_seconds']:.2f} s`")
            lines.append(f"- predict time: `{m['predict_seconds']:.2f} s`")
            if m.get("hyperparameters"):
                lines.append("- hyperparameters:")
                for k, v in m["hyperparameters"].items():
                    lines.append(f"  - `{k}`: `{v}`")
            cm = m.get("confusion_matrix") or []
            if cm:
                lines.append("- confusion matrix `[true][pred]`:")
                for row in cm:
                    lines.append(f"  - `{row}`")
            fi = m.get("feature_importances_top") or []
            if fi:
                lines.append("- top feature importances:")
                for entry in fi:
                    lines.append(
                        f"  - `{entry['feature']}`: `{entry['importance']:.4f}`"
                    )
            lines.append("")

    if payload.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for note in payload["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)
