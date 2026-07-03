"""Progress + elapsed-time logging for multi-stage pipeline runs.

Prints where code is running, predicted duration per step, actual duration,
and a rolling ETA for the rest of the run.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable

# Rough defaults (seconds) — updated as similar steps finish in the same run.
DEFAULT_PREDICTIONS: dict[str, float] = {
    "1.parquet": 120.0,
    "2.embed.column": 1800.0,
    "2.embed.total": 12600.0,
    "3a.plot_ownership": 30.0,
    "3b.plot_corr": 600.0,
    "4.1.model_input.cache_hit": 5.0,
    "4.1.model_input.build": 7200.0,
    "4.1.model_input.tabular": 60.0,
    "4.1.model_input.pc1.column": 900.0,
    "4.1.model_input.pc1.materialize": 1200.0,
    "4.1.model_input.write": 30.0,
    "4.2.train.lightgbm": 180.0,
    "4.2.train.xgboost": 180.0,
    "4.2.load_matrix": 30.0,
    "4.3.model_2.build_features": 1800.0,
    "4.3.model_2.load_target": 30.0,
    "4.3.model_2.train.lightgbm": 180.0,
    "4.3.model_2.train.xgboost": 180.0,
    "5.model_2": 3600.0,
    "qa.1.parquet": 120.0,
    "qa.2.embed": 12600.0,
    "qa.3.model_1": 3600.0,
    "qa.3a.features": 1800.0,
    "qa.3b.predict": 120.0,
    "qa.3c.write": 60.0,
    "qa.4.model_2": 2400.0,
    "qa.4a.features": 1800.0,
    "qa.4b.predict": 120.0,
    "qa.4c.write": 60.0,
    "qa.5.shap": 600.0,
    "qa.5a.shap_values": 540.0,
    "qa.5b.shap_plots": 30.0,
    "qa.pc1.column": 900.0,
}


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:5.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m:3d}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:2d}h {m:02d}m {s:02d}s"


def _fmt_delta(predicted: float | None, actual: float) -> str:
    if predicted is None:
        return "estimate n/a"
    delta = actual - predicted
    sign = "+" if delta >= 0 else ""
    return f"delta {sign}{_fmt_duration(delta)} vs estimate"


@dataclass
class _StepRecord:
    step_id: str
    title: str
    location: str
    predicted_sec: float | None
    start: float
    end: float | None = None

    @property
    def actual_sec(self) -> float | None:
        if self.end is None:
            return None
        return self.end - self.start


class PipelineTimer:
    """One run of the pipeline; use `step()` as a context manager per sub-step."""

    _active: PipelineTimer | None = None

    def __init__(self, run_name: str = "pipeline") -> None:
        self.run_name = run_name
        self.run_start = time.perf_counter()
        self.steps: list[_StepRecord] = []
        self._predictions: dict[str, float] = dict(DEFAULT_PREDICTIONS)
        self._planned: list[tuple[str, str, str, str | None]] = []

    @classmethod
    def active(cls) -> PipelineTimer | None:
        return cls._active

    def set_prediction(self, step_id: str, seconds: float) -> None:
        self._predictions[step_id] = seconds

    def learn_prediction(self, step_id: str, actual_sec: float, blend: float = 0.5) -> None:
        """Blend actual into default so later steps in this run get better ETAs."""
        prev = self._predictions.get(step_id, actual_sec)
        self._predictions[step_id] = prev * (1 - blend) + actual_sec * blend

    def register_planned(self, step_id: str, title: str, location: str) -> None:
        self._planned.append((step_id, title, location, None))

    def remaining_eta(self) -> float:
        done_ids = {s.step_id for s in self.steps if s.end is not None}
        total = 0.0
        for step_id, _, _, _ in self._planned:
            if step_id not in done_ids:
                total += self._predictions.get(step_id, 0.0)
        return total

    def _log(self, msg: str) -> None:
        elapsed_run = time.perf_counter() - self.run_start
        print(f"[{self.run_name}] {msg}  |  run elapsed {_fmt_duration(elapsed_run)}", flush=True)

    def begin_run(self, stages: list[str]) -> None:
        PipelineTimer._active = self
        self._log(f"{'=' * 60}")
        self._log(f"START {self.run_name}")
        for line in stages:
            self._log(f"  planned: {line}")
        self._log(f"{'=' * 60}")

    def end_run(self) -> None:
        total = time.perf_counter() - self.run_start
        self._log(f"{'=' * 60}")
        self._log(f"FINISH {self.run_name}  |  total {_fmt_duration(total)}")
        self._log("Step summary (actual vs predicted):")
        for s in self.steps:
            act = s.actual_sec
            if act is None:
                continue
            pred_s = f"{_fmt_duration(s.predicted_sec)}" if s.predicted_sec else "n/a"
            self._log(
                f"  {s.step_id:<28} {s.title:<22} actual {_fmt_duration(act):>10}  "
                f"predicted {pred_s:>10}  {_fmt_delta(s.predicted_sec, act)}"
            )
        self._log(f"{'=' * 60}")
        if PipelineTimer._active is self:
            PipelineTimer._active = None

    def step(
        self,
        step_id: str,
        title: str,
        location: str,
        *,
        predicted_sec: float | None = None,
    ):
        return _StepContext(self, step_id, title, location, predicted_sec)


@dataclass
class _StepContext:
    timer: PipelineTimer
    step_id: str
    title: str
    location: str
    predicted_sec: float | None

    def __enter__(self) -> _StepContext:
        if self.predicted_sec is None:
            self.predicted_sec = self.timer._predictions.get(self.step_id)
        rec = _StepRecord(
            step_id=self.step_id,
            title=self.title,
            location=self.location,
            predicted_sec=self.predicted_sec,
            start=time.perf_counter(),
        )
        self.timer.steps.append(rec)
        self._record = rec
        pred = _fmt_duration(self.predicted_sec) if self.predicted_sec else "n/a"
        eta = self.timer.remaining_eta()
        self.timer._log(
            f">>> BEGIN {self.step_id} | {self.title}\n"
            f"    at {self.location}\n"
            f"    predicted {_fmt_duration(self.predicted_sec) if self.predicted_sec else 'n/a':>10}"
            f"  |  remaining ~{_fmt_duration(eta)} (planned steps)"
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._record.end = time.perf_counter()
        actual = self._record.actual_sec or 0.0
        self.timer.learn_prediction(self.step_id, actual)
        status = "OK" if exc_type is None else f"FAILED ({exc_type.__name__})"
        self.timer._log(
            f"<<< END   {self.step_id} | {self.title} | {status}\n"
            f"    actual   {_fmt_duration(actual):>10}  |  {_fmt_delta(self.predicted_sec, actual)}"
        )
        return False


def step(
    step_id: str,
    title: str,
    location: str,
    *,
    predicted_sec: float | None = None,
    timer: PipelineTimer | None = None,
):
    """Convenience: use active timer or no-op if none."""
    t = timer or PipelineTimer.active()
    if t is None:
        return _NullContext()
    return t.step(step_id, title, location, predicted_sec=predicted_sec)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False
