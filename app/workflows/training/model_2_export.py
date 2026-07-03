"""Training workflow — export fitted model_2 artifacts (IF_ERROR classifier)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from bootstrap import ensure_app_on_path
from workflows.training.config import (
    MODEL_2_BACKEND,
    MODEL_2_DIR,
    MODEL_2_TARGET_COL,
)

ensure_app_on_path(__file__)

MANIFEST_NAME = "manifest.json"
CLASSIFIER_NAME = "classifier.joblib"


def save_model_2(
    classifier,
    feature_columns: list[str],
    embed_cols: list[str],
    *,
    model_2_dir: Path = MODEL_2_DIR,
    backend: str = MODEL_2_BACKEND,
    target_col: str = MODEL_2_TARGET_COL,
    pred_col: str = "PRED_IF_ERROR",
    proba_col: str = "PROBA_IF_ERROR",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist the chosen model_2 backend + manifest under `model_2_dir`."""
    model_2_dir = Path(model_2_dir)
    model_2_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_2_dir / CLASSIFIER_NAME)

    manifest: dict[str, Any] = {
        "backend": backend,
        "target_col": target_col,
        "feature_columns": feature_columns,
        "embed_cols": embed_cols,
        "pred_col": pred_col,
        "proba_col": proba_col,
    }
    if extra:
        manifest.update(extra)
    (model_2_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[training/model_2_export] saved artifact -> {model_2_dir}", flush=True)
    return model_2_dir
