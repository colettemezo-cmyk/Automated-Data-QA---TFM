"""Training workflow — export fitted model_1 artifacts for scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from bootstrap import ensure_app_on_path
from common.config.columns import ML_PC1_SUFFIX, TARGET_COL
from common.features.reduction import (
    LEGACY_PC1_SUBDIR,
    REDUCTION_SUBDIR,
    ReductionStrategy,
    parse_strategy,
)
from workflows.training.config import MODEL_1_BACKEND, MODEL_1_DIR

ensure_app_on_path(__file__)

MANIFEST_NAME = "manifest.json"
CLASSIFIER_NAME = "classifier.joblib"
PC1_SUBDIR = LEGACY_PC1_SUBDIR  # backward-compat re-export


def model_1_pc1_dir(model_1_dir: Path = MODEL_1_DIR) -> Path:
    """Legacy PC1 axes directory. New training runs write to `reduction/` instead."""
    return Path(model_1_dir) / LEGACY_PC1_SUBDIR


def model_1_reduction_dir(model_1_dir: Path = MODEL_1_DIR) -> Path:
    """Strategy-aware reduction axes directory (preferred)."""
    return Path(model_1_dir) / REDUCTION_SUBDIR


def save_model_1(
    classifier,
    feature_columns: list[str],
    embed_cols: list[str],
    *,
    model_1_dir: Path = MODEL_1_DIR,
    backend: str = MODEL_1_BACKEND,
    pred_col: str = "PRED_IS_OWN_RESTAURANT",
    proba_col: str = "PROBA_IS_OWN_RESTAURANT",
    extra: dict[str, Any] | None = None,
    strategy: ReductionStrategy | str | None = None,
    per_column_reduction_meta: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Persist the fitted model_1 classifier and its inference manifest.

    The manifest now records the reduction strategy + per-column K /
    cumulative variance, so QA and model_2 can rebuild the exact same
    feature columns. Legacy keys (`pc1_suffix`, `pc1_dir`) are kept so the
    older inference path keeps working when the new keys are absent.
    """
    model_1_dir = Path(model_1_dir)
    model_1_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_1_dir / CLASSIFIER_NAME)

    strategy_obj: ReductionStrategy = (
        strategy if isinstance(strategy, ReductionStrategy)
        else parse_strategy(strategy) if isinstance(strategy, str)
        else parse_strategy("pc1")
    )

    manifest: dict[str, Any] = {
        "backend": backend,
        "target_col": TARGET_COL,
        "feature_columns": feature_columns,
        "embed_cols": embed_cols,
        # Legacy keys (kept for old QA / model_2 loaders).
        "pc1_suffix": ML_PC1_SUFFIX,
        "pc1_dir": str(model_1_pc1_dir(model_1_dir)),
        # New strategy-aware keys.
        "reduction_strategy": strategy_obj.name,
        "reduction": strategy_obj.to_manifest_dict(),
        "reduction_dir": str(model_1_reduction_dir(model_1_dir)),
        "reduction_per_column": dict(per_column_reduction_meta or {}),
        "pred_col": pred_col,
        "proba_col": proba_col,
    }
    if extra:
        manifest.update(extra)
    (model_1_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[training/model_1_export] saved artifact -> {model_1_dir} "
        f"(strategy={strategy_obj.name!r}, "
        f"n_features={len(feature_columns)})",
        flush=True,
    )
    return model_1_dir
