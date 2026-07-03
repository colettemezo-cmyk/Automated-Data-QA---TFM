"""QA workflow — load model_2 and flag likely model_1 errors (IF_ERROR).

This is the *inference* counterpart to `workflows.training.model_2`, which only
*trains* the IF_ERROR classifier. Here we take a parquet that model_1 has
already scored (so it carries the same tabular + embedding columns) and apply
the fitted model_2 to predict, per row, whether model_1 likely made a mistake.

Feature matrix is built by the exact same shared helper model_1 and model_2
training use (`common.features.inference_matrix`), so the features model_2 sees
at scoring time match the ones it was trained on byte-for-byte.

Decision threshold: model_2 is trained recall-first (catch nearly every
IF_ERROR=True row). The manifest persists a `decision_threshold` and we flag
`PRED_IF_ERROR = PROBA_IF_ERROR >= decision_threshold` rather than the default
0.5 argmax. If no threshold was persisted we fall back to the classifier's own
`predict` (0.5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from bootstrap import ensure_app_on_path
from common.config.columns import EMBED_COLS
from common.pipeline_timing import step as pipeline_step
from common.storage.parquet_io import atomic_write_parquet
from workflows.qa.config import (
    MODEL_2_DIR,
    MODEL_2_PRED_COL,
    MODEL_2_PROBA_COL,
)
from workflows.qa.inference_features import build_qa_feature_matrix

ensure_app_on_path(__file__)

MANIFEST_NAME = "manifest.json"
CLASSIFIER_NAME = "classifier.joblib"


@dataclass
class Model2ScoreResult:
    """Outcome of scoring a parquet with model_2.

    Holds enough state (classifier, feature matrix, predictions) for a
    downstream SHAP step to run without rebuilding the feature matrix.
    """

    output_path: Path
    classifier: Any
    manifest: dict[str, Any]
    feature_columns: list[str]
    X: np.ndarray
    pred: np.ndarray
    proba: np.ndarray | None
    decision_threshold: float | None
    n_flagged: int


def load_model_2(model_2_dir: Path = MODEL_2_DIR) -> tuple[Any, dict[str, Any]]:
    model_2_dir = Path(model_2_dir)
    manifest_path = model_2_dir / MANIFEST_NAME
    classifier_path = model_2_dir / CLASSIFIER_NAME
    if not manifest_path.exists() or not classifier_path.exists():
        raise FileNotFoundError(
            f"model_2 not found under {model_2_dir}. Train it first:\n"
            "  python app/training_workflow.py --only-model-2"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return joblib.load(classifier_path), manifest


def score_parquet_with_model_2(
    scored_path: Path,
    embed_dir: Path,
    *,
    output_path: Path | None = None,
    model_2_dir: Path = MODEL_2_DIR,
    timer=None,
) -> Model2ScoreResult:
    """Apply model_2 to a model_1-scored parquet, flagging likely errors.

    Parameters
    ----------
    scored_path:
        Parquet produced by `score_parquet_with_model_1` (already carries the
        tabular feature columns; embeddings live in `embed_dir`).
    embed_dir:
        Embedding sidecar directory matching `scored_path` row-for-row.
    output_path:
        Where to write the augmented parquet. Defaults to rewriting
        `scored_path` in place (atomically).
    """
    scored_path = Path(scored_path)
    embed_dir = Path(embed_dir)
    classifier, manifest = load_model_2(model_2_dir)

    feature_columns = list(manifest["feature_columns"])
    embed_cols = list(manifest.get("embed_cols", EMBED_COLS))
    pred_col = manifest.get("pred_col", MODEL_2_PRED_COL)
    proba_col = manifest.get("proba_col", MODEL_2_PROBA_COL)
    # model_2 shares model_1's frozen reduction axes. The manifest records the
    # model_1 reduction directory (legacy `pc1` key name) — `_resolve_artifacts`
    # in the shared matrix builder maps it back to the model dir transparently.
    reduction_dir_str = (
        manifest.get("model_1_pc1_dir")
        or manifest.get("reduction_dir")
        or manifest.get("pc1_dir")
    )
    if not reduction_dir_str:
        raise ValueError(
            "model_2 manifest is missing the model_1 reduction directory "
            "(expected one of model_1_pc1_dir / reduction_dir / pc1_dir). "
            "Re-export model_2 with an up-to-date trainer."
        )
    pc1_dir = Path(reduction_dir_str)
    decision_threshold = manifest.get("decision_threshold")

    with pipeline_step(
        "qa.4a.features",
        "Build model_2 scoring features (shared frozen reduction)",
        "workflows.qa.inference_features.build_qa_feature_matrix",
        timer=timer,
    ):
        features = build_qa_feature_matrix(
            scored_path,
            embed_dir,
            feature_columns,
            embed_cols,
            pc1_dir,
            timer=timer,
        )
        X = features.to_numpy()

    with pipeline_step(
        "qa.4b.predict",
        "model_2 predict (IF_ERROR)",
        "classifier.predict_proba",
        timer=timer,
    ):
        proba = (
            classifier.predict_proba(X)[:, 1].astype(np.float32)
            if hasattr(classifier, "predict_proba")
            else None
        )
        if proba is not None and decision_threshold is not None:
            pred = (proba >= float(decision_threshold)).astype(np.int8)
        else:
            pred = classifier.predict(X).astype(np.int8)

    df = pl.read_parquet(scored_path)
    new_cols = [pl.Series(pred_col, pred)]
    if proba is not None:
        new_cols.append(pl.Series(proba_col, proba))
    df = df.with_columns(new_cols)

    n_flagged = int(pred.sum())
    thr_txt = (
        f"threshold {float(decision_threshold):.6f}"
        if decision_threshold is not None
        else "argmax 0.5"
    )
    print(
        f"[qa/model_2] {pred_col}=1 for {n_flagged:,}/{df.height:,} rows "
        f"({thr_txt}).",
        flush=True,
    )

    if output_path is None:
        output_path = scored_path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pipeline_step(
        "qa.4c.write",
        "Write model_2-scored parquet",
        "common.storage.parquet_io.atomic_write_parquet",
        timer=timer,
    ):
        atomic_write_parquet(df, output_path)

    added = [pred_col]
    if proba is not None:
        added.append(proba_col)
    print(
        f"[qa/model_2] scored {df.height:,} rows -> {output_path} "
        f"(added {', '.join(added)}).",
        flush=True,
    )

    return Model2ScoreResult(
        output_path=output_path,
        classifier=classifier,
        manifest=manifest,
        feature_columns=feature_columns,
        X=X,
        pred=pred,
        proba=proba,
        decision_threshold=(
            float(decision_threshold) if decision_threshold is not None else None
        ),
        n_flagged=n_flagged,
    )
