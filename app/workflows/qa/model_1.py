"""QA workflow — load model_1 and classify an embedded dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from bootstrap import ensure_app_on_path
from common.comparison_log import append_qa_row, timings_from_pipeline_timer
from common.config.columns import EMBED_COLS, TARGET_COL
from common.config.paths import PROJECT_ROOT
from common.pipeline_timing import step as pipeline_step
from common.storage.parquet_io import atomic_write_parquet
from workflows.qa.config import (
    IF_ERROR_COL,
    MODEL_1_DIR,
    MODEL_1_PRED_COL,
    MODEL_1_PROBA_COL,
    default_scored_path_for_csv,
)
from workflows.qa.inference_features import build_qa_feature_matrix

ensure_app_on_path(__file__)

MANIFEST_NAME = "manifest.json"
CLASSIFIER_NAME = "classifier.joblib"
LEGACY_PC1_SUBDIR = "pc1"
REDUCTION_SUBDIR = "reduction"


def model_1_pc1_dir(model_1_dir: Path = MODEL_1_DIR) -> Path:
    """Legacy PC1 axes directory (kept for backward-compat resolution)."""
    return Path(model_1_dir) / LEGACY_PC1_SUBDIR


def model_1_reduction_dir(model_1_dir: Path = MODEL_1_DIR) -> Path:
    """Strategy-aware reduction axes directory (preferred)."""
    return Path(model_1_dir) / REDUCTION_SUBDIR


def _have_reduction_artifacts(model_1_dir: Path) -> bool:
    rdir = model_1_reduction_dir(model_1_dir)
    return (rdir / "reduction_manifest.json").exists()


def _have_legacy_pc1_artifacts(model_1_dir: Path) -> bool:
    pc1_dir = model_1_pc1_dir(model_1_dir)
    return pc1_dir.exists() and any(pc1_dir.glob("*_pc1_axis.npy"))


def load_model_1(model_1_dir: Path = MODEL_1_DIR) -> tuple[Any, dict[str, Any]]:
    model_1_dir = Path(model_1_dir)
    manifest_path = model_1_dir / MANIFEST_NAME
    classifier_path = model_1_dir / CLASSIFIER_NAME
    if not manifest_path.exists() or not classifier_path.exists():
        raise FileNotFoundError(
            f"model_1 not found under {model_1_dir}. "
            "Run the training workflow first to export model_1."
        )
    if not (
        _have_reduction_artifacts(model_1_dir)
        or _have_legacy_pc1_artifacts(model_1_dir)
    ):
        raise FileNotFoundError(
            f"model_1 classifier exists but reduction axes are missing under "
            f"{model_1_dir}. Expected either "
            f"{model_1_reduction_dir(model_1_dir)/'reduction_manifest.json'} "
            f"(new layout) or {model_1_pc1_dir(model_1_dir)}/*_pc1_axis.npy "
            "(legacy). Re-run training or "
            "`python scripts/export_model_1_pc1.py --full`."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return joblib.load(classifier_path), manifest


def score_parquet_with_model_1(
    parquet_path: Path,
    embed_dir: Path,
    *,
    output_path: Path | None = None,
    model_1_dir: Path = MODEL_1_DIR,
    timer=None,
) -> Path:
    parquet_path = Path(parquet_path)
    embed_dir = Path(embed_dir)
    classifier, manifest = load_model_1(model_1_dir)
    feature_columns = manifest["feature_columns"]
    embed_cols = manifest.get("embed_cols", list(EMBED_COLS))
    # Prefer the strategy-aware reduction dir from the manifest. Fall back to
    # the legacy `pc1_dir` key (kept by the new exporter so older manifests
    # — and the resolver in `inference_matrix._resolve_artifacts` — still
    # work transparently).
    reduction_dir_str = manifest.get("reduction_dir") or manifest.get("pc1_dir")
    pc1_dir = (
        Path(reduction_dir_str)
        if reduction_dir_str
        else model_1_reduction_dir(model_1_dir)
    )
    strategy_name = manifest.get("reduction_strategy", "pc1")

    with pipeline_step(
        "qa.3a.features",
        f"Build scoring features (frozen reduction, strategy={strategy_name!r})",
        "workflows.qa.inference_features.build_qa_feature_matrix",
        timer=timer,
    ):
        features = build_qa_feature_matrix(
            parquet_path,
            embed_dir,
            feature_columns,
            embed_cols,
            pc1_dir,
            timer=timer,
        )
        X = features.to_numpy()

    with pipeline_step(
        "qa.3b.predict",
        "model_1 predict",
        "classifier.predict / predict_proba",
        timer=timer,
    ):
        pred = classifier.predict(X).astype(np.int8)
        # `predict_proba(X)[:, 1]` is the model's confidence that the row
        # is IS_OWN_RESTAURANT=True (float in [0, 1]). Kept alongside the
        # hard 0/1 label so the model_2 training corpus can use confidence
        # as a feature and we can re-threshold without re-scoring.
        proba = (
            classifier.predict_proba(X)[:, 1].astype(np.float32)
            if hasattr(classifier, "predict_proba")
            else None
        )

    df = pl.read_parquet(parquet_path)
    pred_col = manifest.get("pred_col", MODEL_1_PRED_COL)
    proba_col = manifest.get("proba_col", MODEL_1_PROBA_COL)
    new_cols = [pl.Series(pred_col, pred)]
    if proba is not None:
        new_cols.append(pl.Series(proba_col, proba))
    df = df.with_columns(new_cols)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Ground-truth column '{TARGET_COL}' is missing from {parquet_path}; "
            f"cannot compute '{IF_ERROR_COL}'. Aborting QA workflow."
        )

    # IF_ERROR is TRUE whenever predicted and ground-truth disagree (either
    # direction). A null ground-truth is treated as an anomaly, so IF_ERROR=True
    # there as well.
    pred_bool = pl.col(pred_col).cast(pl.Boolean)
    truth = pl.col(TARGET_COL).cast(pl.Boolean)
    df = df.with_columns(
        pl.when(truth.is_null())
        .then(pl.lit(True))
        .otherwise(pred_bool != truth)
        .alias(IF_ERROR_COL)
    )

    n_errors = int(df.select(pl.col(IF_ERROR_COL).sum()).item())
    print(
        f"[qa/model_1] {IF_ERROR_COL}=True for {n_errors:,}/{df.height:,} rows "
        f"(predicted {pred_col} != {TARGET_COL}, including null truth).",
        flush=True,
    )

    if output_path is None:
        output_path = default_scored_path_for_csv(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pipeline_step(
        "qa.3c.write",
        "Write scored parquet",
        "common.storage.parquet_io.atomic_write_parquet",
        timer=timer,
    ):
        atomic_write_parquet(df, output_path)

    added = [pred_col]
    if proba is not None:
        added.append(proba_col)
    added.append(IF_ERROR_COL)
    print(
        f"[qa/model_1] scored {df.height:,} rows -> {output_path} "
        f"(added {', '.join(added)}).",
        flush=True,
    )

    # ---- Append one row to the QA comparison log -------------------------
    # We compute the QA-time precision metrics opportunistically, only when
    # ground truth is present and non-null (the workflow already errors out
    # earlier if the column is missing, so absence of truth here just means
    # the rows are nulls). This is the same per-row truth check used to set
    # IF_ERROR above; the metrics are computed on the *non-null* slice so a
    # corpus with sparse ground truth still produces usable numbers.
    qa_metrics: dict[str, float | None] = {}
    n_with_truth = int(df.select(pl.col(TARGET_COL).is_not_null().sum()).item())
    if n_with_truth > 0:
        truth_arr = df.get_column(TARGET_COL).cast(pl.Boolean).to_numpy()
        pred_arr = df.get_column(pred_col).cast(pl.Boolean).to_numpy()
        proba_arr = (
            df.get_column(proba_col).cast(pl.Float32).to_numpy()
            if proba_col in df.columns
            else None
        )
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        valid = ~np.isnan(truth_arr.astype(np.float64))
        if valid.any():
            yt = truth_arr[valid].astype(np.int8)
            yp = pred_arr[valid].astype(np.int8)
            qa_metrics = {
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "roc_auc": (
                    float(roc_auc_score(yt, proba_arr[valid]))
                    if proba_arr is not None and len(np.unique(yt)) > 1
                    else None
                ),
            }

    try:
        stage_timings = timings_from_pipeline_timer(timer)
        reduction_strategy_dict = manifest.get(
            "reduction", {"name": manifest.get("reduction_strategy", "pc1")}
        )
        per_col_meta = manifest.get("reduction_per_column", {}) or {}
        n_reduction_features = sum(
            int(meta.get("n_features", meta.get("k", 0)))
            for meta in per_col_meta.values()
        )
        run_id_for_log = manifest.get("run_id") or output_path.stem
        append_qa_row(
            run_id=str(run_id_for_log),
            input_csv=parquet_path.with_suffix(".csv"),
            input_parquet=parquet_path,
            scored_parquet=output_path,
            model_dir=model_1_dir,
            reduction_strategy_dict=reduction_strategy_dict,
            n_rows_scored=int(df.height),
            n_features=len(feature_columns),
            n_reduction_features=int(n_reduction_features),
            per_column_meta=per_col_meta,
            stage_timings=stage_timings,
            metrics=qa_metrics,
            n_rows_with_ground_truth=int(n_with_truth),
            n_if_error_true=int(n_errors),
        )
    except Exception as exc:  # noqa: BLE001
        # Never let logging break a scoring run.
        print(f"[qa/model_1] WARNING: comparison_log append failed: {exc}", flush=True)

    return output_path


def ensure_project_root() -> Path:
    import os

    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT
