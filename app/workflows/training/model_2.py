"""Stage 5: binary IF_ERROR classification (XGBoost vs LightGBM, model_2).

Trains a binary classifier whose target is the QA workflow's `IF_ERROR` column
(produced by model_1 — `IF_ERROR=True` when model_1's prediction disagrees with
ground truth or ground truth is missing).

Inputs:
  * QA scored parquets (output of `workflows.qa.model_1.score_parquet_with_model_1`),
    discovered by glob (`data/qa_scored/*.scored.parquet`) or passed explicitly.
  * For each scored parquet, the matching embeddings sidecar directory
    (`data/<stem>_embeddings/`) — used to project the frozen model_1 PC1 axes.

Feature matrix:
  Identical to model_1 — tabular numerics + frozen `{col}_EMB_PC1` columns.
  Explicitly *forbidden* from features (asserted at runtime):
    - `IS_OWN_RESTAURANT`           (model_1's target / ground truth)
    - `PRED_IS_OWN_RESTAURANT`      (model_1's hard label)
    - `PROBA_IS_OWN_RESTAURANT`     (model_1's confidence)

Output:
  * `data/ml/model_2/classifier.joblib` + `manifest.json`
  * `data/ml/model_2/runs/<run_id>_model_2/run_report.{json,md}` — full record of
    config, dataset, split, feature list, per-model hyperparameters, fit/predict
    timings, metrics (accuracy/precision/recall/F1/ROC-AUC), classification
    report, confusion matrix, and top feature importances. Reports never
    overwrite (timestamp-prefixed) so we can diff exact values across runs.

Run from repo root:

    python app/training_workflow.py --train-model-2          # train + compare
    python app/training_workflow.py --only-model-2           # skip model_1 stage
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from bootstrap import ensure_app_on_path
from common.config.columns import IF_ERROR_COL, SPLIT_GROUP_COL
from common.config.paths import PROJECT_ROOT
from common.features.inference_matrix import build_inference_feature_matrix
from common.pipeline_timing import step as pipeline_step
from common.training_run_report import TrainingRunReport
from workflows.training.config import (
    MODEL_1_DIR,
    MODEL_2_BACKEND,
    MODEL_2_DIR,
    MODEL_2_FORBIDDEN_FEATURE_COLS,
    MODEL_2_RUNS_DIR,
    MODEL_2_SCORED_GLOB,
    MODEL_2_TARGET_COL,
    MODEL_2_TARGET_RECALL,
    RANDOM_STATE,
    TEST_SIZE,
)
from workflows.training.model_2_export import save_model_2

ensure_app_on_path(__file__)


MODEL_1_MANIFEST_NAME = "manifest.json"
MODEL_1_CLASSIFIER_NAME = "classifier.joblib"
PC1_SUBDIR = "pc1"
TOP_FEATURE_IMPORTANCES = 20


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    fit_seconds: float
    predict_seconds: float


def _load_model_1_manifest(model_1_dir: Path) -> dict:
    manifest_path = Path(model_1_dir) / MODEL_1_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"model_1 manifest not found at {manifest_path}. "
            "Train model_1 first (python app/training_workflow.py)."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _derive_embed_dir(scored_path: Path) -> Path:
    """`data/qa_scored/<stem>.scored.parquet` -> `data/<stem>_embeddings/`."""
    stem = scored_path.name.removesuffix(".parquet").removesuffix(".scored")
    return PROJECT_ROOT / "data" / f"{stem}_embeddings"


def _discover_scored_parquets(
    scored_paths: list[Path] | None,
    glob_pattern: str,
) -> list[Path]:
    if scored_paths:
        resolved = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in scored_paths]
    else:
        resolved = sorted(PROJECT_ROOT.glob(glob_pattern))
    missing = [p for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "model_2 input scored parquets missing:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\nRun the QA workflow first to populate {glob_pattern}."
        )
    if not resolved:
        raise FileNotFoundError(
            f"No scored parquets found under {glob_pattern}. "
            "Run `python app/qa_workflow.py` first to produce QA scored output."
        )
    return resolved


def _assert_no_forbidden_features(feature_columns: list[str]) -> None:
    leaked = [c for c in feature_columns if c in MODEL_2_FORBIDDEN_FEATURE_COLS]
    if leaked:
        raise ValueError(
            "model_2 feature matrix contains forbidden columns "
            f"{leaked}. These come from model_1's output and must never be "
            "used as features for the IF_ERROR classifier."
        )


def _split_execution_ids(
    execution_ids: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(execution_ids)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(unique_ids, groups=unique_ids))
    train_set = set(unique_ids[train_idx])
    test_set = set(unique_ids[test_idx])
    train_mask = np.isin(execution_ids, list(train_set))
    test_mask = np.isin(execution_ids, list(test_set))
    return train_mask, test_mask


def _evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    fit_seconds: float,
    predict_seconds: float,
) -> ModelMetrics:
    roc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, y_prob))
    return ModelMetrics(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )


def _recall_oriented_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> dict:
    """Pick the highest probability threshold meeting a minimum recall.

    The QA goal is to catch every IF_ERROR=True row, so we sweep the
    precision/recall curve and choose the largest threshold whose recall on
    the positive class is still >= ``target_recall``. Picking the *largest*
    such threshold keeps precision as high as possible while honouring the
    recall floor. Falls back to the smallest observed threshold (max recall)
    if the target is unreachable.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds of length n-1, aligned to
    # precision[:-1]/recall[:-1]. Consider only those operating points.
    prec = precision[:-1]
    rec = recall[:-1]
    feasible = np.where(rec >= target_recall)[0]
    if feasible.size > 0:
        # Among points meeting the recall floor, take the one with the
        # highest threshold (== best precision along this curve).
        best_idx = feasible[np.argmax(thresholds[feasible])]
        achieved_target = True
    else:
        # Target unreachable: fall back to the threshold with maximum recall.
        best_idx = int(np.argmax(rec))
        achieved_target = False
    threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= threshold).astype(np.int8)
    return {
        "decision_threshold": threshold,
        "target_recall": float(target_recall),
        "achieved_target_recall": bool(achieved_target),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_at_threshold": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _print_metrics_table(metrics: list[ModelMetrics]) -> None:
    header = (
        f"{'model':<12} {'accuracy':>9} {'precision':>10} {'recall':>8} "
        f"{'f1':>8} {'roc_auc':>8} {'fit_s':>8} {'pred_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        roc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "n/a"
        print(
            f"{m.name:<12} {m.accuracy:>9.4f} {m.precision:>10.4f} "
            f"{m.recall:>8.4f} {m.f1:>8.4f} {roc:>8} "
            f"{m.fit_seconds:>8.2f} {m.predict_seconds:>8.2f}"
        )


def _build_dataset(
    scored_paths: list[Path],
    feature_columns: list[str],
    embed_cols: list[str],
    pc1_dir: Path,
    timer=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Build (X, y, exec_ids, source_records) from all scored parquets."""
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    exec_parts: list[np.ndarray] = []
    source_records: list[dict] = []

    for idx, scored_path in enumerate(scored_paths, start=1):
        embed_dir = _derive_embed_dir(scored_path)
        if not embed_dir.exists():
            raise FileNotFoundError(
                f"Embeddings directory not found for {scored_path}: {embed_dir}. "
                "Re-run the QA workflow so embeddings are regenerated."
            )

        with pipeline_step(
            "4.3.model_2.build_features",
            f"Build features {scored_path.name} ({idx}/{len(scored_paths)})",
            "workflows.training.model_2._build_dataset",
            timer=timer,
        ):
            features = build_inference_feature_matrix(
                scored_path,
                embed_dir,
                feature_columns,
                embed_cols,
                pc1_dir,
                timer=timer,
                pc1_step_id="4.3.model_2.pc1.column",
            )
            X = features.to_numpy()

        with pipeline_step(
            "4.3.model_2.load_target",
            f"Load target+groups {scored_path.name}",
            "workflows.training.model_2._build_dataset",
            timer=timer,
        ):
            meta = pl.read_parquet(
                scored_path, columns=[IF_ERROR_COL, SPLIT_GROUP_COL]
            )
            y = meta.get_column(IF_ERROR_COL).cast(pl.Int8).to_numpy()
            raw_exec = meta.get_column(SPLIT_GROUP_COL).cast(pl.Utf8).to_numpy()

        # Namespacing the EXECUTION_ID by source file keeps group-split safe
        # if model_2 is ever trained on multiple QA scored corpora at once
        # (where two different datasets might legally share an EXECUTION_ID).
        dataset_tag = scored_path.stem
        exec_ids = np.array(
            [f"{dataset_tag}::{e}" for e in raw_exec], dtype=object
        )

        if X.shape[0] != y.shape[0] or X.shape[0] != exec_ids.shape[0]:
            raise ValueError(
                f"Row count mismatch for {scored_path}: "
                f"X={X.shape[0]} y={y.shape[0]} exec_ids={exec_ids.shape[0]}"
            )

        X_parts.append(X)
        y_parts.append(y)
        exec_parts.append(exec_ids)
        source_records.append(
            {
                "scored_parquet": str(scored_path),
                "embed_dir": str(embed_dir),
                "n_rows": int(X.shape[0]),
                "n_if_error_true": int(y.sum()),
                "n_executions": int(np.unique(exec_ids).shape[0]),
            }
        )

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    exec_all = np.concatenate(exec_parts)
    return X_all, y_all, exec_all, source_records


def _top_feature_importances(
    model, feature_columns: list[str], top_n: int = TOP_FEATURE_IMPORTANCES
) -> list[dict]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return []
    arr = np.asarray(raw, dtype=float)
    order = np.argsort(arr)[::-1][:top_n]
    return [
        {"feature": feature_columns[i], "importance": float(arr[i])}
        for i in order
    ]


def _hyperparameters(model) -> dict:
    try:
        params = model.get_params()
    except Exception:  # noqa: BLE001
        return {}
    safe: dict = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = repr(v)
    return safe


def train_and_compare_model_2(
    *,
    scored_paths: list[Path] | None = None,
    model_1_dir: Path = MODEL_1_DIR,
    model_2_dir: Path = MODEL_2_DIR,
    backend: str = MODEL_2_BACKEND,
    runs_dir: Path = MODEL_2_RUNS_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 200,
    timer=None,
) -> dict:
    """Train + compare LightGBM and XGBoost on `IF_ERROR`, document the run."""
    model_1_manifest = _load_model_1_manifest(model_1_dir)
    feature_columns: list[str] = list(model_1_manifest["feature_columns"])
    embed_cols: list[str] = list(model_1_manifest["embed_cols"])
    pc1_dir = Path(
        model_1_manifest.get("pc1_dir", Path(model_1_dir) / PC1_SUBDIR)
    )

    _assert_no_forbidden_features(feature_columns)

    resolved_paths = _discover_scored_parquets(scored_paths, MODEL_2_SCORED_GLOB)

    report = TrainingRunReport(run_name="model_2", runs_dir=runs_dir)
    report.record_config(
        backend_compared=["lightgbm", "xgboost"],
        canonical_backend=backend,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        target_col=MODEL_2_TARGET_COL,
        split_group_col=SPLIT_GROUP_COL,
        model_1_dir=str(model_1_dir),
        model_2_dir=str(model_2_dir),
    )
    report.record_leakage_guard(
        excluded_columns=list(MODEL_2_FORBIDDEN_FEATURE_COLS),
        checked_columns=list(MODEL_2_FORBIDDEN_FEATURE_COLS),
    )
    report.record_feature_columns(feature_columns)

    print(
        f"[model_2] inputs ({len(resolved_paths)} scored parquet(s)):",
        flush=True,
    )
    for p in resolved_paths:
        print(f"  - {p}", flush=True)
    print(
        f"[model_2] features ({len(feature_columns)}): {feature_columns}",
        flush=True,
    )

    X, y, exec_ids, source_records = _build_dataset(
        resolved_paths,
        feature_columns,
        embed_cols,
        pc1_dir,
        timer=timer,
    )

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    report.record_dataset(
        sources=source_records,
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_executions=int(np.unique(exec_ids).shape[0]),
        target_distribution={
            "IF_ERROR=True": n_pos,
            "IF_ERROR=False": n_neg,
            "positive_rate": float(n_pos / max(len(y), 1)),
        },
    )

    train_mask, test_mask = _split_execution_ids(
        exec_ids, test_size=test_size, random_state=random_state
    )
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    train_pos = int(y_train.sum())
    train_neg = int(len(y_train) - train_pos)
    test_pos = int(y_test.sum())
    test_neg = int(len(y_test) - test_pos)
    report.record_split(
        split_strategy="GroupShuffleSplit on EXECUTION_ID",
        test_size=test_size,
        random_state=random_state,
        n_train_rows=int(len(y_train)),
        n_test_rows=int(len(y_test)),
        n_train_executions=int(np.unique(exec_ids[train_mask]).shape[0]),
        n_test_executions=int(np.unique(exec_ids[test_mask]).shape[0]),
        train_target_distribution={
            "IF_ERROR=True": train_pos,
            "IF_ERROR=False": train_neg,
        },
        test_target_distribution={
            "IF_ERROR=True": test_pos,
            "IF_ERROR=False": test_neg,
        },
    )

    print(
        f"[model_2] train {X_train.shape[0]:,} ({train_pos:,} pos / {train_neg:,} neg) | "
        f"test {X_test.shape[0]:,} ({test_pos:,} pos / {test_neg:,} neg)",
        flush=True,
    )

    if train_pos == 0 or test_pos == 0:
        report.add_note(
            "WARNING: at least one split has zero IF_ERROR=True rows; metrics may be degenerate."
        )

    pos = max(train_pos, 1)
    neg = max(train_neg, 1)
    scale_pos_weight = neg / pos

    models = {
        "lightgbm": LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
    }

    metrics_list: list[ModelMetrics] = []
    fitted_models: dict = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        with pipeline_step(
            f"4.3.model_2.train.{name}",
            f"Train model_2 / {name}",
            f"workflows.training.model_2 / {name}",
            timer=timer,
        ):
            t0 = perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = perf_counter() - t0

            t1 = perf_counter()
            y_pred = model.predict(X_test)
            y_prob = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )
            predict_seconds = perf_counter() - t1

            m = _evaluate(name, y_test, y_pred, y_prob, fit_seconds, predict_seconds)
            metrics_list.append(m)
            fitted_models[name] = model
            if y_prob is not None:
                test_probabilities[name] = y_prob

            print(f"\n[model_2] {name} classification report (test):", flush=True)
            print(classification_report(y_test, y_pred, digits=4))

            report.record_model(
                name=name,
                backend=name,
                hyperparameters=_hyperparameters(model),
                metrics={
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "roc_auc": m.roc_auc,
                },
                classification_report=classification_report(
                    y_test, y_pred, digits=4, output_dict=True, zero_division=0
                ),
                confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                n_train_rows=int(len(y_train)),
                n_test_rows=int(len(y_test)),
                feature_importances_top=_top_feature_importances(
                    model, feature_columns
                ),
            )

    _print_metrics_table(metrics_list)

    primary_metric = "f1"
    ranking = sorted(
        (
            {
                "name": m.name,
                "f1": m.f1,
                "roc_auc": m.roc_auc,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
            }
            for m in metrics_list
        ),
        key=lambda r: r[primary_metric],
        reverse=True,
    )
    best = ranking[0]["name"] if ranking else None

    selected_backend: str | None = None
    if backend in fitted_models:
        # Recall-first threshold: catch (nearly) every IF_ERROR=True row.
        # Computed on the held-out test probabilities of the selected backend
        # and persisted so any model_2 scoring flags errors via
        # `PROBA_IF_ERROR >= decision_threshold` instead of the default 0.5.
        threshold_info: dict | None = None
        if backend in test_probabilities:
            threshold_info = _recall_oriented_threshold(
                y_test, test_probabilities[backend], MODEL_2_TARGET_RECALL
            )
            print(
                f"\n[model_2] recall-oriented threshold for {backend}: "
                f"{threshold_info['decision_threshold']:.6f} "
                f"(target recall {threshold_info['target_recall']:.4f}, "
                f"achieved={threshold_info['achieved_target_recall']}) -> "
                f"recall {threshold_info['recall_at_threshold']:.4f}, "
                f"precision {threshold_info['precision_at_threshold']:.4f}, "
                f"f1 {threshold_info['f1_at_threshold']:.4f}",
                flush=True,
            )
            report.add_note(
                "model_2 recall-oriented decision threshold "
                f"{threshold_info['decision_threshold']:.6f} selected for "
                f"target recall >= {threshold_info['target_recall']:.4f} "
                f"(achieved={threshold_info['achieved_target_recall']}): "
                f"recall={threshold_info['recall_at_threshold']:.4f}, "
                f"precision={threshold_info['precision_at_threshold']:.4f}, "
                f"f1={threshold_info['f1_at_threshold']:.4f}. Scoring should "
                "flag IF_ERROR via PROBA_IF_ERROR >= decision_threshold."
            )

        extra = {
            "model_1_dir": str(model_1_dir),
            "model_1_pc1_dir": str(pc1_dir),
            "sources": [r["scored_parquet"] for r in source_records],
            "run_report_dir": str(report.dir),
            "best_model_by_f1": best,
        }
        if threshold_info is not None:
            extra.update(threshold_info)

        save_model_2(
            fitted_models[backend],
            feature_columns,
            embed_cols,
            model_2_dir=model_2_dir,
            backend=backend,
            extra=extra,
        )
        selected_backend = backend
    else:
        report.add_note(
            f"MODEL_2_BACKEND={backend!r} is not one of the compared models; "
            "no canonical model_2 artifact was persisted."
        )
        print(
            f"[model_2] MODEL_2_BACKEND={backend!r} not in {list(models)}; "
            "skipping artifact save.",
            flush=True,
        )

    report.record_comparison(
        primary_metric=primary_metric,
        ranking=ranking,
        best_model=best or "",
        selected_backend=selected_backend,
    )
    report_dir = report.finalize_and_save()

    return {
        "metrics": metrics_list,
        "feature_columns": feature_columns,
        "model_2_dir": str(model_2_dir),
        "run_report_dir": str(report_dir),
        "best_model": best,
        "selected_backend": selected_backend,
    }


if __name__ == "__main__":
    from common.pipeline_timing import PipelineTimer

    t = PipelineTimer("model_2")
    t.begin_run(["5 - model_2: IF_ERROR classifier (LightGBM vs XGBoost)"])
    try:
        train_and_compare_model_2(timer=t)
    finally:
        t.end_run()
