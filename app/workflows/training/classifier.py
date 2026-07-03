"""Stage 4: binary IS_OWN_RESTAURANT classification (XGBoost vs LightGBM).

Features come only from `build_model_input_dataset()` — tabular numerics plus
train-fitted PC1 columns. Raw text / embed-source columns are never passed in.

Run from repo root:

    python app/training_workflow.py            # smoke train (cached)
    python app/training_workflow.py --full     # train on all executions
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from bootstrap import ensure_app_on_path
from common.comparison_log import (
    append_per_column_rows,
    append_training_row,
)
from common.config.columns import TARGET_COL
from common.features.reduction import ReductionStrategy, parse_strategy
from common.pipeline_timing import step as pipeline_step
from common.training_run_report import TrainingRunReport
from workflows.training.config import (
    ML_MAX_EXECUTIONS,
    MODEL_1_BACKEND,
    MODEL_1_DIR,
    MODEL_1_RUNS_DIR,
    PARQUET_PATH,
    RANDOM_STATE,
    REDUCTION_STRATEGY,
    TEST_SIZE,
)
from workflows.training.model_1_export import (
    model_1_reduction_dir,
    save_model_1,
)
from workflows.training.model_input import (
    META_COLUMNS,
    _embed_cols_present,
    _reduction_axis_files_complete,
    build_model_input_dataset,
    ensure_project_root,
    export_model_1_pc1_axes,
)

TOP_FEATURE_IMPORTANCES = 20

ensure_app_on_path(__file__)


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0


def split_execution_ids(
    execution_ids: np.ndarray,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train, test) aligned to `execution_ids` rows."""
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
    fit_seconds: float = 0.0,
    predict_seconds: float = 0.0,
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


def train_and_compare_binary_classifiers(
    parquet_path: Path = PARQUET_PATH,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    max_executions: int | None = ML_MAX_EXECUTIONS,
    n_estimators: int = 200,
    force_rebuild_input: bool = False,
    timer=None,
    strategy: ReductionStrategy | str | None = None,
    model_1_dir: Path = MODEL_1_DIR,
    runs_dir: Path = MODEL_1_RUNS_DIR,
) -> dict:
    """Build/load model input, then train XGBoost and LightGBM on held-out executions.

    `strategy` controls the embedding-reduction step. Defaults to
    `REDUCTION_STRATEGY` from the training config when not supplied.
    `model_1_dir` lets the comparison harness write per-strategy artifacts
    into separate directories without clobbering the canonical model_1.
    """
    ensure_project_root()

    if isinstance(strategy, str):
        strategy_obj = parse_strategy(strategy)
    elif isinstance(strategy, ReductionStrategy):
        strategy_obj = strategy
    else:
        strategy_obj = parse_strategy(REDUCTION_STRATEGY)

    embed_cols = _embed_cols_present(parquet_path)
    reduction_save = model_1_reduction_dir(model_1_dir)

    report = TrainingRunReport(run_name="model_1", runs_dir=runs_dir)
    report.record_config(
        backend_compared=["lightgbm", "xgboost"],
        canonical_backend=MODEL_1_BACKEND,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        max_executions=max_executions,
        target_col=TARGET_COL,
        force_rebuild_input=force_rebuild_input,
        reduction_strategy=strategy_obj.name,
        reduction_mode=strategy_obj.mode,
        reduction_variance_target=strategy_obj.variance_target,
        reduction_k_max=strategy_obj.k_max,
        reduction_fixed_k=strategy_obj.fixed_k,
        model_1_dir=str(model_1_dir),
    )

    model_path, feature_columns = build_model_input_dataset(
        parquet_path=parquet_path,
        test_size=test_size,
        random_state=random_state,
        max_executions=max_executions,
        force=force_rebuild_input,
        reduction_save_dir=reduction_save,
        timer=timer,
        strategy=strategy_obj,
    )
    if not _reduction_axis_files_complete(reduction_save, embed_cols, strategy_obj):
        export_model_1_pc1_axes(
            parquet_path=parquet_path,
            max_executions=max_executions,
            test_size=test_size,
            random_state=random_state,
            reduction_save_dir=reduction_save,
            timer=timer,
            strategy=strategy_obj,
        )

    with pipeline_step(
        "4.2.load_matrix",
        "Load model_input into memory",
        "polars.read_parquet -> numpy",
        timer=timer,
    ):
        df = pl.read_parquet(model_path)
        X = df.select(feature_columns).to_numpy()
        y = df.get_column(TARGET_COL).cast(pl.Int8).to_numpy()
        is_train = df.get_column("is_train").to_numpy()
        is_test = df.get_column("is_test").to_numpy()

    X_train, X_test = X[is_train], X[is_test]
    y_train, y_test = y[is_train], y[is_test]

    print(
        f"[binary_classifier] train {X_train.shape[0]:,} / test {X_test.shape[0]:,} rows | "
        f"{len(feature_columns)} features (no raw embed columns)",
        flush=True,
    )
    print(f"[binary_classifier] feature columns: {feature_columns}", flush=True)

    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - pos), 1)
    scale_pos_weight = neg / pos

    train_pos = int(y_train.sum())
    test_pos = int(y_test.sum())
    report.record_feature_columns(feature_columns)
    report.record_dataset(
        source_parquet=str(parquet_path),
        model_input_parquet=str(model_path),
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_embed_cols=len(embed_cols),
        target_distribution={
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "positive_rate": float(y.sum() / max(len(y), 1)),
        },
    )
    report.record_split(
        split_strategy="GroupShuffleSplit on EXECUTION_ID (via model_input_dataset)",
        test_size=test_size,
        random_state=random_state,
        n_train_rows=int(len(y_train)),
        n_test_rows=int(len(y_test)),
        train_target_distribution={
            "positive": train_pos,
            "negative": int(len(y_train) - train_pos),
        },
        test_target_distribution={
            "positive": test_pos,
            "negative": int(len(y_test) - test_pos),
        },
    )

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

    results: dict = {
        "metrics": [],
        "feature_names": feature_columns,
        "model_input_path": str(model_path),
        "meta_columns": META_COLUMNS,
    }
    metrics_list: list[ModelMetrics] = []

    for name, model in models.items():
        step_id = f"4.2.train.{name}"
        with pipeline_step(
            step_id,
            f"Train {name}",
            f"workflows.training.classifier / {name}",
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
            print(f"\n{name} classification report (test):", flush=True)
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
    best = ranking[0]["name"] if ranking else ""

    per_col_reduction_meta: dict = {}
    reduction_manifest_path = reduction_save / "reduction_manifest.json"
    if reduction_manifest_path.exists():
        try:
            payload = json.loads(reduction_manifest_path.read_text(encoding="utf-8"))
            per_col_reduction_meta = payload.get("per_column", {})
        except (json.JSONDecodeError, OSError):
            pass

    selected_backend: str | None = None
    if MODEL_1_BACKEND in models:
        save_model_1(
            models[MODEL_1_BACKEND],
            feature_columns,
            embed_cols,
            model_1_dir=model_1_dir,
            backend=MODEL_1_BACKEND,
            extra={"training_model_input": str(model_path)},
            strategy=strategy_obj,
            per_column_reduction_meta=per_col_reduction_meta,
        )
        selected_backend = MODEL_1_BACKEND
    else:
        report.add_note(
            f"MODEL_1_BACKEND={MODEL_1_BACKEND!r} not trained; "
            "model_1 artifact not saved."
        )
        print(
            f"[binary_classifier] MODEL_1_BACKEND={MODEL_1_BACKEND!r} not trained; "
            "model_1 artifact not saved.",
            flush=True,
        )

    report.record_comparison(
        primary_metric=primary_metric,
        ranking=ranking,
        best_model=best,
        selected_backend=selected_backend,
    )
    report_dir = report.finalize_and_save()

    # ---- Append one CSV row per backend to the comparison log ------------
    # `data/ml/comparison/training_runs.csv` accumulates across runs so we
    # can diff strategies side-by-side without re-parsing every JSON report.
    strategy_dict = strategy_obj.to_manifest_dict()
    n_reduction_features = sum(
        int(meta.get("n_features", meta.get("k", 0)))
        for meta in per_col_reduction_meta.values()
    ) or max(len(feature_columns) - X.shape[1] + X.shape[1], 0)
    # Recompute reduction-feature count from feature_columns since
    # per_col_reduction_meta only covers embed cols (tabular features come
    # from a separate list).
    tabular_feature_count = X.shape[1] - n_reduction_features
    if tabular_feature_count < 0:
        n_reduction_features = X.shape[1]
        tabular_feature_count = 0
    for m in metrics_list:
        append_training_row(
            run_id=report.run_id,
            run_name="model_1",
            reduction_strategy_dict=strategy_dict,
            backend=m.name,
            metrics={
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "roc_auc": m.roc_auc,
            },
            per_column_meta=per_col_reduction_meta,
            fit_seconds=m.fit_seconds,
            predict_seconds=m.predict_seconds,
            n_estimators=n_estimators,
            n_train_rows=int(len(y_train)),
            n_test_rows=int(len(y_test)),
            n_features=int(X.shape[1]),
            n_reduction_features=int(n_reduction_features),
            n_rows=int(X.shape[0]),
            max_executions=max_executions,
            model_dir=model_1_dir,
            model_input_parquet=model_path,
            run_report_dir=report_dir,
            reduction_dir=reduction_save,
            selected_as_canonical=(
                m.name == selected_backend if selected_backend else None
            ),
        )
    append_per_column_rows(
        run_id=report.run_id,
        run_name="model_1",
        reduction_strategy_dict=strategy_dict,
        per_column_meta=per_col_reduction_meta,
    )

    results["metrics"] = metrics_list
    results["model_1_dir"] = str(model_1_dir)
    results["run_report_dir"] = str(report_dir)
    results["best_model"] = best
    results["reduction_strategy"] = strategy_obj.name
    results["reduction"] = strategy_obj.to_manifest_dict()
    results["per_column_reduction"] = per_col_reduction_meta
    results["n_features"] = len(feature_columns)
    results["n_reduction_features"] = int(n_reduction_features)
    return results


# Backward-compatible alias (deprecated).
train_and_compare_classifiers = train_and_compare_binary_classifiers


if __name__ == "__main__":
    from common.pipeline_timing import PipelineTimer

    ensure_project_root()
    t = PipelineTimer("binary_classifier")
    t.begin_run(["4 - model input + XGBoost/LightGBM"])
    try:
        train_and_compare_binary_classifiers(max_executions=ML_MAX_EXECUTIONS, timer=t)
    finally:
        t.end_run()
