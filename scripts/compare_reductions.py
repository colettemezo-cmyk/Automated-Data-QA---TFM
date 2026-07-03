"""Train + optionally score model_1 across every reduction strategy in one run.

For each strategy in `--strategies` (default: `pc1`, `top5`, `adaptive_0.90`,
`adaptive_0.95`, `raw`) the script:

  1. Trains LightGBM + XGBoost against `IS_OWN_RESTAURANT` using the strategy
     to reduce the embedding columns. Each strategy writes its artifacts to a
     separate per-strategy directory under `--comparison-dir` so the
     canonical `data/ml/model_1/` is never clobbered.
  2. Appends one row PER BACKEND to `data/ml/comparison/training_runs.csv`.
     Each row records the strategy, per-column K + cumulative variance,
     fit/predict time, and the standard precision/recall/F1/ROC-AUC.
  3. Optionally runs the QA workflow (`--score-csv <path>`) against the
     freshly-trained model and appends a row to
     `data/ml/comparison/qa_runs.csv` with per-stage timings and (when
     ground truth is present) test metrics.

Tips:
  * `raw` requires substantial RAM (~241 GB at 11M rows × 5,376 features).
    Always pair it with `--executions` for full-corpus comparisons; for the
    typical smoke comparison `--executions 20` keeps everything well under
    16 GB.
  * Use `--strategies pc1,top5,adaptive_0.95` to skip strategies you don't
    want to benchmark this round.

Usage:
    # Smoke comparison (cached embeddings, 80 executions, all strategies):
    python scripts/compare_reductions.py

    # With QA scoring of a fresh CSV after each training:
    python scripts/compare_reductions.py --score-csv data/ODM_Latam_26_mar_FN1.csv

    # Only the heavier strategies, smaller sample:
    python scripts/compare_reductions.py --strategies adaptive_0.95,raw --executions 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from common.comparison_log import TRAINING_LOG, QA_LOG  # noqa: E402
from common.config.columns import EMBED_COLS  # noqa: E402
from common.config.paths import PROJECT_ROOT  # noqa: E402
from common.features.reduction import parse_strategy  # noqa: E402
from common.pipeline_timing import PipelineTimer  # noqa: E402
from workflows.training.classifier import (  # noqa: E402
    train_and_compare_binary_classifiers,
)
from workflows.training.config import (  # noqa: E402
    CSV_PATH,
    EMBED_DIR,
    MODEL_1_RUNS_DIR,
    PARQUET_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

# `run_qa_pipeline` is imported lazily inside `main()` because it pulls in
# `onnxruntime` via the embedding module — we don't want `--help` (or a
# pure-training comparison run) to require the ONNX install.

DEFAULT_STRATEGIES: tuple[str, ...] = (
    "pc1",
    "top5",
    "adaptive_0.90",
    "adaptive_0.95",
    "raw",
)


def _parse_strategies(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


def _utcnow_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _per_strategy_model_dir(comparison_dir: Path, strategy_name: str) -> Path:
    return comparison_dir / f"model_1_{strategy_name}"


def _missing_embedding_sidecars() -> list[str]:
    """Embed columns whose `{col}_EMBEDDING.parquet` is not on disk yet."""
    missing: list[str] = []
    for col in EMBED_COLS:
        sidecar = EMBED_DIR / f"{col}_EMBEDDING.parquet"
        if not sidecar.exists():
            missing.append(col)
    return missing


def _preflight_or_bootstrap(*, bootstrap: bool) -> None:
    """Ensure the training parquet + embedding sidecars exist.

    `compare_reductions.py` re-uses the trained-feature parquet and the
    per-column embedding sidecars produced by stages 1 and 2 of the normal
    training workflow. If either is missing we either bootstrap them (when
    `--bootstrap` is set) or abort with an actionable hint — bubbling up
    Polars' raw `FileNotFoundError` is unfriendly because the trainer fails
    deep inside a stack trace.
    """
    parquet_missing = not PARQUET_PATH.exists()
    embed_missing = _missing_embedding_sidecars()

    if not parquet_missing and not embed_missing:
        return

    if not bootstrap:
        lines = ["[compare_reductions] cannot start — prerequisite artifacts are missing:"]
        if parquet_missing:
            lines.append(
                f"  - parquet : {PARQUET_PATH} (built from {CSV_PATH} by stage 1)"
            )
        if embed_missing:
            lines.append(
                f"  - embeddings : missing sidecars under {EMBED_DIR} for: "
                + ", ".join(embed_missing)
            )
        lines.append("")
        lines.append("Fix it with either:")
        lines.append("  1. python scripts/compare_reductions.py --bootstrap ...   "
                     "(runs stages 1+2 first, then the comparison)")
        lines.append("  2. python app/training_workflow.py --from-scratch         "
                     "(builds everything + trains the default strategy once)")
        raise SystemExit("\n".join(lines))

    print("[compare_reductions] --bootstrap requested; running stage 1+2 first.", flush=True)
    if parquet_missing:
        if not CSV_PATH.exists():
            raise SystemExit(
                f"[compare_reductions] cannot bootstrap parquet: source CSV is missing "
                f"({CSV_PATH}). Place the training corpus there or update "
                "workflows.training.config.CSV_PATH."
            )
        print(f"[compare_reductions] stage 1: building {PARQUET_PATH.name} from CSV ...", flush=True)
        from workflows.training.ingest import build_parquet_from_csv  # local import

        build_parquet_from_csv()
    if embed_missing:
        print(
            f"[compare_reductions] stage 2: embedding {len(embed_missing)} missing column(s): "
            f"{embed_missing}",
            flush=True,
        )
        from workflows.training.embed import embed_training_columns  # local import

        embed_training_columns()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train model_1 across multiple reduction strategies and append "
            "results to data/ml/comparison/*.csv for diffing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help=(
            "Comma-separated list of strategy specs. Each must parse via "
            "`common.features.reduction.parse_strategy`."
        ),
    )
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument(
        "--full",
        action="store_true",
        help="Train each strategy on every execution (slow + memory-heavy).",
    )
    sample.add_argument(
        "--executions",
        type=int,
        default=80,
        metavar="N",
        help="Subsample N executions for each training run. Ignored with --full.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for both LightGBM and XGBoost.",
    )
    parser.add_argument(
        "--score-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional CSV to score with each freshly-trained model. If set, "
            "the QA workflow runs after each training and a row is appended "
            "to data/ml/comparison/qa_runs.csv."
        ),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ml" / "comparison",
        help="Where per-strategy model artifacts are persisted.",
    )
    parser.add_argument(
        "--force-rebuild-input",
        action="store_true",
        help="Force model_input cache rebuild per strategy (otherwise reused if intact).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help=(
            "Log the error to a per-run JSON and keep going. By default the "
            "first failure aborts so an OOM on `raw` doesn't silently skip."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "If the training parquet or per-column embedding sidecars are "
            "missing, run stage 1 (CSV->parquet) and stage 2 (embed columns) "
            "first instead of failing. Safe to set on a fully-bootstrapped "
            "corpus — both stages are no-ops when artifacts are up to date."
        ),
    )
    args = parser.parse_args()

    strategies = _parse_strategies(args.strategies)
    if not strategies:
        parser.error("--strategies must contain at least one entry")

    # Validate every spec up front so the user sees typos before waiting on
    # the first training run.
    for spec in strategies:
        parse_strategy(spec)

    # Make sure stages 1 + 2 outputs exist before we hand off to the trainer,
    # otherwise the deep failure mode is an opaque Polars `FileNotFoundError`.
    _preflight_or_bootstrap(bootstrap=args.bootstrap)

    max_executions = None if args.full else args.executions
    run_stamp = _utcnow_compact()
    comparison_root = Path(args.comparison_dir) / run_stamp
    comparison_root.mkdir(parents=True, exist_ok=True)
    runs_root = comparison_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Defer the QA import: pulls in onnxruntime, which we only need with
    # --score-csv. Keeping it inside this guard lets pure-training runs work
    # on machines without the ONNX install.
    run_qa_pipeline = None
    if args.score_csv is not None:
        from workflows.qa.pipeline import run_qa_pipeline as _run_qa_pipeline

        run_qa_pipeline = _run_qa_pipeline

    print(
        f"\n=== compare_reductions ===\n"
        f"  strategies     : {strategies}\n"
        f"  max_executions : {max_executions}\n"
        f"  comparison_dir : {comparison_root}\n"
        f"  training_log   : {TRAINING_LOG}\n"
        f"  qa_log         : {QA_LOG}\n"
        f"  score_csv      : {args.score_csv}\n",
        flush=True,
    )

    summary_path = comparison_root / "summary.json"
    summary: dict = {
        "run_stamp": run_stamp,
        "strategies": strategies,
        "max_executions": max_executions,
        "n_estimators": args.n_estimators,
        "score_csv": str(args.score_csv) if args.score_csv else None,
        "runs": [],
    }

    for spec in strategies:
        strategy = parse_strategy(spec)
        model_dir = _per_strategy_model_dir(comparison_root, strategy.name)
        run_record: dict = {
            "strategy": strategy.name,
            "model_dir": str(model_dir),
            "training": {"ok": False},
            "qa": None,
        }

        print(f"\n=== [{strategy.name}] training ===", flush=True)
        timer = PipelineTimer(f"compare_reductions/{strategy.name}")
        timer.begin_run([f"strategy={strategy.name!r}"])
        t0 = perf_counter()
        try:
            result = train_and_compare_binary_classifiers(
                parquet_path=PARQUET_PATH,
                max_executions=max_executions,
                n_estimators=args.n_estimators,
                force_rebuild_input=args.force_rebuild_input,
                timer=timer,
                strategy=strategy,
                model_1_dir=model_dir,
                runs_dir=MODEL_1_RUNS_DIR / "comparison" / run_stamp / strategy.name,
                test_size=args.test_size,
                random_state=args.random_state,
            )
            run_record["training"] = {
                "ok": True,
                "elapsed_seconds": round(perf_counter() - t0, 3),
                "best_model": result.get("best_model"),
                "n_features": result.get("n_features"),
                "n_reduction_features": result.get("n_reduction_features"),
                "reduction_strategy": result.get("reduction_strategy"),
                "run_report_dir": result.get("run_report_dir"),
            }
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            run_record["training"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "elapsed_seconds": round(perf_counter() - t0, 3),
            }
            print(f"[compare_reductions] {strategy.name}: TRAIN FAILED\n{tb}", flush=True)
            if not args.skip_on_error:
                summary["runs"].append(run_record)
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                raise
        finally:
            timer.end_run()

        if args.score_csv and run_record["training"].get("ok"):
            assert run_qa_pipeline is not None  # set under `if args.score_csv` above
            print(f"\n=== [{strategy.name}] QA scoring ===", flush=True)
            qa_timer = PipelineTimer(f"compare_reductions/{strategy.name}/qa")
            qa_timer.begin_run([f"QA scoring with model_1={model_dir}"])
            t0 = perf_counter()
            try:
                scored_path = run_qa_pipeline(
                    csv_path=args.score_csv,
                    timer=qa_timer,
                    model_1_dir=model_dir,
                    run_cluster=False,  # keep strategy sweeps lean
                )
                run_record["qa"] = {
                    "ok": True,
                    "elapsed_seconds": round(perf_counter() - t0, 3),
                    "scored_parquet": str(scored_path),
                }
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                run_record["qa"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                    "elapsed_seconds": round(perf_counter() - t0, 3),
                }
                print(
                    f"[compare_reductions] {strategy.name}: QA FAILED\n{tb}",
                    flush=True,
                )
                if not args.skip_on_error:
                    summary["runs"].append(run_record)
                    summary_path.write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
                    raise
            finally:
                qa_timer.end_run()

        summary["runs"].append(run_record)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_summary_table(summary)
    _print_log_tails()


def _print_summary_table(summary: dict) -> None:
    print("\n=== compare_reductions summary ===")
    header = f"{'strategy':<18} {'train_ok':>8} {'train_s':>10} {'qa_ok':>6} {'qa_s':>10}"
    print(header)
    print("-" * len(header))
    for run in summary["runs"]:
        train = run.get("training", {})
        qa = run.get("qa") or {}
        print(
            f"{run['strategy']:<18} "
            f"{'yes' if train.get('ok') else 'NO':>8} "
            f"{train.get('elapsed_seconds', ''):>10} "
            f"{('yes' if qa.get('ok') else 'NO') if qa else '-':>6} "
            f"{qa.get('elapsed_seconds', ''):>10}"
        )


def _print_log_tails() -> None:
    for path, label in [(TRAINING_LOG, "training_runs.csv"), (QA_LOG, "qa_runs.csv")]:
        if not path.exists():
            continue
        print(f"\n=== last rows in {label} ===")
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header = rows[0]
        # Show a compact subset of columns for the tail print.
        compact_cols = [
            "timestamp_utc",
            "reduction_strategy",
            "backend",
            "n_features",
            "fit_seconds",
            "predict_seconds",
            "accuracy",
            "f1",
            "roc_auc",
            "mean_cumulative_variance",
        ]
        if label.startswith("qa"):
            compact_cols = [
                "timestamp_utc",
                "reduction_strategy",
                "n_rows_scored",
                "qa_features_seconds",
                "qa_predict_seconds",
                "qa_total_seconds",
                "accuracy",
                "f1",
            ]
        idxs = [header.index(c) for c in compact_cols if c in header]
        line = " | ".join(header[i] for i in idxs)
        print(line)
        print("-" * len(line))
        for row in rows[-min(10, len(rows) - 1):]:
            if row is header:
                continue
            print(" | ".join(row[i] for i in idxs))


if __name__ == "__main__":
    main()
