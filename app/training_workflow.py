"""Training workflow entry point — CSV -> parquet -> embed -> train -> export model_1.

Run from repo root:

    python app/training_workflow.py                  # smoke train (80 executions, cached)
    python app/training_workflow.py --full           # train on all executions
    python app/training_workflow.py --executions 200 # custom sample size
    python app/training_workflow.py --rebuild        # force rebuild of model_input cache
    python app/training_workflow.py --from-scratch   # also rebuild CSV->parquet + embeddings
    python app/training_workflow.py --plots          # also render diagnostic heatmaps

The default ("smoke" train) reuses the cached `data/ml/cache/<id>/model_input.parquet`
when present, so iteration is fast. Use `--rebuild` or `--from-scratch` to invalidate
the cache or rebuild the pre-training stages respectively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from workflows.training.pipeline import run_training_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training workflow: build model_1 (CSV -> parquet -> embed -> train -> export).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument(
        "--full",
        action="store_true",
        help="Train on every execution in the corpus (slow).",
    )
    sample.add_argument(
        "--executions",
        type=int,
        default=80,
        metavar="N",
        help="Subsample N executions for smoke runs. Ignored with --full.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the model_input cache before training.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Also rebuild CSV->parquet and embeddings before training.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also generate ownership + correlation heatmaps under figures/.",
    )
    model_2 = parser.add_mutually_exclusive_group()
    model_2.add_argument(
        "--train-model-2",
        action="store_true",
        help=(
            "Also train + compare model_2 (LightGBM vs XGBoost) on IF_ERROR "
            "using QA scored parquets. Documented under data/ml/model_2/runs/."
        ),
    )
    model_2.add_argument(
        "--only-model-2",
        action="store_true",
        help=(
            "Skip model_1 training and only run the model_2 stage "
            "(requires an existing model_1 artifact + at least one scored parquet)."
        ),
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default=None,
        metavar="STRATEGY",
        help=(
            "Embedding-reduction strategy. One of: pc1, topN / pcN (e.g. "
            "top5 == pc5), adaptive_0.90, adaptive_0.95, adaptive_0.99, raw. "
            "Default = workflows.training.config.REDUCTION_STRATEGY (top5)."
        ),
    )
    args = parser.parse_args()

    run_training_pipeline(
        run_parquet=args.from_scratch,
        run_embed=args.from_scratch,
        run_plot_ownership=args.plots,
        run_plot_corr=args.plots,
        run_train_models=not args.only_model_2,
        run_train_model_2=args.train_model_2 or args.only_model_2,
        interactive_plots=False,
        max_executions=None if args.full else args.executions,
        force_rebuild_input=args.rebuild,
        reduction_strategy=args.reduction,
    )


if __name__ == "__main__":
    main()
