"""QA workflow entry point — score a CSV through model_1 and model_2.

1. Set INPUT_CSV in `app/workflows/qa/config.py`
2. Run from repo root:

       python app/qa_workflow.py

Stages: CSV -> Parquet -> embeddings -> model_1 (IS_OWN_RESTAURANT + IF_ERROR)
-> model_2 (flag likely errors, PRED_IF_ERROR) -> optional SHAP explanations.

Embeddings go to `data/<csv_stem>_embeddings/` automatically (never training's
`data/embeddings/`). Stale or mismatched sidecars are rebuilt without extra flags.

Optional:
  --csv other.csv     one-off file (paths still auto-derived)
  --no-model-2        stop after model_1 (legacy behaviour)
  --shap              compute SHAP explanations for model_2 after scoring
  --no-cluster        skip the error-clustering stage (on by default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from common.pipeline_timing import PipelineTimer  # noqa: E402
from workflows.qa.config import INPUT_CSV, SHAP_SAMPLE_SIZE  # noqa: E402
from workflows.qa.pipeline import run_qa_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QA workflow: ingest CSV, embed, classify with model_1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Input CSV (default: INPUT_CSV in config → {INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Scored parquet path (default: <csv>.scored.parquet)",
    )
    parser.add_argument(
        "--force-parquet",
        action="store_true",
        help="Re-parse CSV even if parquet is up to date.",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Re-embed all columns even if sidecars look current.",
    )
    parser.add_argument(
        "--model-1-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Override which model_1 artifact directory to load. Defaults to "
            "`data/ml/model_1/`. Useful for comparing strategies via "
            "`scripts/compare_reductions.py`, which writes per-strategy "
            "models to separate directories."
        ),
    )
    parser.add_argument(
        "--model-2-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override which model_2 artifact directory to load (default: data/ml/model_2/).",
    )
    parser.add_argument(
        "--no-model-2",
        action="store_true",
        help="Stop after model_1 (skip the model_2 IF_ERROR flagging stage).",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Compute SHAP explanations for model_2 after scoring (whole dataset).",
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=SHAP_SAMPLE_SIZE,
        metavar="N",
        help=(
            "Run SHAP on N random rows instead of the whole dataset (faster, "
            "approximate). Default: whole dataset."
        ),
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help=(
            "Skip the error-clustering stage. By default (model_2 on), the "
            "workflow groups the IF_ERROR=True rows into error archetypes by "
            "clustering their model_2 grouped-SHAP signatures, then cross-tabs "
            "each cluster against the scored file's columns to point at the "
            "likely root-cause stage. Output: "
            "data/ml/model_2/clusters/<stem>/cluster_report.json."
        ),
    )
    args = parser.parse_args()

    run_model_2 = not args.no_model_2
    run_cluster = run_model_2 and not args.no_cluster

    planned = [
        "1 - CSV -> Parquet",
        "2 - Embeddings",
        "3 - model_1 classify",
    ]
    if run_model_2:
        planned.append("4 - model_2 flag errors")
    if run_model_2 and args.shap:
        planned.append("5 - SHAP (model_2)")
    if run_cluster:
        planned.append("6 - Error clustering")

    timer = PipelineTimer("qa")
    timer.begin_run(planned)
    timer.register_planned("qa.1.parquet", "CSV -> Parquet", "workflows.qa.ingest")
    timer.register_planned("qa.2.embed", "Embeddings", "workflows.qa.embed")
    timer.register_planned("qa.3.model_1", "Classify", "workflows.qa.model_1")
    if run_model_2:
        timer.register_planned("qa.4.model_2", "Flag errors", "workflows.qa.model_2")
    if run_model_2 and args.shap:
        timer.register_planned("qa.5.shap", "SHAP", "workflows.qa.shap_eval")
    if run_cluster:
        timer.register_planned("qa.6.cluster", "Error clustering", "workflows.qa.cluster")
    try:
        out = run_qa_pipeline(
            csv_path=args.csv,
            output_parquet=args.output,
            force_parquet=args.force_parquet,
            force_embed=args.force_embed,
            timer=timer,
            model_1_dir=args.model_1_dir,
            run_model_2=run_model_2,
            model_2_dir=args.model_2_dir,
            run_shap=args.shap,
            shap_sample_size=args.shap_sample,
            run_cluster=run_cluster,
        )
        print(f"\nQA workflow finished.\n  Scored parquet: {out}")
    finally:
        timer.end_run()


if __name__ == "__main__":
    main()
