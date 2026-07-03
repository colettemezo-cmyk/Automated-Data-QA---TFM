"""Export frozen reduction axes to data/ml/model_1/reduction/ (rescue script).

Fixes a QA-workflow break where the classifier is present but the frozen
reduction axes are not (e.g. after deleting the cache by accident). Uses the
same train split as your cached model_input (full data if cache is execall).

Defaults to the `pc1` strategy for backward compatibility with old model_1
classifiers that were trained against scalar PC1 features. To export axes
for a different strategy (must match the one model_1 was trained with), pass
`--reduction <name>`.

Usage:
    python scripts/export_model_1_pc1.py
    python scripts/export_model_1_pc1.py --full
    python scripts/export_model_1_pc1.py --reduction adaptive_0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from common.pipeline_timing import PipelineTimer  # noqa: E402
from workflows.training.config import MODEL_1_DIR, PARQUET_PATH  # noqa: E402
from workflows.training.model_1_export import model_1_pc1_dir  # noqa: E402
from workflows.training.model_input import export_model_1_pc1_axes  # noqa: E402


def _max_executions_from_manifest() -> int | None:
    manifest = MODEL_1_DIR / "manifest.json"
    if not manifest.exists():
        return 80
    data = json.loads(manifest.read_text(encoding="utf-8"))
    mi = data.get("training_model_input")
    if not mi:
        return None
    meta_path = Path(mi).parent / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8")).get("max_executions")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export train-fitted PC1 axes for model_1 (QA workflow prerequisite)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use all executions (match full model_input cache).",
    )
    args = parser.parse_args()
    max_executions = None if args.full else _max_executions_from_manifest()

    timer = PipelineTimer("export_model_1_pc1")
    timer.begin_run(["Export frozen PC1 -> data/ml/model_1/pc1/"])
    try:
        out = export_model_1_pc1_axes(
            parquet_path=PARQUET_PATH,
            pc1_save_dir=model_1_pc1_dir(),
            max_executions=max_executions,
            timer=timer,
        )
        print(f"Done: {out}")
    finally:
        timer.end_run()


if __name__ == "__main__":
    main()
