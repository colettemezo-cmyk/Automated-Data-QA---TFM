"""Build model_input cache (training workflow only).

Usage:
    python scripts/build_model_input.py
    python scripts/build_model_input.py --full --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from workflows.training.model_input import build_model_input_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Training workflow: build model_input cache.")
    parser.add_argument("--full", action="store_true", help="All executions (slow first run).")
    parser.add_argument("--force", action="store_true", help="Ignore existing cache.")
    args = parser.parse_args()
    path, features = build_model_input_dataset(
        max_executions=None if args.full else 80,
        force=args.force,
    )
    print(f"Done: {path}")
    print(f"Features ({len(features)}): {features}")


if __name__ == "__main__":
    main()
