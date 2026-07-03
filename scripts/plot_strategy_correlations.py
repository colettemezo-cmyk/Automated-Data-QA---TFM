"""Generate correlation heatmaps for every strategy in a `scripts/compare_reductions.py` run.

Two modes:

  * `--mode blocks` (default) — UNIFIED view. Each text column's K
    reduction PCs collapse back into a single row/col via first canonical
    correlation. Result is a 17x17 matrix (1 target + 9 tabular + 7 text)
    regardless of strategy, so heatmaps line up cleanly across `pc1`,
    `top5`, `adaptive_0.90`, `adaptive_0.95`. This is the figure that
    answers "how strongly does FRANCHISE correlate with IS_OWN_RESTAURANT?"
    in ONE cell, exactly like the version_zero heatmaps but using all K
    PCs rather than just PC1.

  * `--mode pcs` — RAW per-PC view. Every reduction feature is its own
    row/col (so adaptive_0.95 has ~150 entries). Useful for inspecting
    PC-level redundancy or which specific PCs carry target signal.

For each `model_1_<strategy>` directory under the chosen comparison run, the
script:

  1. Resolves the cached `model_input.parquet` that was built during that
     strategy's training (cache dir name embeds the strategy).
  2. Joins `EXECUTION_ID -> (PARENT_APP_NAME, APP_GOOGLE_COUNTRY_NAME)`
     from the source parquet to subset by (app, country), and plots one
     heatmap per combo with at least `--min-rows` rows.
  3. Saves under `figures/correlations/<run_stamp>/<strategy>/`.

Usage:
    # Latest run, default unified-block view, all strategies
    python scripts/plot_strategy_correlations.py

    # Same with signed +/- cells on the top row (target column)
    python scripts/plot_strategy_correlations.py --signed

    # Top 5 combos only
    python scripts/plot_strategy_correlations.py --top-combos 5

    # Raw per-PC view (the previous default)
    python scripts/plot_strategy_correlations.py --mode pcs

    # Single strategy, specific run
    python scripts/plot_strategy_correlations.py --strategies adaptive_0.95 --run 20260528T104745Z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from common.config.paths import PROJECT_ROOT  # noqa: E402
from workflows.training.config import (  # noqa: E402
    MIN_ROWS_FOR_CORR,
    ML_CACHE_DIR,
    PARQUET_PATH,
)
from workflows.training.plots import (  # noqa: E402
    plot_strategy_block_correlation_heatmaps,
    plot_strategy_correlation_heatmaps,
)


COMPARISON_DIR = PROJECT_ROOT / "data" / "ml" / "comparison"
FIGURES_DIR = PROJECT_ROOT / "figures" / "correlations"


def _latest_run(comparison_dir: Path) -> str:
    runs = [p for p in comparison_dir.iterdir() if p.is_dir() and p.name[0].isdigit()]
    if not runs:
        raise SystemExit(
            f"[plot_strategy_correlations] no comparison runs found under "
            f"{comparison_dir}. Run scripts/compare_reductions.py first."
        )
    runs.sort(key=lambda p: p.name)
    return runs[-1].name


def _discover_strategies(run_dir: Path) -> list[str]:
    """Strategy names from `model_1_<name>` subdirs of a run dir."""
    found = []
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.startswith("model_1_"):
            found.append(child.name[len("model_1_") :])
    return found


def _cache_dir_for_strategy(strategy: str) -> Path | None:
    """Pick the most-recently-modified cache dir whose name ends in `_red<strategy>`.

    The cache tag includes the source fingerprint + exec/test-size/random-state,
    so multiple cache dirs can coexist per strategy (e.g. after rebuilding the
    source parquet). Newest-wins matches what the trainer just used.
    """
    if not ML_CACHE_DIR.exists():
        return None
    suffix = f"_red{strategy}"
    candidates = [p for p in ML_CACHE_DIR.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-(parent_app, country) correlation heatmaps for every "
            "strategy in a compare_reductions run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Comparison run stamp (default: latest under data/ml/comparison/).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of strategies to plot. Default = every "
            "model_1_<strategy>/ found under the run."
        ),
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=PARQUET_PATH,
        help="Training parquet supplying EXECUTION_ID -> (app, country) lookup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir (default: figures/correlations/<run>/).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=MIN_ROWS_FOR_CORR,
        help="Skip (app, country) combos with fewer than this many rows.",
    )
    parser.add_argument(
        "--top-combos",
        type=int,
        default=None,
        help=(
            "Only plot the N most-populous combos per strategy. Useful when "
            "the corpus has 100+ unique pairs and you only want the heavy ones."
        ),
    )
    parser.add_argument(
        "--annot-threshold",
        type=int,
        default=25,
        help=(
            "(--mode pcs only) Show per-cell numeric annotations only when "
            "feature count is at or below this. adaptive_0.95 has ~146 "
            "features; default 25 keeps pc1/top5 annotated and adaptive_* "
            "as colour blocks. Ignored in `--mode blocks` (block matrices "
            "are always 17x17 and fully annotated)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("blocks", "pcs"),
        default="blocks",
        help=(
            "blocks: collapse each text column's K PCs back to one row/col "
            "via first canonical correlation (apples-to-apples across "
            "strategies). pcs: raw per-PC heatmap (every PC is its own "
            "row/col, like the initial version_zero figure)."
        ),
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help=(
            "(--mode blocks only) Attach the sign of each block's strongest "
            "individual feature-vs-target Pearson r to the top row + left "
            "column, then plot on a diverging [-1, 1] colormap. Default is "
            "unsigned canonical correlation in [0, 1] with a sequential cmap."
        ),
    )
    args = parser.parse_args()

    if not args.source_parquet.exists():
        raise SystemExit(
            f"[plot_strategy_correlations] source parquet not found at "
            f"{args.source_parquet}. Run `python app/training_workflow.py "
            f"--from-scratch` first to materialise it."
        )

    run_stamp = args.run or _latest_run(COMPARISON_DIR)
    run_dir = COMPARISON_DIR / run_stamp
    if not run_dir.is_dir():
        raise SystemExit(
            f"[plot_strategy_correlations] no run dir at {run_dir}. "
            f"Check `data/ml/comparison/` for available stamps."
        )

    available = _discover_strategies(run_dir)
    if not available:
        raise SystemExit(
            f"[plot_strategy_correlations] no model_1_* subdirs under "
            f"{run_dir}. Did the run finish?"
        )

    if args.strategies:
        wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in available]
        if unknown:
            raise SystemExit(
                f"[plot_strategy_correlations] unknown strategies {unknown}. "
                f"Available in this run: {available}"
            )
        strategies = wanted
    else:
        strategies = available

    base_out = Path(args.output_dir) if args.output_dir else FIGURES_DIR / run_stamp
    base_out.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== plot_strategy_correlations ===\n"
        f"  run            : {run_stamp}\n"
        f"  mode           : {args.mode}\n"
        f"  strategies     : {strategies}\n"
        f"  source parquet : {args.source_parquet}\n"
        f"  output         : {base_out}\n"
        f"  min_rows       : {args.min_rows}\n"
        f"  top_combos     : {args.top_combos}\n"
        f"  signed         : {args.signed}\n"
        f"  annot threshold: {args.annot_threshold} (pcs mode only)\n",
        flush=True,
    )

    summary: list[tuple[str, dict]] = []
    for strategy in strategies:
        cache_dir = _cache_dir_for_strategy(strategy)
        if cache_dir is None or not (cache_dir / "model_input.parquet").exists():
            print(
                f"[{strategy}] SKIP - no cache dir found in {ML_CACHE_DIR} "
                f"matching `_red{strategy}`",
                flush=True,
            )
            summary.append((strategy, {"error": "no cache"}))
            continue

        out_dir = base_out / strategy
        print(f"\n[{strategy}] cache    : {cache_dir.name}")
        print(f"[{strategy}] output   : {out_dir}", flush=True)
        try:
            if args.mode == "blocks":
                stats = plot_strategy_block_correlation_heatmaps(
                    cache_dir=cache_dir,
                    source_parquet=args.source_parquet,
                    output_dir=out_dir,
                    min_rows=args.min_rows,
                    top_combos=args.top_combos,
                    signed=args.signed,
                )
            else:
                stats = plot_strategy_correlation_heatmaps(
                    cache_dir=cache_dir,
                    source_parquet=args.source_parquet,
                    output_dir=out_dir,
                    min_rows=args.min_rows,
                    top_combos=args.top_combos,
                    annot_threshold=args.annot_threshold,
                )
            summary.append((strategy, stats))
            print(
                f"[{strategy}] DONE     : seen={stats['combos_seen']}, "
                f"plotted={stats['combos_plotted']}, "
                f"skipped={stats['combos_skipped']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{strategy}] FAILED   : {type(exc).__name__}: {exc}", flush=True)
            summary.append((strategy, {"error": f"{type(exc).__name__}: {exc}"}))

    print("\n=== summary ===")
    header = f"{'strategy':<20} {'seen':>6} {'plotted':>8} {'skipped':>8}"
    print(header)
    print("-" * len(header))
    for strategy, stats in summary:
        if "error" in stats:
            print(f"{strategy:<20} {'-':>6} {'-':>8} {'-':>8}  ({stats['error']})")
        else:
            print(
                f"{strategy:<20} {stats['combos_seen']:>6} "
                f"{stats['combos_plotted']:>8} {stats['combos_skipped']:>8}"
            )
    print(f"\nFigures written to {base_out}")


if __name__ == "__main__":
    main()
