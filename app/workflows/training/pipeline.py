"""Training workflow orchestrator (CSV -> parquet -> embed -> train -> export model_1)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from bootstrap import ensure_app_on_path
from common.config.paths import PROJECT_ROOT
from common.pipeline_timing import PipelineTimer
from workflows.training.classifier import train_and_compare_binary_classifiers
from workflows.training.config import PARQUET_PATH
from workflows.training.ingest import build_parquet_from_csv
from workflows.training.model_2 import train_and_compare_model_2

ensure_app_on_path(__file__)
os.chdir(PROJECT_ROOT)


def run_training_pipeline(
    *,
    run_parquet: bool = False,
    run_embed: bool = False,
    run_plot_ownership: bool = False,
    run_plot_corr: bool = False,
    run_train_models: bool = True,
    run_train_model_2: bool = False,
    interactive_plots: bool = False,
    max_executions: int | None = None,
    force_rebuild_input: bool = False,
    reduction_strategy: str | None = None,
) -> None:
    timer = PipelineTimer("training")
    planned = []
    if run_parquet:
        planned.append("1 - CSV -> Parquet")
    if run_embed:
        planned.append("2 - Embeddings")
    if run_plot_ownership:
        planned.append("3a - Ownership heatmap")
    if run_plot_corr:
        planned.append("3b - Correlation heatmaps")
    if run_train_models:
        planned.append("4 - Model input + classifiers + model_1 export")
    if run_train_model_2:
        planned.append("5 - model_2: IF_ERROR classifier (LightGBM vs XGBoost)")
    timer.begin_run(planned)

    try:
        if run_parquet:
            timer.register_planned("1.parquet", "CSV -> Parquet", "workflows.training.ingest")
            with timer.step("1.parquet", "CSV -> Parquet", "workflows.training.ingest.build_parquet_from_csv"):
                build_parquet_from_csv()
        if run_embed:
            from workflows.training.embed import embed_training_columns

            timer.register_planned("2.embed.total", "Embed columns", "workflows.training.embed")
            with timer.step("2.embed.total", "Embed columns", "workflows.training.embed.embed_training_columns"):
                embed_training_columns(timer=timer)
        if run_plot_ownership:
            from workflows.training.plots import plot_ownership_heatmap

            timer.register_planned("3a.plot_ownership", "Ownership heatmap", "workflows.training.plots")
            with timer.step("3a.plot_ownership", "Ownership heatmap", "workflows.training.plots.plot_ownership_heatmap"):
                plot_ownership_heatmap(close_after_save=not interactive_plots)
        if run_plot_corr:
            from workflows.training.plots import plot_correlation_heatmaps

            timer.register_planned("3b.plot_corr", "Correlation heatmaps", "workflows.training.plots")
            with timer.step("3b.plot_corr", "Correlation heatmaps", "workflows.training.plots.plot_correlation_heatmaps"):
                plot_correlation_heatmaps(close_after_save=not interactive_plots)
        if run_train_models:
            timer.register_planned("4.train", "Train + export model_1", "workflows.training.classifier")
            with timer.step(
                "4.train",
                "Model input + classifiers + model_1",
                "workflows.training.classifier.train_and_compare_binary_classifiers",
            ):
                train_and_compare_binary_classifiers(
                    parquet_path=PARQUET_PATH,
                    max_executions=max_executions,
                    force_rebuild_input=force_rebuild_input,
                    timer=timer,
                    strategy=reduction_strategy,
                )
        if run_train_model_2:
            timer.register_planned(
                "5.model_2",
                "Train + export model_2",
                "workflows.training.model_2",
            )
            with timer.step(
                "5.model_2",
                "model_2: IF_ERROR classifier",
                "workflows.training.model_2.train_and_compare_model_2",
            ):
                train_and_compare_model_2(timer=timer)
        if interactive_plots and (run_plot_ownership or run_plot_corr):
            plt.show()
    finally:
        timer.end_run()
