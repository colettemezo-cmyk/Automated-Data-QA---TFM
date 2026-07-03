"""QA workflow orchestrator.

CSV -> parquet -> embed -> model_1 classify -> model_2 flag errors -> SHAP.

model_1 predicts IS_OWN_RESTAURANT and derives the IF_ERROR target; model_2
then predicts, per row, whether model_1 likely erred (PRED_IF_ERROR). The
optional SHAP stage explains model_2's flags.
"""

from __future__ import annotations

from pathlib import Path

from bootstrap import ensure_app_on_path
from common.pipeline_timing import PipelineTimer, step as pipeline_step
from workflows.qa.config import MODEL_2_DIR, SHAP_SAMPLE_SIZE, resolve_qa_paths
from workflows.qa.embed import embed_cols_present, embed_qa_columns
from workflows.qa.inference_features import assert_embedding_row_counts
from workflows.qa.ingest import build_parquet_from_csv
from workflows.qa.model_1 import score_parquet_with_model_1

ensure_app_on_path(__file__)


def run_qa_pipeline(
    csv_path: Path | None = None,
    *,
    parquet_path: Path | None = None,
    embed_dir: Path | None = None,
    output_parquet: Path | None = None,
    force_parquet: bool = False,
    force_embed: bool = False,
    timer: PipelineTimer | None = None,
    model_1_dir: Path | None = None,
    run_model_2: bool = True,
    model_2_dir: Path | None = None,
    run_shap: bool = False,
    shap_sample_size: int | None = SHAP_SAMPLE_SIZE,
    run_cluster: bool = True,
) -> Path:
    csv_path, parquet_path, embed_dir, output_scored = resolve_qa_paths(
        csv_path,
        parquet_path=parquet_path,
        embed_dir=embed_dir,
        output_scored=output_parquet,
    )

    print(f"[qa] input CSV:     {csv_path}")
    print(f"[qa] parquet:       {parquet_path}")
    print(f"[qa] embeddings:    {embed_dir}")
    print(f"[qa] scored output: {output_scored}")
    if model_1_dir is not None:
        print(f"[qa] model_1 dir:   {model_1_dir}")
    print(f"[qa] model_2:       {'on' if run_model_2 else 'off'}")
    if run_model_2 and model_2_dir is not None:
        print(f"[qa] model_2 dir:   {model_2_dir}")
    print(f"[qa] shap:          {'on' if run_shap else 'off'}")
    print(f"[qa] cluster:       {'on' if run_cluster else 'off'}")

    with pipeline_step(
        "qa.1.parquet",
        "CSV -> Parquet",
        "workflows.qa.ingest.build_parquet_from_csv",
        timer=timer,
    ):
        build_parquet_from_csv(
            csv_path=csv_path,
            parquet_path=parquet_path,
            force=force_parquet,
        )

    with pipeline_step(
        "qa.2.embed",
        "Embed text columns",
        "workflows.qa.embed.embed_qa_columns",
        timer=timer,
    ):
        embed_qa_columns(
            parquet_path=parquet_path,
            embed_dir=embed_dir,
            force=force_embed,
            timer=timer,
        )

    embed_cols = embed_cols_present(parquet_path)
    if embed_cols:
        assert_embedding_row_counts(parquet_path, embed_dir, embed_cols)

    with pipeline_step(
        "qa.3.model_1",
        "Classify with model_1",
        "workflows.qa.model_1.score_parquet_with_model_1",
        timer=timer,
    ):
        kwargs: dict = dict(
            parquet_path=parquet_path,
            embed_dir=embed_dir,
            output_path=output_scored,
            timer=timer,
        )
        if model_1_dir is not None:
            kwargs["model_1_dir"] = Path(model_1_dir)
        scored_path = score_parquet_with_model_1(**kwargs)

    if not run_model_2:
        return scored_path

    # ---- Stage 4: model_2 flags likely model_1 errors (IF_ERROR) ---------
    from workflows.qa.model_2 import score_parquet_with_model_2

    with pipeline_step(
        "qa.4.model_2",
        "Flag likely errors with model_2",
        "workflows.qa.model_2.score_parquet_with_model_2",
        timer=timer,
    ):
        m2_kwargs: dict = dict(
            scored_path=scored_path,
            embed_dir=embed_dir,
            output_path=scored_path,
            timer=timer,
        )
        if model_2_dir is not None:
            m2_kwargs["model_2_dir"] = Path(model_2_dir)
        m2_result = score_parquet_with_model_2(**m2_kwargs)

    # ---- Stage 5: SHAP explanations for model_2 (optional) ---------------
    if run_shap:
        from workflows.qa.shap_eval import run_shap_evaluation

        with pipeline_step(
            "qa.5.shap",
            "SHAP explanations for model_2",
            "workflows.qa.shap_eval.run_shap_evaluation",
            timer=timer,
        ):
            run_shap_evaluation(
                m2_result,
                scored_path,
                sample_size=shap_sample_size,
                timer=timer,
            )

    # ---- Stage 6: unsupervised root-cause grouping of errors (optional) --
    if run_cluster:
        from workflows.qa.cluster import run_error_clustering

        with pipeline_step(
            "qa.6.cluster",
            "Cluster IF_ERROR rows in SHAP space",
            "workflows.qa.cluster.run_error_clustering",
            timer=timer,
        ):
            run_error_clustering(m2_result, scored_path, timer=timer)

    return scored_path
