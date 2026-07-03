"""QA workflow config + path resolution.

Edit `INPUT_CSV` (and optionally `OUTPUT_PARQUET` / `OUTPUT_SCORED` for one-off
overrides), then run:

    python app/qa_workflow.py

Everything else — the parquet path, embedding sidecar directory, and scored
output path — is derived from `INPUT_CSV` automatically.

To verify the derived layout for a given config:

    python -m workflows.qa.config
"""

from __future__ import annotations

import sys
from pathlib import Path

# Self-bootstrap: put `app/` on sys.path so `import bootstrap` works even when
# this file is executed directly by path (`python app/workflows/qa/config.py`).
_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from bootstrap import ensure_app_on_path  # noqa: E402

ensure_app_on_path(__file__)

from common.config.columns import IF_ERROR_COL  # noqa: E402,F401  re-exported
from common.config.embedding import EMBED_ROW_CHUNK  # noqa: E402
from common.config.paths import PROJECT_ROOT  # noqa: E402

# ---------------------------------------------------------------------------
# Edit between QA runs
# ---------------------------------------------------------------------------
INPUT_CSV = PROJECT_ROOT / "data" / "ODM_Latam_26_06_FN1.csv"

# One-off path overrides — leave None to derive from INPUT_CSV.
OUTPUT_PARQUET: Path | None = None
OUTPUT_SCORED: Path | None = None

# ---------------------------------------------------------------------------
# Model + scored-parquet column names (rarely need to change)
# ---------------------------------------------------------------------------
MODEL_1_DIR = PROJECT_ROOT / "data/ml/model_1"
MODEL_1_PRED_COL = "PRED_IS_OWN_RESTAURANT"
MODEL_1_PROBA_COL = "PROBA_IS_OWN_RESTAURANT"
# `IF_ERROR_COL` is the boolean flag set per row to TRUE when model_1's
# PRED_IS_OWN_RESTAURANT disagrees with the imported ground-truth value or
# when ground truth is null. The constant is defined in `common.config.columns`
# so the training workflow (model_2) can reference it without importing QA.

# model_2 (IF_ERROR classifier) lives under the same ML root. The QA workflow
# loads the fitted artifact to *score* new datasets — i.e. flag which rows are
# likely model_1 errors — using the recall-oriented `decision_threshold`
# persisted in the manifest at train time. PRED_IF_ERROR/PROBA_IF_ERROR are the
# default column names (the manifest can override them).
MODEL_2_DIR = PROJECT_ROOT / "data/ml/model_2"
MODEL_2_PRED_COL = "PRED_IF_ERROR"
MODEL_2_PROBA_COL = "PROBA_IF_ERROR"

# Where SHAP explanations for model_2 are written (one subdir per scored file).
SHAP_DIR = PROJECT_ROOT / "data" / "ml" / "model_2" / "shap"
# SHAP runs over the WHOLE scored dataset by default (sample_size=None). Global
# feature importance (mean |SHAP|) and every flagged row's top contributors are
# therefore exact. SHAP values are computed in row-chunks so peak memory stays
# bounded regardless of file size. Set a positive SHAP_SAMPLE_SIZE (or pass
# `--shap-sample N`) only if you want a fast approximate run on N random rows.
SHAP_SAMPLE_SIZE: int | None = None
# Rows per SHAP batch when streaming over the full dataset.
SHAP_CHUNK_SIZE = 200_000
# A beeswarm scatter cannot legibly render millions of points, so the beeswarm
# plot draws at most this many rows. The importance table/bar plot remain exact
# (aggregated over every processed row).
SHAP_BEESWARM_MAX = 50_000

# ---------------------------------------------------------------------------
# Error clustering (unsupervised root-cause grouping)
# ---------------------------------------------------------------------------
# After model_2 + SHAP, the QA workflow can group the rows where IF_ERROR=True
# (model_1 disagreed with the imported label) into error "archetypes". We
# cluster in model_2's *grouped SHAP space* (per-row signed SHAP summed into
# source columns) rather than in raw feature space, so rows are grouped by *why*
# they look like errors, not by the natural data distribution. Each cluster is
# then cross-tabbed against lineage columns (which execution / pipeline / day a
# row came from) to point the QA person at the likely root-cause stage.
CLUSTER_DIR = PROJECT_ROOT / "data" / "ml" / "model_2" / "clusters"
# Fixed number of clusters; leave None to auto-pick the best k in CLUSTER_K_RANGE
# by silhouette score.
CLUSTER_K: int | None = None
CLUSTER_K_RANGE = (2, 8)
# Skip clustering when fewer than this many IF_ERROR=True rows exist (too few to
# form meaningful archetypes).
CLUSTER_MIN_ROWS = 20
# Columns cross-tabbed against each cluster ("which executions / pipelines /
# days / cuisines / ... is this error archetype made of"). Leave None (default)
# to use EVERY column in the scored parquet except CLUSTER_LINEAGE_EXCLUDE, so
# the report profiles each cluster across all available fields. Pin an explicit
# list to restrict it (e.g. just the pipeline-stage identifiers).
CLUSTER_LINEAGE_COLS: list[str] | None = None
# Columns barred from the whole cluster report — never a cross-tab dimension AND
# never a tree driver (even if they are model_2 feature sources, e.g. CUISINES /
# FOOD_CATEGORIES). Holds model_1/model_2 outputs and the target (circular — they
# define the error) plus any feature we don't want to root-cause on.
CLUSTER_LINEAGE_EXCLUDE = [
    "IS_OWN_RESTAURANT",
    "PRED_IS_OWN_RESTAURANT",
    "PROBA_IS_OWN_RESTAURANT",
    "IF_ERROR",
    "PRED_IF_ERROR",
    "PROBA_IF_ERROR",
    "CUISINES",
    "FOOD_CATEGORIES",
    "REGION_NAME",
    "RESTAURANT_AVG_RATING",
    "RESTAURANT_NB_REVIEWS",
]
# Continuous numeric columns with more than this many distinct values are
# bucketed into this many equal-frequency (quantile) bins before cross-tabbing,
# so a numeric field still yields interpretable groups instead of one row each.
CLUSTER_NUMERIC_BINS = 5
# Columns that may be cross-tabbed / shown but must never become a tree driver
# (too high-cardinality / noisy to branch on). RESTAURANT_BRAND_NAMES is a raw
# brand-list string: we branch on its cheap BRAND_COUNT proxy instead and show
# the actual brand lists as a description (see CLUSTER_DESCRIBE_UNDER).
CLUSTER_DRIVER_EXCLUDE = ["RESTAURANT_BRAND_NAMES"]
# {describe_col: driver_col} — under each `driver_col` tree node, attach a
# value->count breakdown of `describe_col` over the rows in that node (how many
# times each distinct brand-list array appears in the grouped category).
CLUSTER_DESCRIBE_UNDER = {"RESTAURANT_BRAND_NAMES": "BRAND_COUNT"}
# A column is a cluster's "main driver" when its total_enrichment (sum of the
# enrichment of its over-represented values, enrichment > 1, that are also
# substantial — see CLUSTER_MIN_GROUP_ROWS) exceeds this threshold. Columns below
# it are reported under "not_important".
CLUSTER_DRIVER_IMPORTANCE_LEVEL = 2
# Each SHAP archetype is subdivided by the value-combination of its main-driver
# columns so every group resolves to ONE value per main driver (e.g. one country
# + one parent_app + one execution). A value/sub-group is only kept when it has
# at least this many error rows (drops near-unique noise like RESTAURANT_NAME and
# bounds the number of groups).
CLUSTER_MIN_GROUP_ROWS = 20
# Safety caps so value-splitting can't explode: split on at most this many
# main-driver columns (the top ones by total_enrichment) ...
CLUSTER_MAX_SPLIT_COLS = 6
# ... and keep at most this many final groups overall (largest first).
CLUSTER_MAX_GROUPS = 200
# Per cluster, keep at most this many distinct values per lineage cross-tab
# (bounds output size for high-cardinality columns like EXECUTION_ID).
CLUSTER_CROSSTAB_TOP_N = 20

ROW_CHUNK = EMBED_ROW_CHUNK

# ---------------------------------------------------------------------------
# Derived path locations
# ---------------------------------------------------------------------------
# Training corpus embeddings live here; the QA workflow MUST NOT reuse this
# directory (would risk overwriting training-time sidecars with QA-input
# sidecars of different row counts).
TRAINING_EMBED_DIR = PROJECT_ROOT / "data" / "embeddings"

# Scored QA parquets accumulate here. Each file pairs the original rows with
# PRED_IS_OWN_RESTAURANT, PROBA_IS_OWN_RESTAURANT and IF_ERROR. The directory
# as a whole will become the training corpus for the second model (model_2).
QA_SCORED_DIR = PROJECT_ROOT / "data" / "qa_scored"


# ---------------------------------------------------------------------------
# Path-resolution helpers
# ---------------------------------------------------------------------------
def _abs_csv(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path if csv_path.is_absolute() else PROJECT_ROOT / csv_path


def default_embed_dir_for_csv(csv_path: Path) -> Path:
    """Dataset-specific embeddings: `data/<stem>_embeddings/`."""
    csv_path = _abs_csv(csv_path)
    return csv_path.parent / f"{csv_path.stem}_embeddings"


def default_scored_path_for_csv(csv_path: Path) -> Path:
    """Default scored-parquet path: `data/qa_scored/<stem>.scored.parquet`."""
    csv_path = _abs_csv(csv_path)
    return QA_SCORED_DIR / f"{csv_path.stem}.scored.parquet"


def resolve_qa_paths(
    csv_path: Path | None = None,
    *,
    parquet_path: Path | None = None,
    embed_dir: Path | None = None,
    output_scored: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """All paths derived from the CSV unless explicitly overridden on the CLI.

    The `OUTPUT_PARQUET` / `OUTPUT_SCORED` overrides at the top of this module
    apply only when the configured `INPUT_CSV` is the one being run. Ad-hoc
    `--csv` runs always use derived paths.
    """
    csv_path = _abs_csv(csv_path or INPUT_CSV)
    is_config_csv = csv_path.resolve() == _abs_csv(INPUT_CSV).resolve()

    if parquet_path is None:
        parquet_path = (
            OUTPUT_PARQUET
            if is_config_csv and OUTPUT_PARQUET
            else csv_path.with_suffix(".parquet")
        )
    if embed_dir is None:
        embed_dir = default_embed_dir_for_csv(csv_path)
    embed_dir = Path(embed_dir)
    if embed_dir.resolve() == TRAINING_EMBED_DIR.resolve():
        embed_dir = default_embed_dir_for_csv(csv_path)
    if output_scored is None:
        output_scored = (
            OUTPUT_SCORED
            if is_config_csv and OUTPUT_SCORED
            else default_scored_path_for_csv(csv_path)
        )
    return csv_path, Path(parquet_path), embed_dir, Path(output_scored)


if __name__ == "__main__":
    print("QA workflow paths (from INPUT_CSV):\n")
    csv_path, parquet_path, embed_dir, output_scored = resolve_qa_paths()
    for key, path in [
        ("INPUT_CSV", csv_path),
        ("OUTPUT_PARQUET", parquet_path),
        ("EMBED_DIR", embed_dir),
        ("OUTPUT_SCORED", output_scored),
        ("MODEL_1_DIR", MODEL_1_DIR),
    ]:
        print(f"  {key}: {path}")
