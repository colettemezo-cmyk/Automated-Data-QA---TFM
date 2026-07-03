"""Training workflow paths and knobs."""

from common.config.columns import (
    CORR_EXCLUDE_COLS,
    EMBED_COLS,
    ML_EXCLUDE_COLS,
    ML_PC1_SUFFIX,
    ML_TABULAR_COLS,
    SPLIT_GROUP_COL,
    TARGET_COL,
)
from common.config.embedding import (
    EMBED_ROW_CHUNK,
    ENCODE_BATCH_SIZE_CPU,
    ENCODE_BATCH_SIZE_GPU,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    ONNX_FILE_CPU_INT8,
    ONNX_FILE_FALLBACK,
    ONNX_FILE_GPU_FP16,
    PASSAGE_PREFIX,
)
from common.config.paths import PROJECT_ROOT

CSV_PATH = PROJECT_ROOT / "data/ODM_Latam_25-oct-dec.csv"
PARQUET_PATH = CSV_PATH.with_suffix(".parquet")
EMBED_DIR = PROJECT_ROOT / "data/embeddings"

ML_DIR = PROJECT_ROOT / "data/ml"
ML_CACHE_DIR = ML_DIR / "cache"
MODEL_INPUT_PATH = ML_DIR / "model_input.parquet"

MODEL_1_DIR = ML_DIR / "model_1"
# XGBoost selected as canonical model_1 backend: on the top5 feature set it
# dominated LightGBM on both recall (0.99983 vs 0.99930) and precision
# (0.99997 vs 0.99989), which suits the recall-first goal (catch every
# IS_OWN_RESTAURANT=True) and unifies both stages on XGBoost.
MODEL_1_BACKEND = "xgboost"
MODEL_1_RUNS_DIR = MODEL_1_DIR / "runs"

# Stage 5: model_2 trains on the QA scored corpus (output of model_1) and
# predicts IF_ERROR. Features must be identical to model_1's feature set —
# tabular + frozen PC1 — and MUST NOT include IS_OWN_RESTAURANT,
# PRED_IS_OWN_RESTAURANT or PROBA_IS_OWN_RESTAURANT.
MODEL_2_DIR = ML_DIR / "model_2"
# XGBoost was selected as the canonical model_2 backend after a head-to-head
# comparison on the top5 QA-scored corpus (6 datasets, ~11.85M rows, scored by
# the XGBoost model_1): XGBoost beat LightGBM on F1 (0.766 vs 0.586, mostly
# precision: 0.623 vs 0.415) at near-equal recall on the rare IF_ERROR=True
# class, plus higher accuracy and ROC-AUC.
MODEL_2_BACKEND = "xgboost"
MODEL_2_RUNS_DIR = MODEL_2_DIR / "runs"
MODEL_2_TARGET_COL = "IF_ERROR"
MODEL_2_SCORED_GLOB = "data/qa_scored/*.scored.parquet"
# Recall-first decision threshold for model_2. The QA goal is to catch ALL
# minorities (IF_ERROR=True): a missed error is far more costly than a false
# alarm. After fitting, we pick the highest probability threshold whose recall
# on the held-out positive class is >= MODEL_2_TARGET_RECALL and persist it in
# the manifest as `decision_threshold`. Any model_2 scoring then flags
# IF_ERROR via `PROBA_IF_ERROR >= decision_threshold` instead of the default
# 0.5 argmax, trading some precision for near-complete error capture.
MODEL_2_TARGET_RECALL = 0.999
# Columns that are explicitly forbidden in the model_2 feature matrix; the
# trainer asserts they are absent before fitting.
MODEL_2_FORBIDDEN_FEATURE_COLS = (
    "IS_OWN_RESTAURANT",
    "PRED_IS_OWN_RESTAURANT",
    "PROBA_IS_OWN_RESTAURANT",
)

MIN_ROWS_FOR_CORR = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
ML_ROW_CHUNK = 50_000
ML_MAX_EXECUTIONS = 80

# Default embedding-reduction strategy used by stage 4 (model_input fit) and
# carried through into the model_1 manifest for QA / model_2 to consume.
# Recognised values (resolved via `common.features.reduction.parse_strategy`):
#   "pc1"                  — scalar PC1 per column (legacy default).
#   "top5"/"pc5", ...      — fixed-K top principal components per column
#                            ("pcN" is an alias for "topN").
#   "adaptive_0.90"        — adaptive K per column until cumulative variance
#                            >= 0.90, capped at K_MAX_DEFAULTS[0.90] = 32.
#   "adaptive_0.95"        — same, target 0.95, k_max=64 by default.
#   "adaptive_0.99"        — same, target 0.99, k_max=128 by default.
#   "raw"                  — no projection (768 features per column).
#
# Override per-run via `python app/training_workflow.py --reduction <name>`.
# Default is now the top-5 principal components per embed column ("pc5"), which
# canonicalises to "top5" in manifests / caches / run reports.
REDUCTION_STRATEGY = "top5"
