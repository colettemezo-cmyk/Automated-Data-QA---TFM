"""Column lists and ML feature definitions (shared by training and scoring)."""

# NOTE: RESTAURANT_BRAND_NAMES is intentionally NOT embedded. IS_OWN_RESTAURANT
# is essentially a deterministic function of the brand list (does it contain an
# own-brand keyword?), so embedding it made model_1 just re-derive the label and
# blinded the QA stage to the cases we care about (brand list incomplete, but
# context says "own"). We replace it with a cheap BRAND_COUNT tabular feature
# (see common.features.preprocess) so model_1/model_2 must lean on context.
EMBED_COLS = [
    "PARENT_APP_NAME",
    "FRANCHISE",
    "CUISINES",
    "FOOD_CATEGORIES",
    "APP_GOOGLE_COUNTRY_NAME",
    "REGION_NAME",
]

CORR_EXCLUDE_COLS = ["EXECUTION_ID", "PIPELINE_FLAG"]

TARGET_COL = "IS_OWN_RESTAURANT"
SPLIT_GROUP_COL = "EXECUTION_ID"

ML_TABULAR_COLS = [
    "IS_DATAGROUP_SECTION_RESTAURANT",
    "QTD_TOTAL_ITEMS",
    "QTD_SECTION_ITEMS",
    "RESTAURANT_AVG_RATING",
    "RESTAURANT_NB_REVIEWS",
    "IS_ODM",
    "IS_QCA",
    "EXECUTION_ROW_COUNT",
    "START_EXECUTION_DATETIME",
    # Count of brands in RESTAURANT_BRAND_NAMES (derived at ingest). Replaces
    # the brand-name embedding: a cheap, leakage-aware signal of "how many
    # brands were listed" without re-deriving the IS_OWN_RESTAURANT label.
    "BRAND_COUNT",
]

# Raw text / id columns that must never become features. RESTAURANT_BRAND_NAMES
# is listed explicitly now that it has been removed from EMBED_COLS (we keep the
# raw column in the parquet only to derive BRAND_COUNT).
ML_EXCLUDE_COLS = [
    TARGET_COL,
    "EXECUTION_ID",
    "PIPELINE_FLAG",
    "RESTAURANT_NAME",
    "RESTAURANT_BRAND_NAMES",
] + EMBED_COLS

ML_PC1_SUFFIX = "_EMB_PC1"
# Format for the i-th principal component feature column produced by the
# strategy-aware reduction (`common.features.reduction`). 1-indexed so the
# K=1 case yields the legacy `_EMB_PC1` name byte-for-byte — model_1 manifests
# trained with the old PC1-only pipeline keep working.
ML_PC_SUFFIX_FMT = "_EMB_PC{i}"
ML_RAW_SUFFIX_FMT = "_EMB_DIM{i:03d}"

# Column written by the QA workflow (model_1) and consumed as model_2's target.
# Defined in `common` (not in `workflows/qa/`) so the training workflow can
# reference it without importing QA code.
IF_ERROR_COL = "IF_ERROR"
