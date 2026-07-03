# Training Workflow — Replication Guide

> Companion document: **[QA_WORKFLOW.md](QA_WORKFLOW.md)** (scoring/inference).
> Short index + cross-cutting design decisions: **[SYSTEM_REPLICATION_GUIDE.md](SYSTEM_REPLICATION_GUIDE.md)**.
>
> This file documents the **training** workflow end-to-end: how `model_1`
> (`IS_OWN_RESTAURANT`) and `model_2` (`IF_ERROR`) are fitted and exported. The
> shared `app/common/` primitives that BOTH workflows use are described in
> §10 ("Shared building blocks") and explicitly flagged as overlap.

---

## 0. A note on decision attribution

Tags used throughout:

- **[DESIGN]** — a deliberate, documented engineering/ML decision (with its
  "why"). I cannot certify whether it originated with you or me, only that it is
  intentional and why the code says it exists.
- **[TUNABLE]** — a judgement-call knob, a likely candidate to revisit on new data.
- **[HARDCODED]** — a literal baked into source (not a config file).

---

## 1. Where training fits in the system

The training workflow turns a labelled CSV corpus into two fitted classifiers:

- **model_1** predicts `IS_OWN_RESTAURANT` from tabular signals + reduced text
  embeddings. Its frozen artifacts (classifier + reduction axes + manifest) are
  the contract the QA workflow consumes.
- **model_2** predicts `IF_ERROR` — "did model_1 likely get this row wrong" —
  trained on the **output of the QA workflow** (`data/qa_scored/*.scored.parquet`).
  This closes the loop: QA produces the corpus model_2 learns from.

**[DESIGN]** Training code never imports QA code. The only coupling is on-disk
artifacts: `data/ml/model_1/manifest.json` (read by QA) and the scored parquets
QA writes (read by model_2 training).

```
CSV ─ingest→ parquet ─embed→ {col}_EMBEDDING/{col}_EMB_PC1 sidecars
    │
    └─ model_input (fit reduction on TRAIN rows only) ─→ model_input.parquet + frozen reduction/ axes
                                                       │
                                         classifier (LGBM vs XGB on IS_OWN_RESTAURANT)
                                                       │
                                         save_model_1 ─→ model_1/{classifier.joblib, manifest.json, reduction/}

model_2 (separate stage, after ≥1 QA run exists):
  data/qa_scored/*.scored.parquet ─(IF_ERROR target + model_1's frozen features)→ model_2/{classifier, manifest}
```

---

## 2. Entry point & CLI flags — `app/training_workflow.py`

**Run:** `python app/training_workflow.py [flags]`
**Orchestrator:** `workflows.training.pipeline.run_training_pipeline`

| Flag | Effect |
|---|---|
| *(none)* | "Smoke" train on 80 executions, reuse cached `model_input.parquet`. |
| `--full` | Train on **every** execution. |
| `--executions N` | Subsample N executions (default 80). Ignored with `--full`. |
| `--rebuild` | Force rebuild of the model_input cache before training. |
| `--from-scratch` | Also rebuild CSV→parquet (stage 1) and embeddings (stage 2). |
| `--plots` | Also render ownership + correlation heatmaps (stage 3). |
| `--train-model-2` | After model_1, also train+compare model_2 (IF_ERROR). |
| `--only-model-2` | Skip model_1; only run the model_2 stage. |
| `--reduction STRATEGY` | Override the reduction strategy (`pc1`/`top5`/`adaptive_0.95`/`raw`/…). Default = `REDUCTION_STRATEGY` in config (currently **`top5`**). |

`main()` maps these to `run_training_pipeline(...)` kwargs.

---

## 3. Config — `workflows/training/config.py`

| Knob | Value | Notes |
|---|---|---|
| `CSV_PATH` | `data/ODM_Latam_25-oct-dec.csv` | **[HARDCODED]** training corpus path. |
| `PARQUET_PATH` | `CSV_PATH.with_suffix(".parquet")` | derived. |
| `EMBED_DIR` | `data/embeddings` | training-corpus sidecars. |
| `MODEL_1_DIR` | `data/ml/model_1` | canonical model_1 artifact dir. |
| `MODEL_1_BACKEND` | `xgboost` | **[DESIGN]** XGB beat LGBM on both recall (0.99983 vs 0.99930) and precision (0.99997 vs 0.99989) on the top5 feature set → fits the recall-first goal and unifies both stages on XGBoost. |
| `MODEL_2_DIR` | `data/ml/model_2` | |
| `MODEL_2_BACKEND` | `xgboost` | **[DESIGN]** XGB beat LGBM on F1 (0.766 vs 0.586, mostly precision 0.623 vs 0.415) at near-equal recall on the rare `IF_ERROR=True` class (top5 corpus, 6 datasets, ~11.85M rows). |
| `MODEL_2_TARGET_COL` | `IF_ERROR` | |
| `MODEL_2_SCORED_GLOB` | `data/qa_scored/*.scored.parquet` | model_2's training inputs. |
| `MODEL_2_TARGET_RECALL` | `0.999` | **[TUNABLE]/[DESIGN]** recall-first threshold target; a missed error is far costlier than a false alarm. |
| `MODEL_2_FORBIDDEN_FEATURE_COLS` | `IS_OWN_RESTAURANT, PRED_IS_OWN_RESTAURANT, PROBA_IS_OWN_RESTAURANT` | **[DESIGN]** asserted absent before fitting model_2 (circular otherwise). |
| `TEST_SIZE` | `0.2` | |
| `RANDOM_STATE` | `42` | **[HARDCODED]** seed (split, etc.). |
| `ML_ROW_CHUNK` | `50_000` | streaming chunk for projection. |
| `ML_MAX_EXECUTIONS` | `80` | default smoke sample. |
| `REDUCTION_STRATEGY` | `top5` | **[DESIGN]** default per-column reduction = top-5 PCs. |

---

## 4. Orchestrator — `run_training_pipeline(...)`

Builds a `PipelineTimer`, `os.chdir(PROJECT_ROOT)`, then conditionally runs each
stage based on the boolean flags, in order: (1) parquet → (2) embed → (3a/3b)
plots → (4) model_input + classifiers + model_1 export → (5) model_2. Every stage
is wrapped in `timer.step(...)`.

---

## 5. Stage 1 — ingest (`workflows/training/ingest.py`)

`build_parquet_from_csv(csv, parquet, force)`:
- Skips re-parsing if the parquet exists, is newer than the CSV, and is readable
  (else re-parses). On skip, still backfills derived columns.
- Reads the CSV with explicit `schema_overrides` (**[HARDCODED]** dtype map:
  `IS_DATAGROUP_SECTION_RESTAURANT`/`IS_TOPTIER_RESTAURANT`/`IS_OWN_RESTAURANT`→
  Boolean, `QTD_*`/`RESTAURANT_AVG_RATING`→Float64) and parses
  `START_EXECUTION_DATETIME` as a Datetime.
- Adds derived features (`common.features.preprocess.add_derived_features`, §10.2),
  atomically writes the parquet.

> **New data:** update `schema_overrides` and the datetime column to your schema.

**Output:** `data/ODM_Latam_25-oct-dec.parquet`.

---

## 6. Stage 2 — embed (`workflows/training/embed.py`)

`embed_training_columns(timer)` calls the **shared** `common.features.embeddings.
embed_text_columns(PARQUET_PATH, EMBED_DIR, cols=EMBED_COLS, ...)` (§10.6) → two
sidecars per text column under `data/embeddings/`:
`{col}_EMBEDDING.parquet` (768-d float32 per row) and `{col}_EMB_PC1.parquet`
(scalar PC1 per row).

First run on the full corpus is the dominant cost (~20 min/column on CPU);
re-runs skip existing/row-aligned sidecars.

---

## 7. Stage 3 (optional) — plots (`workflows/training/plots.py`)

Diagnostics only — not on the model path. Render with `--plots`.

- `plot_ownership_heatmap` — `IS_OWN_RESTAURANT` ratio per (parent_app × country).
- `plot_correlation_heatmaps` — Pearson correlation per (app, country) subset,
  each text column represented by its PC1 scalar sidecar. **[DESIGN]** all
  correlation math runs inside polars (`_polars_corr_matrix`) — never a pandas
  `to_numpy(float64)` broadcast, which previously OOM'd on the 8.2M-row
  iFood/Brazil subset.
- `plot_strategy_correlation_heatmaps` / `plot_strategy_block_correlation_heatmaps`
  — strategy-aware versions driven off a cached `model_input.parquet`. The
  "block" view collapses each text column's K PCs into one row/col via the
  **first canonical correlation**, so every strategy yields the same 17×17 layout
  for visual side-by-side diffing. Includes numerical hardening (float64, rank-
  deficiency / overfit-cell NaN guards) so degenerate single-value subsets don't
  produce spurious r=1.0 cells.

---

## 8. The model_1 path (stage 4)

### 8.1 Stage 4a — model input (`workflows/training/model_input.py`)

Builds the **leakage-safe** training matrix and **fits + freezes** the reduction
axes. This is the heart of the leakage policy.

- `_execution_masks(parquet, test_size, random_state, max_executions)` →
  `(train_mask, test_mask, row_active)`. **[DESIGN]** subsampling chooses random
  *executions* (not rows) via a seeded RNG; the split is `GroupShuffleSplit` over
  **unique EXECUTION_IDs** so no execution straddles train/test.
- `_cache_dir(...)` → `data/ml/cache/<tag>/` where tag =
  `src<fingerprint>_exec<N|all>_ts<test_size>_rs<seed>_red<strategy>`. **[DESIGN]**
  The cache key includes the source-file fingerprint (path + mtime of parquet and
  every embedding sidecar) **and the strategy name**, so swapping strategy or
  touching inputs never reuses stale features.
- `compute_reduction_columns(...)` — per-column fit + project:
  - Fits the strategy axes on **train rows only** (`_fit_reduction_*` → weighted
    PCA on unique train vectors). **[DESIGN]** Train-only fit is the core
    anti-leakage guarantee.
  - When subsampling, materialises a compact per-column embedding subset
    (`_materialize_embedding_subset`) once, then fits/projects from it (avoids
    re-streaming ~13M rows). With all rows, streams the full sidecar.
  - Projects **all active rows** (train+test) with the train-fitted axes.
  - If `reduction_save_dir` is set, persists frozen axes via
    `save_reduction_artifacts` (§10.5).
  - Returns projected matrices, per-column feature names, per-column meta (k,
    achieved cumulative variance, ratios).
- `build_model_input_dataset(...)` → `(model_input.parquet path, feature_columns)`.
  Loads tabular + meta columns, calls `compute_reduction_columns`, fans the
  per-column (n, k) matrices into named float32 series, writes
  `model_input.parquet` (features + `EXECUTION_ID`, `IS_OWN_RESTAURANT`,
  `is_train`, `is_test`) + `meta.json`. Cache hit short-circuits to the cached
  parquet (re-exporting axes if missing).
- `export_model_1_pc1_axes(...)` — fits+saves *only* the frozen axes (used by the
  rescue script and as a fallback when the cache exists but axes don't).

**Output:** `data/ml/cache/<tag>/model_input.parquet` + `meta.json`, plus the
frozen axes under `data/ml/model_1/reduction/`.

### 8.2 Stage 4b — classifier (`workflows/training/classifier.py`)

`train_and_compare_binary_classifiers(...)`:
1. Resolves the strategy, builds/loads `model_input.parquet`, ensures reduction
   axes exist.
2. Loads the matrix, splits via persisted `is_train`/`is_test` masks.
3. Trains **two** models for comparison:
   - `LGBMClassifier(n_estimators=200, class_weight="balanced", ...)`
   - `XGBClassifier(n_estimators=200, eval_metric="logloss",
     scale_pos_weight=neg/pos, ...)`
   **[DESIGN]** Both handle the class imbalance (LGBM via `class_weight`, XGB via
   `scale_pos_weight`). **[TUNABLE]** `n_estimators=200`.
4. Evaluates accuracy/precision/recall/F1/ROC-AUC on held-out executions, prints a
   table + classification reports, records everything to a `TrainingRunReport`
   (§10.9).
5. **Persists the canonical backend = `MODEL_1_BACKEND` (xgboost)** via
   `save_model_1` — **not** necessarily the per-run F1 winner. **[DESIGN]** the
   ranking's `primary_metric` is `f1` (recorded in the report), but which model we
   *ship* is fixed by config; this decouples "what we ship" from "what won today".
6. Appends one `training_runs.csv` row per backend + per-column rows (§10.10).

### 8.3 Stage 4c — export (`workflows/training/model_1_export.py`)

`save_model_1(classifier, feature_columns, embed_cols, strategy,
per_column_reduction_meta, ...)` writes:
- `data/ml/model_1/classifier.joblib`
- `data/ml/model_1/manifest.json` — `backend, target_col, feature_columns,
  embed_cols, reduction_strategy, reduction (dict), reduction_dir,
  reduction_per_column, pred_col, proba_col`, plus **legacy keys** `pc1_suffix`/
  `pc1_dir` so old loaders still work.
- (the `reduction/` axes were written in stage 4a.)

**The manifest is the contract** QA and model_2 read to rebuild identical features.

---

## 9. The model_2 path (stage 5)

Trains the IF_ERROR classifier on the **QA scored corpus** (model_1's output). Run
with `--train-model-2` (after model_1) or `--only-model-2`.

### 9.1 Training — `workflows/training/model_2.py`

`train_and_compare_model_2(...)`:
- `_discover_scored_parquets` — globs `data/qa_scored/*.scored.parquet` (or an
  explicit list); errors if none.
- `_derive_embed_dir(scored_path)` — maps `data/qa_scored/<stem>.scored.parquet`
  → `data/<stem>_embeddings/`.
- `_assert_no_forbidden_features(feature_columns)` — raises if any of
  `MODEL_2_FORBIDDEN_FEATURE_COLS` leaked in. **[DESIGN]** model_2 uses model_1's
  **exact feature set**, never model_1's predictions.
- `_build_dataset(...)` — for each scored parquet builds the feature matrix via
  the **shared** `build_inference_feature_matrix` (frozen model_1 axes, §10.7),
  reads `IF_ERROR` (target) + `EXECUTION_ID` (group). **[DESIGN]** When training on
  multiple corpora, `EXECUTION_ID` is namespaced `"<stem>::<id>"` so the
  `GroupShuffleSplit` can't leak across datasets that share ids.
- Trains LGBM + XGB (same imbalance handling), evaluates, records the report.
- **Recall-first threshold** — `_recall_oriented_threshold(y_test, y_prob,
  MODEL_2_TARGET_RECALL)` sweeps the precision/recall curve and picks the
  **largest** threshold whose recall ≥ target (best precision under the recall
  floor; falls back to the max-recall threshold if unreachable). Persisted as
  `decision_threshold`.
- Saves the canonical backend via `save_model_2`.

### 9.2 Export — `workflows/training/model_2_export.py`

`save_model_2(...)` writes `data/ml/model_2/classifier.joblib` + `manifest.json`
(`backend, target_col, feature_columns, embed_cols, pred_col=PRED_IF_ERROR,
proba_col=PROBA_IF_ERROR`, plus `extra`: `model_1_dir`, `model_1_pc1_dir`,
`sources`, `decision_threshold`, `best_model_by_f1`, recall-threshold stats).

---

## 10. Shared building blocks (`app/common/`) — **OVERLAP WITH QA**

> **This is the overlap surface.** Every primitive below is used by *both*
> workflows. Training is the canonical home for the full descriptions; the QA doc
> ([QA_WORKFLOW.md §8](QA_WORKFLOW.md)) references this section and only adds
> QA-specific usage notes. The hard rule: training and QA must extract features
> *identically*, which is exactly why these live in `common/` and not in either
> workflow.

### 10.1 Column contract — `common/config/columns.py`  *(shared)*

Single source of truth for "feature / target / never-leak". The main porting
surface for new data.

| Constant | Value (current) | Meaning / why |
|---|---|---|
| `EMBED_COLS` | `PARENT_APP_NAME, FRANCHISE, CUISINES, FOOD_CATEGORIES, APP_GOOGLE_COUNTRY_NAME, REGION_NAME` | Text columns embedded (768-d each). **[DESIGN]** `RESTAURANT_BRAND_NAMES` was deliberately removed — it nearly determines `IS_OWN_RESTAURANT`, so embedding it makes model_1 re-derive the label and blinds QA to the cases we care about. Replaced by tabular `BRAND_COUNT`. |
| `TARGET_COL` | `IS_OWN_RESTAURANT` | model_1's target. |
| `SPLIT_GROUP_COL` | `EXECUTION_ID` | leakage-safe split key. |
| `ML_TABULAR_COLS` | `IS_DATAGROUP_SECTION_RESTAURANT, QTD_TOTAL_ITEMS, QTD_SECTION_ITEMS, RESTAURANT_AVG_RATING, RESTAURANT_NB_REVIEWS, IS_ODM, IS_QCA, EXECUTION_ROW_COUNT, START_EXECUTION_DATETIME, BRAND_COUNT` | non-text features. |
| `ML_EXCLUDE_COLS` | target + `EXECUTION_ID, PIPELINE_FLAG, RESTAURANT_NAME, RESTAURANT_BRAND_NAMES` + `EMBED_COLS` | never features. |
| `ML_PC1_SUFFIX` | `_EMB_PC1` | legacy single-PC feature name (back-compat). |
| `ML_PC_SUFFIX_FMT` | `{col}_EMB_PC{i}` (1-indexed) | PCA-strategy feature names; K=1 collapses to `_EMB_PC1`. |
| `ML_RAW_SUFFIX_FMT` | `{col}_EMB_DIM{i:03d}` | raw-passthrough feature names. |
| `IF_ERROR_COL` | `IF_ERROR` | model_1's derived flag, model_2's target. Defined in `common` so training references it without importing QA. |

### 10.2 Derived features — `common/features/preprocess.py`  *(shared)*

`add_derived_features(df)` adds (when source cols present, derived absent):
`IS_ODM` = `PIPELINE_FLAG=="FSA"` (**[HARDCODED]** literal — change when upstream
renames the tag), `IS_QCA` = `PIPELINE_FLAG=="QCA"`, `EXECUTION_ROW_COUNT` =
`len().over("EXECUTION_ID")`, `BRAND_COUNT` = `count('"')//2` over the JSON-array
brand string (comma/whitespace-robust, 0 for `[]`/null). `derive_extra_features`
backfills only missing derived columns into an existing parquet (idempotent).

### 10.3 Atomic parquet I/O — `common/storage/parquet_io.py`  *(shared)*

`atomic_write_parquet` (tmp + `replace()` — survives killed runs),
`parquet_is_readable` (footer-only probe). **[DESIGN]** package named `storage`
(not `io`) to avoid shadowing stdlib.

### 10.4 Weighted PCA — `common/features/pca.py`  *(shared)*

**[DESIGN]** Embeddings are reduced by **frequency-weighted PCA on unique vectors**
— text values repeat heavily, so we embed each unique string once and weight the
covariance by row counts (cheap + equivalent to full-data covariance).
- `first_pc_axis` → `(axis, mean)`, sign-normalised so axis files are diffable.
- `top_k_pc_axes(..., variance_target | fixed_k, k_max, k_min)` → top-K axes
  (k×dim), clamped to effective rank. Adaptive = smallest K reaching cumulative
  variance ≥ target; fixed = constant K.

### 10.5 Reduction strategies — `common/features/reduction.py`  *(shared)*

The central tunable. `ReductionStrategy` (frozen) + `parse_strategy(spec)`:
- `pc1` → K=1 (legacy name kept).
- `pcN`/`topN` (e.g. `pc5`==`top5`) → fixed K=N.
- `adaptive_v` → adaptive to variance v, `k_max = K_MAX_DEFAULTS[v]`. **[TUNABLE]**
  `{0.90:32, 0.95:64, 0.99:128}`.
- `raw` → 768 features/col. **[DESIGN]** memory-heavy (~241 GB at full corpus) —
  only with `--executions`.

`save_reduction_artifacts` writes `reduction/{col}_axes.npy`/`_mean.npy`/
`_ratios.npy` + `reduction_manifest.json`. `load_reduction_artifacts` resolves the
new `reduction/` layout first, then **legacy `pc1/`** (promoted to a synthetic K=1
`pc1` strategy) — this is what keeps pre-strategy model_1 artifacts loadable.

### 10.6 ONNX embeddings — `common/features/embeddings.py` + `config/embedding.py`  *(shared)*

**Model:** `Snowflake/snowflake-arctic-embed-m-v2.0` via prebuilt **ONNX** weights
through `onnxruntime` — **[DESIGN]** no torch/sentence-transformers (~2× CPU
speedup via int8, small footprint; Snowflake's export is already pooled).
- `OnnxArcticEncoder` auto-selects GPU+fp16 / CPU+int8 / fp32 fallback, pins CPU
  to physical cores, L2-normalises. `single_pass` tokenise is the measured fast
  path (`auto` effectively always picks it).
- `_embed_one_column` dedups → encodes uniques once → fits weighted PC1 → streams
  the two sidecars. `embed_text_columns` loops columns, skipping aligned ones.
- Knobs: `MAX_SEQ_LENGTH=128` **[TUNABLE]**, `ENCODE_BATCH_SIZE_CPU/GPU=32/256`,
  `EMBED_ROW_CHUNK=50_000`, `PASSAGE_PREFIX=""`.

### 10.7 Frozen-projection feature builder — `common/features/inference_matrix.py`  *(shared)*

**The most important shared module.** Builds the feature matrix at QA model_1, QA
model_2, **and model_2 training** from **frozen** reduction axes — never refit on
new data. **[DESIGN]** Living in `common/` is what lets training and QA share
*byte-for-byte identical* feature extraction without importing each other.
- `assert_embedding_row_counts` — embeddings must be row-aligned with the parquet.
- `cast_tabular` — `ML_TABULAR_COLS` → float32 (bool/datetime/numeric).
- `build_inference_feature_matrix(parquet, embed_dir, feature_columns, embed_cols,
  pc1_dir, ...)` → polars DataFrame in exact `feature_columns` order; raises a
  descriptive error (strategy + per-col K) if a column is missing.

> In training, this module is consumed by **model_2 training** (`model_2.py`) to
> rebuild model_1's exact feature set on the scored corpus.

### 10.8 Timing — `common/pipeline_timing.py`  *(shared)*

`PipelineTimer` + `step(...)` log predicted-vs-actual per-stage duration + rolling
ETA. Observability only; no effect on results.

### 10.9 Run reports — `common/training_run_report.py`  *(training-owned, used by model_1 & model_2)*

`TrainingRunReport` writes `<runs_dir>/<run_id>_<run_name>/run_report.{json,md}`
with full config/dataset/split/leakage-guard/features/hyperparameters/timings/
metrics/confusion-matrix/feature-importances/winning-backend. **[DESIGN]**
timestamped `run_id` → reports never overwrite, diffable across iterations.

### 10.10 Comparison logs — `common/comparison_log.py`  *(shared)*

Append-only CSVs under `data/ml/comparison/`: `training_runs.csv` (1 row per
run×backend), `reduction_per_column.csv` (per run×column K + cumulative variance),
and `qa_runs.csv` (written by the QA workflow — see QA doc). **[DESIGN]** stable
column order; new metrics appended only; unknown keys dropped with a one-time warn.

---

## 11. Training artifacts produced

```
data/ml/model_1/
  classifier.joblib
  manifest.json
  reduction/{col}_axes.npy, {col}_mean.npy, {col}_ratios.npy, reduction_manifest.json
  runs/<run_id>_model_1/run_report.{json,md}
data/ml/model_2/
  classifier.joblib
  manifest.json
  runs/<run_id>_model_2/run_report.{json,md}
data/ml/cache/<tag>/model_input.parquet, meta.json, emb_subset/*
data/ml/comparison/training_runs.csv, reduction_per_column.csv
```

---

## 12. Hardcoded / to-revise when porting training to new data

1. `workflows/training/config.CSV_PATH` — training corpus filename.
2. CSV `schema_overrides` + datetime column in `ingest.py`.
3. `common/features/preprocess.py` — `PIPELINE_FLAG=="FSA"`, `BRAND_COUNT` JSON
   format assumption, `EXECUTION_ROW_COUNT` ← `EXECUTION_ID`. *(shared)*
4. `common/config/columns.py` — all column lists. *(shared, main porting surface)*
5. `common/config/embedding.py` — model id, `MAX_SEQ_LENGTH`, batch sizes, ONNX
   file names. *(shared)*
6. `RANDOM_STATE=42`, `TEST_SIZE=0.2`, `n_estimators=200`.
7. `REDUCTION_STRATEGY="top5"`, `MODEL_*_BACKEND="xgboost"`,
   `MODEL_2_TARGET_RECALL=0.999`.
8. `RAW_EMBED_DIM=768` in `reduction.py` — tied to the Arctic v2 model. *(shared)*

---

## 13. Maintenance scripts (training)

- `scripts/build_model_input.py` — rebuild the `model_input.parquet` cache
  (`--full`, `--force`).
- `scripts/export_model_1_pc1.py` — re-export frozen reduction axes when the
  classifier exists but `reduction/` (or legacy `pc1/`) is missing. Defaults to
  `pc1`; `--reduction <name>` must match model_1's training strategy; `--full`
  matches a full-data cache.
- `scripts/report_classifier_pc1_variance.py` — per-column train-only PC1
  explained-variance report (uses the same train-only weighted fit).
- `scripts/compare_reductions.py` — trains model_1 across several reduction
  strategies in one run (default `pc1, top5, adaptive_0.90, adaptive_0.95, raw`),
  each to its own dir under `data/ml/comparison/<stamp>/` (never clobbering the
  canonical model). `--score-csv` runs QA after each train; `--bootstrap` runs
  stages 1+2 first; `--skip-on-error` keeps going past an OOM (e.g. `raw`).
- `scripts/plot_strategy_correlations.py` — renders the strategy correlation
  heatmaps.

---

## 14. Training-relevant design decisions (the "why")

1. **[DESIGN] Group-split on `EXECUTION_ID`** — rows in one crawl share structure;
   split at the execution level to prevent train/test leakage.
2. **[DESIGN] Fit reduction axes on TRAIN rows only, then freeze** — QA and model_2
   reuse the frozen axes, never refit. Guarantees consistency + prevents leakage.
3. **[DESIGN] Frequency-weighted PCA on unique vectors** — cheap + statistically
   equivalent to full-data covariance.
4. **[DESIGN] Configurable reduction strategy** (default `top5`) with the strategy
   baked into the cache key + manifest, so swaps never reuse stale features and
   inference always matches training.
5. **[DESIGN] Don't embed `RESTAURANT_BRAND_NAMES`** — replaced by `BRAND_COUNT`.
6. **[DESIGN] XGBoost canonical for both models**, LightGBM trained alongside for
   comparison and recorded in the report.
7. **[DESIGN] model_2 trains on `IF_ERROR`** using model_1's *exact features*, never
   its predictions (forbidden-feature assertion); recall-first threshold persisted.
8. **[DESIGN] Atomic writes + timestamped reports + append-only CSV logs** for
   crash-safety and diffable iterations.
```
