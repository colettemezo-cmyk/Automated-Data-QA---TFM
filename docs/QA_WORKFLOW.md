# QA Workflow — Replication Guide

> Companion document: **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** (model fitting).
> Short index + cross-cutting design decisions: **[SYSTEM_REPLICATION_GUIDE.md](SYSTEM_REPLICATION_GUIDE.md)**.
>
> This file documents the **QA / scoring** workflow end-to-end: how a new CSV is
> classified by `model_1`, flagged by `model_2`, explained with SHAP, and grouped
> into root-cause clusters. The shared `app/common/` primitives that BOTH
> workflows use are summarised in §8 ("Shared building blocks") and explicitly
> flagged as overlap, with deep detail cross-referenced to the training doc.

---

## 0. A note on decision attribution

Tags: **[DESIGN]** (deliberate, documented choice — origin
unattributable), **[TUNABLE]** (judgement-call knob), **[HARDCODED]** (literal in
source).

---

## 1. Where QA fits in the system

The QA workflow scores a *new* CSV with the artifacts the training workflow froze:

```
CSV ─ingest→ parquet ─embed→ sidecars
    │
    └─ model_1 (frozen axes from manifest) ─→ PRED/PROBA_IS_OWN_RESTAURANT ─→ IF_ERROR ─→ scored.parquet
                                                          │
                                            model_2 (same frozen features) ─→ PROBA_IF_ERROR ≥ threshold ─→ PRED_IF_ERROR
                                                          │
                                            SHAP (grouped) + clustering (SHAP-space) ─→ cluster_report.json
```

**[DESIGN]** QA never imports training code. It depends only on the on-disk
**contract**: `data/ml/model_1/manifest.json` + frozen `reduction/` axes (for
model_1) and `data/ml/model_2/manifest.json` (for model_2). The scored parquets QA
writes to `data/qa_scored/` become the **training corpus for model_2** — closing
the loop (see [TRAINING_WORKFLOW.md §9](TRAINING_WORKFLOW.md)).

---

## 2. Entry point & CLI flags — `app/qa_workflow.py`

**Run:** `python app/qa_workflow.py [flags]`
**Orchestrator:** `workflows.qa.pipeline.run_qa_pipeline`

| Flag | Effect |
|---|---|
| *(none)* | Score the CSV configured as `INPUT_CSV`; run model_1 → model_2 → clustering. |
| `--csv PATH` | One-off input CSV (all paths still auto-derived). |
| `--output PATH` | Override scored-parquet path. |
| `--force-parquet` | Re-parse CSV even if parquet is current. |
| `--force-embed` | Re-embed all columns. |
| `--model-1-dir DIR` / `--model-2-dir DIR` | Load models from alternate dirs (strategy comparison). |
| `--no-model-2` | Stop after model_1 (legacy behaviour). |
| `--shap` | Compute SHAP explanations for model_2 (stage 5). |
| `--shap-sample N` | SHAP on N random rows instead of the whole dataset. |
| `--no-cluster` | Skip the error-clustering stage (on by default). |

`main()` builds a `PipelineTimer`, registers the planned stages, and calls
`run_qa_pipeline(...)`.

---

## 3. Config & path derivation — `workflows/qa/config.py`

- **`INPUT_CSV`** (currently `data/ODM_Latam_26_05_FN2.csv`) — **edit between runs**
  (or pass `--csv`).
- `resolve_qa_paths(...)` derives everything from the CSV:
  - parquet = `<csv>.parquet`
  - embeddings = `data/<stem>_embeddings/` — **[DESIGN]** never the training
    `data/embeddings/` (guarded explicitly so QA-input sidecars can't overwrite
    training sidecars of a different row count).
  - scored output = `data/qa_scored/<stem>.scored.parquet`.
- Model dirs (`MODEL_1_DIR`, `MODEL_2_DIR`), prediction/proba column names, SHAP
  knobs, and the full clustering knob set (see §7.4).

---

## 4. Orchestrator — `run_qa_pipeline(...)`

Resolves + prints paths, then runs (each in a `pipeline_step`): (1) CSV→parquet →
(2) embed (with a row-count assertion) → (3) model_1 classify → (4) model_2 flag
(unless `--no-model-2`) → (5) SHAP (if `--shap`) → (6) cluster (unless
`--no-cluster`). Returns the scored parquet path. **[DESIGN]** stages 4–6 are
imported lazily inside the function so a model_1-only run needn't import shap/etc.

---

## 5. Stages 1–2 — ingest & embed

### 5.1 Ingest (`workflows/qa/ingest.py`)
`build_parquet_from_csv(csv, parquet, force)` — identical logic to training ingest
(see [TRAINING_WORKFLOW.md §5](TRAINING_WORKFLOW.md)) applied to the QA CSV:
mtime/readability skip-or-reparse, the same **[HARDCODED]** `schema_overrides`,
datetime parse, derived features, atomic write.

### 5.2 Embed (`workflows/qa/embed.py`)
`embed_qa_columns(parquet, embed_dir, force, timer)` embeds the present
`EMBED_COLS` into the dataset-specific dir via the shared encoder (§8.6).
**[DESIGN]** `embeddings_need_refresh` auto-rebuilds when sidecars are missing,
older than the parquet, or row-count-mismatched — so you rarely need
`--force-embed`. After embedding, the orchestrator calls
`assert_embedding_row_counts` to guarantee row alignment before scoring.

---

## 6. Stage 3 — model_1 classify (`workflows/qa/model_1.py`)

`score_parquet_with_model_1(parquet, embed_dir, ...)`:
1. `load_model_1` — loads classifier + manifest; **requires** reduction axes (new
   `reduction/` or legacy `pc1/`) else raises with the fix command
   (`python scripts/export_model_1_pc1.py --full`).
2. Builds the feature matrix via `build_qa_feature_matrix` → the shared
   `build_inference_feature_matrix` with the manifest's **frozen** axes (§8.7).
3. `predict` → `PRED_IS_OWN_RESTAURANT` (int8); `predict_proba[:,1]` →
   `PROBA_IS_OWN_RESTAURANT` (float32). **[DESIGN]** the proba is kept so model_2
   training can use confidence as a feature and the threshold can be retuned
   without re-scoring.
4. Derives **`IF_ERROR`**: `True` when `PRED != IS_OWN_RESTAURANT`, **or when the
   ground-truth label is null** (null = anomaly). Requires `IS_OWN_RESTAURANT`
   present (errors out otherwise).
5. Atomically writes `data/qa_scored/<stem>.scored.parquet`.
6. Appends a `qa_runs.csv` row (§8.10); computes QA-time accuracy/precision/recall/
   F1/ROC-AUC opportunistically on the non-null-truth slice. **[DESIGN]** logging
   failures never break scoring (wrapped in try/except).

**Output columns added:** `PRED_IS_OWN_RESTAURANT`, `PROBA_IS_OWN_RESTAURANT`,
`IF_ERROR`.

---

## 7. Stages 4–6 — flag, explain, group

### 7.1 Stage 4 — model_2 flag (`workflows/qa/model_2.py`)

`score_parquet_with_model_2(scored_path, embed_dir, ...)`:
1. `load_model_2` — classifier + manifest (incl. `decision_threshold` and the
   model_1 reduction dir).
2. Builds the **same** feature matrix (shared helper, model_1's frozen axes) —
   model_2 sees features identical to what it trained on.
3. `predict_proba[:,1]` → `PROBA_IF_ERROR`. **[DESIGN]** flags
   `PRED_IF_ERROR = PROBA_IF_ERROR >= decision_threshold` (recall-first), falling
   back to the classifier's 0.5 argmax only if no threshold was persisted.
4. Rewrites the scored parquet in place (atomically) with the two new columns.
5. Returns a `Model2ScoreResult` (classifier, X, pred, proba, threshold,
   n_flagged) so SHAP/clustering can reuse the feature matrix without rebuilding.

**Output columns added:** `PRED_IF_ERROR`, `PROBA_IF_ERROR`.

### 7.2 Stage 5 (optional) — SHAP (`workflows/qa/shap_eval.py`)

Explains model_2. Whole dataset by default (chunked, bounded memory) or a sample
with `--shap-sample`.

**[DESIGN]** Grouped by **source column**: an individual embedding PC has no
business meaning, so each column's PCs are collapsed into one number. Since SHAP
is additive, a column's per-row contribution is the **sum of its PCs' signed
SHAP**, then mean-abs over rows (summing per-PC |SHAP| would double-count
partially-cancelling PCs). `_build_feature_groups` builds the membership one-hot;
`sv @ onehot` collapses. `_shap_values_positive_class` normalises shape
differences across SHAP/backends.

Outputs under `data/ml/model_2/shap/<stem>/`:
`shap_grouped_importance.csv` (headline, per source column), `shap_grouped_bar.png`,
`shap_feature_importance.csv` (per-PC detail), `shap_summary_beeswarm.png` (capped
to `SHAP_BEESWARM_MAX=50_000` rows — importances above remain exact),
`shap_flagged_top_contributors.parquet` (top-k source columns per flagged row),
`shap_meta.json`. Knobs: `SHAP_SAMPLE_SIZE=None`, `SHAP_CHUNK_SIZE=200_000`.

### 7.3 Stage 6 — error clustering (`workflows/qa/cluster.py`)

The most elaborate stage. Groups `IF_ERROR=True` rows into interpretable
root-cause groups. On by default (disable with `--no-cluster`).

**[DESIGN] Why cluster in SHAP space, not raw features:** clustering raw features
mostly recovers the natural data distribution (region/cuisine), not the *error
modes*. Clustering each error row by its model_2 **grouped signed-SHAP vector**
groups rows model_2 flags *for the same reason* — far closer to "same root cause".
The grouping is shared byte-for-byte with the SHAP stage.

Pipeline inside the stage:
1. Select `IF_ERROR=True` rows (aligned to `result.X`). Skip if fewer than
   `CLUSTER_MIN_ROWS=20` (writes a `skipped` report).
2. Compute per-row grouped signed SHAP for the subset (chunked), `StandardScaler`
   it (so no one column dominates distance).
3. **KMeans** with k auto-selected by **silhouette** over `CLUSTER_K_RANGE=(2,8)`
   (or pinned via `CLUSTER_K`) → "archetypes".
4. Tag each row's `ERROR_DIRECTION` (meaning-first names + embedded glossary):
   `own_not_captured` (pred own=True, label False), `unexpected_own_flag` (pred
   False, label True), `label_missing` (null label).
5. **Lineage cross-tabs:** each archetype is cross-tabbed against **every column**
   in the scored parquet (minus `CLUSTER_LINEAGE_EXCLUDE` and model outputs/ids)
   with `enrichment = cluster_share / overall_share` (>1 = over-represented →
   "look here"). Continuous numerics quantile-binned (`CLUSTER_NUMERIC_BINS=5`),
   datetimes by day, categoricals as-is; top `CLUSTER_CROSSTAB_TOP_N=20` values/col.
6. **Value-resolved groups:** each archetype is subdivided by the value-combination
   of its **main-driver** columns (those whose `total_enrichment` >
   `CLUSTER_DRIVER_IMPORTANCE_LEVEL=1.0`). Every group pins each main driver to a
   single value (one country + one app + one execution…). Kept only if ≥
   `CLUSTER_MIN_GROUP_ROWS=20` error rows (drops near-unique noise like
   `RESTAURANT_NAME`); bounded by `CLUSTER_MAX_SPLIT_COLS=6`, `CLUSTER_MAX_GROUPS=200`.
7. **Output:** `data/ml/model_2/clusters/<stem>/cluster_report.json` — a `groups`
   list (biggest first), each with `size`/`share`, `error_direction_mix`,
   `main_drivers` (`{column, total_enrichment {max,mean,sum}, value,
   value_enrichment}`), and `not_important`. Includes a `reading_guide` +
   `error_direction_glossary` so the file reads standalone. Plus
   `cluster_assignments.parquet` mapping each error row to its `ERROR_CLUSTER`
   (archetype) and final `ERROR_GROUP`.

### 7.4 Clustering knobs to revise on new data (`workflows/qa/config.py`)

`CLUSTER_LINEAGE_EXCLUDE` (currently excludes model outputs/target plus the
near-unique/free-text `CUISINES, FOOD_CATEGORIES, RESTAURANT_AVG_RATING,
RESTAURANT_NB_REVIEWS`), `CLUSTER_MIN_ROWS`, `CLUSTER_MIN_GROUP_ROWS`,
`CLUSTER_DRIVER_IMPORTANCE_LEVEL`, `CLUSTER_NUMERIC_BINS`, `CLUSTER_K[_RANGE]`,
`CLUSTER_CROSSTAB_TOP_N`, `CLUSTER_MAX_SPLIT_COLS`, `CLUSTER_MAX_GROUPS`.

---

## 8. Shared building blocks (`app/common/`) — **OVERLAP WITH TRAINING**

> **This is the overlap surface.** Every primitive below is used by *both*
> workflows. The **full descriptions live in [TRAINING_WORKFLOW.md §10](TRAINING_WORKFLOW.md)**
> (training is their canonical home); here we give a concise summary + the
> **QA-specific** usage. The hard rule: QA must extract features *identically* to
> training, which is exactly why these live in `common/`. At inference QA always
> uses **frozen** parameters — it never refits PCA/axes on the QA input.

### 8.1 Column contract — `common/config/columns.py`  *(shared)*
Source of truth for `EMBED_COLS`, `TARGET_COL=IS_OWN_RESTAURANT`,
`SPLIT_GROUP_COL=EXECUTION_ID`, `ML_TABULAR_COLS`, `ML_EXCLUDE_COLS`, the feature-
name suffixes, and `IF_ERROR_COL=IF_ERROR`. QA reads `EMBED_COLS`/`TARGET_COL`/
`IF_ERROR_COL`; the per-model `feature_columns`/`embed_cols` actually used come
from each model's manifest, not this file. Full table: training doc §10.1.

### 8.2 Derived features — `common/features/preprocess.py`  *(shared)*
QA ingest applies the same `add_derived_features` (IS_ODM/IS_QCA/
EXECUTION_ROW_COUNT/BRAND_COUNT). Same **[HARDCODED]** `PIPELINE_FLAG=="FSA"`
caveat. Detail: training doc §10.2.

### 8.3 Atomic parquet I/O — `common/storage/parquet_io.py`  *(shared)*
QA uses `atomic_write_parquet` for the scored parquet (and in-place model_2
rewrite) and `parquet_is_readable` in ingest. Detail: training doc §10.3.

### 8.4 Weighted PCA — `common/features/pca.py`  *(shared)*
Used at QA **embed** time only (the PC1 sidecar). The reduction axes QA projects
with are **not** fitted in QA — they were frozen during training. Detail: §10.4.

### 8.5 Reduction strategies — `common/features/reduction.py`  *(shared)*
QA **loads** the frozen artifacts via `load_reduction_artifacts` (new `reduction/`
layout, legacy `pc1/` fallback) — the strategy + per-column K come from the model
manifest, so QA reproduces training's projection exactly. QA never calls
`save_reduction_artifacts`. Detail: §10.5.

### 8.6 ONNX embeddings — `common/features/embeddings.py`  *(shared)*
QA calls the same `embed_text_columns` / `OnnxArcticEncoder` as training, just
targeting the dataset-specific embed dir. Same model + knobs. Detail: §10.6.

### 8.7 Frozen-projection feature builder — `common/features/inference_matrix.py`  *(shared — the key overlap)*
**This is where QA and training meet most directly.** QA model_1, QA model_2, and
model_2 *training* all call `build_inference_feature_matrix(...)` to rebuild the
exact same feature matrix from model_1's frozen axes + tabular numerics, in the
manifest's `feature_columns` order. Any drift here would silently corrupt model_2.
In QA it is reached via the thin wrapper `workflows/qa/inference_features.py`
(`build_qa_feature_matrix`, which only sets QA-specific row-chunk + step ids).
Detail: §10.7.

### 8.8 Timing — `common/pipeline_timing.py`  *(shared)*
QA wraps each stage in `pipeline_step`; the recorded timings feed the per-stage
seconds columns of `qa_runs.csv`. Detail: §10.8.

### 8.9 (Training-owned) run reports — `common/training_run_report.py`
Not used at QA scoring time (QA produces no `TrainingRunReport`). Listed for
completeness; model_2 *training* uses it. Detail: §10.9.

### 8.10 Comparison logs — `common/comparison_log.py`  *(shared)*
QA appends one row per scoring run to **`data/ml/comparison/qa_runs.csv`** via
`append_qa_row` — per-stage timings (from the pipeline timer) plus QA-time
accuracy/precision/recall/F1/ROC-AUC when ground truth is present, `n_if_error_true`,
and the reduction strategy/K aggregates pulled from the model manifest. Detail:
§10.10.

---

## 9. The training↔QA contract (what crosses the boundary)

QA depends on training **only** through these artifacts — nothing else:

| Artifact (written by training) | Read by QA for |
|---|---|
| `data/ml/model_1/classifier.joblib` | model_1 prediction |
| `data/ml/model_1/manifest.json` | `feature_columns`, `embed_cols`, reduction strategy + axes dir |
| `data/ml/model_1/reduction/*` (or legacy `pc1/*`) | frozen projection axes/means |
| `data/ml/model_2/classifier.joblib` | model_2 flagging |
| `data/ml/model_2/manifest.json` | `feature_columns`, `decision_threshold`, model_1 reduction dir |

And QA feeds back to training:

| Artifact (written by QA) | Consumed by |
|---|---|
| `data/qa_scored/<stem>.scored.parquet` | **model_2 training** (`IF_ERROR` target + features) |
| `data/<stem>_embeddings/*` | model_2 training (projects model_1's frozen axes) |

---

## 10. QA artifacts produced

```
data/<stem>.parquet
data/<stem>_embeddings/{col}_EMBEDDING.parquet, {col}_EMB_PC1.parquet
data/qa_scored/<stem>.scored.parquet   # + PRED/PROBA_IS_OWN_RESTAURANT, IF_ERROR, PRED/PROBA_IF_ERROR
data/ml/model_2/shap/<stem>/...        # if --shap
data/ml/model_2/clusters/<stem>/cluster_report.json, cluster_assignments.parquet
data/ml/comparison/qa_runs.csv         # appended
```

---

## 11. Hardcoded / to-revise when porting QA to new data

1. `workflows/qa/config.INPUT_CSV` — the QA input CSV.
2. CSV `schema_overrides` + datetime column in `workflows/qa/ingest.py`. *(mirrors training)*
3. `common/config/columns.py` + `common/features/preprocess.py` — shared with
   training (see [TRAINING_WORKFLOW.md §12](TRAINING_WORKFLOW.md)).
4. The clustering knobs in §7.4 (esp. `CLUSTER_LINEAGE_EXCLUDE`).
5. The QA workflow assumes `IS_OWN_RESTAURANT` is present in the input (it errors
   out otherwise, since `IF_ERROR` can't be derived without it).

---

## 12. QA-relevant design decisions (the "why")

1. **[DESIGN] QA never refits** — it loads model_1's frozen reduction axes from the
   manifest and projects new data through them; the shared
   `inference_matrix.build_inference_feature_matrix` guarantees features match
   training byte-for-byte.
2. **[DESIGN] Dataset-specific embed dir** (`data/<stem>_embeddings/`), never the
   training dir — prevents overwriting training sidecars.
3. **[DESIGN] `IF_ERROR` = disagreement OR null label** — nulls are treated as
   anomalies (completeness gaps), not silently passed.
4. **[DESIGN] Recall-first model_2 flagging** via the persisted `decision_threshold`
   (not 0.5 argmax) — catch nearly every suspected error.
5. **[DESIGN] SHAP grouped by source column** (signed-sum of PCs), and **clustering
   in grouped-SHAP space** — groups mean "flagged for the same reason".
6. **[DESIGN] Value-resolved error groups** pinned to single driver values, with
   enrichment scoring + noise floors, so a reviewer gets concrete "look here"
   targets rather than abstract clusters.
7. **[DESIGN] Logging never breaks scoring** — the `qa_runs.csv` append is wrapped
   in try/except.

> **Gap to note:** `docs/ARCHITECTURE.md` references a `scripts/score_dataset.py`
> that does not exist in the tree — use `python app/qa_workflow.py` instead.
```
