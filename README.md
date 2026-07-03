A project / master's thesis to automate the QA process. Initially on ODM (former FSA) and an energy case, later on DSM and CMI as well.

## ML pipeline (stage 4)

Classifiers use **`data/ml/cache/<id>/model_input.parquet`** — tabular numerics plus train-fitted embedding-reduction features (no raw text / embed-source columns).

The reduction strategy is configurable. Recognised values (resolved by `common.features.reduction.parse_strategy`):

| Spec | Description | Default `k_max` |
|---|---|---:|
| `pc1` | Scalar PC1 per column (legacy default; K=1) | 1 |
| `topN` (e.g. `top5`) | Fixed top-N principal components per column | N |
| `adaptive_0.90` | Adaptive K per column until cumulative variance ≥ 0.90 | 32 |
| `adaptive_0.95` | Adaptive K per column until cumulative variance ≥ 0.95 | 64 |
| `adaptive_0.99` | Adaptive K per column until cumulative variance ≥ 0.99 | 128 |
| `raw` | No projection — 768 features per embed column | n/a |

Override per-run with `--reduction <spec>` on the training entry point; the cache key includes the strategy so swaps never reuse stale features. The model_1 manifest carries the strategy + per-column K + cumulative variance, and QA / model_2 load it back at inference time. Legacy `pc1/` artifacts from older trainings remain readable.

### Comparing strategies (CSV logs)

Every training and QA run appends rows to `data/ml/comparison/`:

| File | Rows | Use |
|---|---|---|
| `training_runs.csv` | 1 per (run, backend) | Compare strategies by training time, n_features, info content, precision/recall/F1/ROC-AUC |
| `qa_runs.csv` | 1 per QA run | Compare scoring time per stage + QA-time precision when ground truth is present |
| `reduction_per_column.csv` | 1 per (run, embed column) | Inspect K and cumulative variance achieved per column |

Run the whole sweep in one go:

```powershell
# Smoke comparison (80 executions, all 5 default strategies, no QA scoring)
python scripts/compare_reductions.py

# Only specific strategies, with QA scoring after each training
python scripts/compare_reductions.py --strategies pc1,top5,adaptive_0.95 --score-csv data/ODM_Latam_26_mar_FN1.csv
```

Each strategy's model_1 artifacts are written to `data/ml/comparison/<run_stamp>/model_1_<strategy>/` so the canonical `data/ml/model_1/` is never clobbered. To score with a per-strategy model directly: `python app/qa_workflow.py --model-1-dir data/ml/comparison/<run_stamp>/model_1_<strategy>`.

### Fix broken `.venv` pip

If `pip install` fails with a wrong nested `.venv` path, use:

```powershell
python -m pip install -r requirements.txt
```

Or recreate the venv: `python -m venv .venv` then `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

### Commands (run from repo root)

Two entry points, one per workflow:

```powershell
# Training: smoke run (80 executions, reuses cache when present)
python app/training_workflow.py

# Training: all executions
python app/training_workflow.py --full

# Training: rebuild the model_input cache (slow first run; instant on reuse)
python app/training_workflow.py --rebuild

# Training: rebuild parquet + embeddings + cache + train from scratch
python app/training_workflow.py --from-scratch

# Training: also train + compare model_2 (IF_ERROR) on QA scored output
python app/training_workflow.py --train-model-2

# Training: only run the model_2 stage (skip model_1)
python app/training_workflow.py --only-model-2

# QA: classify the CSV in workflows/qa/config.py (model_1 + model_2 + clustering)
python app/qa_workflow.py

# QA: also write the global SHAP importance report for model_2
python app/qa_workflow.py --shap

# QA: skip the error-clustering stage
python app/qa_workflow.py --no-cluster
```

### Error clustering (root-cause grouping, QA stage 6)

Runs after model_2 **by default** (disable with `--no-cluster`). It takes the
rows where `IF_ERROR=True` (model_1 disagreed with the imported
`IS_OWN_RESTAURANT` label, or the label was null) and groups them into error
**archetypes** by clustering each row's *model_2 grouped-SHAP signature* (per-row
signed SHAP summed into source columns) with KMeans — so rows are grouped by
*why* they look like errors, not by the natural data distribution. `k` is
auto-selected by silhouette (override via `CLUSTER_K` in `workflows/qa/config.py`).

Each archetype is then cross-tabbed against **every column** in the scored parquet
(override via `CLUSTER_LINEAGE_COLS`; model outputs / target / near-unique ids are
excluded by `CLUSTER_LINEAGE_EXCLUDE`) with a `cluster_share / overall_share`
enrichment score. Continuous numeric columns are quantile-binned
(`CLUSTER_NUMERIC_BINS`) so they yield interpretable bands rather than one row per
value; datetimes bucket by day; categoricals/booleans are used as-is.

**From archetypes to value-resolved groups.** A single archetype can still mix
many concrete culprits (Mexico vs Colombia, exec A vs exec B). So each archetype
is **subdivided by the value-combination of its main-driver columns** — the
columns whose `total_enrichment` (sum of their over-represented values'
enrichment) clears `CLUSTER_DRIVER_IMPORTANCE_LEVEL` (default 1.0). The result is
one group per concrete pattern, where every `main_driver` column is pinned to a
single value (one country + one parent_app + one execution_id, …). A value/sub-
group is only kept when it holds at least `CLUSTER_MIN_GROUP_ROWS` (default 20)
error rows — this drops near-unique noise (e.g. `RESTAURANT_NAME`) and bounds the
group count (further capped by `CLUSTER_MAX_SPLIT_COLS` / `CLUSTER_MAX_GROUPS`).

Everything lands in **one consolidated file**, `data/ml/model_2/clusters/<scored_stem>/cluster_report.json`.
`groups` is a **tree**: it branches first by the **error direction** (meaning-first
names — `own_not_captured`, `unexpected_own_flag`, `label_missing`). Direction is a
hard partition: every group is direction-pure, so each top-level node holds exactly
one direction (its `error_direction_mix` has a single key). Beneath that it branches
by the 1st `main_driver` (`column`=`value`), then the 2nd, and so on. Groups that share a
prefix share branches, so each path down the tree spells out one error pattern
and common sub-patterns are stated once. Every node carries a `size` / `share`;
driver nodes also carry `{column, value, total_enrichment, value_enrichment}`,
where `total_enrichment` is the column's strength in the parent archetype as
`{max, mean, sum}` over its over-represented values' enrichment, and
`value_enrichment` is how over-represented that node's value is vs all errors
(>1 = over-represented). Branch order follows the main-driver order
(`total_enrichment.max`, descending — the strongest single value sits at the
root). A branch is **cut at the first driver whose `value_enrichment` falls below
`CLUSTER_DRIVER_IMPORTANCE_LEVEL`**: that weak value and everything beneath it are
hidden, so a group terminates at its last strong driver. Columns in `CLUSTER_DRIVER_EXCLUDE` (e.g. the high-cardinality
`RESTAURANT_BRAND_NAMES`) never branch; instead `CLUSTER_DESCRIBE_UNDER` hangs
their raw value→count breakdown as a `descriptions` field under a proxy driver
node (so the actual brand-list arrays show up under the `BRAND_COUNT` node). A
leaf (no `drivers`) is a final group: it
carries its `group` id, `archetype`, and `not_important` (the model's other
suspected columns that were *not* over-represented in that archetype). A
`reading_guide` and `error_direction_glossary` are embedded so the file reads
standalone, and `cluster_assignments.parquet` maps each scored row to its
`ERROR_CLUSTER` (archetype) and final `ERROR_GROUP`.

Every training run (model_1 and model_2) writes a timestamped record to
`data/ml/<model>/runs/<run_id>_<model>/run_report.{json,md}` containing the full
config, feature list, split sizes, per-model hyperparameters, fit/predict
timings, accuracy/precision/recall/F1/ROC-AUC, classification report, confusion
matrix, top feature importances, and the chosen winning backend. Reports never
overwrite each other, so iterations can be compared exact-value side-by-side.

First cache build for a subsample still scans full embedding sidecars once per text column (~20 min per column on CPU). Re-runs with the same data skip that step.

**Architecture & call graph:** see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**If QA fails with missing reduction axes** under `data/ml/model_1/reduction/` (or `data/ml/model_1/pc1/` for legacy models):

```powershell
python scripts/export_model_1_pc1.py --full
```

**QA workflow** (classify a CSV with model_1):

1. Edit `INPUT_CSV` in `app/workflows/qa/config.py`
2. Run `python app/qa_workflow.py`

Paths are automatic: `<csv>.parquet`, `data/<csv_stem>_embeddings/`, `data/qa_scored/<csv_stem>.scored.parquet`. Embeddings refresh when missing, outdated, or row-mismatched — no manual `--force-embed` unless you want a full rebuild.

**Progress / timing:** `app/common/pipeline_timing.py`. **Layout:** `app/workflows/training/` and `app/workflows/qa/` — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
