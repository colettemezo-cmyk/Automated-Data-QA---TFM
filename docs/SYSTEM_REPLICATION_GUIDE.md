# System Replication Guide — Index

This guide was split into **two workflow-specific documents**. Each is
self-contained and replicable on its own; the shared `app/common/` primitives
(the **overlap**) are flagged in both.

| Document | Covers |
|---|---|
| **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** | Fitting & exporting `model_1` (`IS_OWN_RESTAURANT`) and `model_2` (`IF_ERROR`): ingest → embed → leakage-safe model_input + frozen reduction axes → classifier compare/export → model_2. **Canonical home for the full `common/` primitive descriptions** (§10). |
| **[QA_WORKFLOW.md](QA_WORKFLOW.md)** | Scoring a new CSV: ingest → embed → model_1 classify → `IF_ERROR` → model_2 flag → SHAP → error clustering. References the shared primitives and adds QA-specific (frozen-axes) usage. |

## The two workflows are independent

```
TRAINING                                          QA / SCORING
labelled CSV → models                             new CSV → predictions + analysis
   │                                                  │
   ├─ model_1 ─→ data/ml/model_1/{classifier,         ├─ load model_1 (frozen axes from manifest)
   │             manifest, reduction/} ───────────────┤   → PRED/PROBA_IS_OWN_RESTAURANT, IF_ERROR
   │                                                  ├─ load model_2 → PROBA_IF_ERROR ≥ threshold
   │                                                  ├─ SHAP (grouped) + clustering
   └─ model_2 ←── data/qa_scored/*.scored.parquet ◄───┘   → data/qa_scored/<stem>.scored.parquet
      (trains on QA output)                                  (becomes model_2's training corpus)
```

**[DESIGN]** Neither workflow imports the other. The only coupling is on-disk
artifacts (see the contract tables in each doc). Everything that must behave
identically across both lives in `app/common/` — documented once in
[TRAINING_WORKFLOW.md §10](TRAINING_WORKFLOW.md) and cross-referenced from
[QA_WORKFLOW.md §8](QA_WORKFLOW.md).

## Shared primitives (the overlap), at a glance

`config/columns.py` (column contract) · `features/preprocess.py` (derived
features) · `storage/parquet_io.py` (atomic I/O) · `features/pca.py`
(frequency-weighted PCA) · `features/reduction.py` (reduction strategies) ·
`features/embeddings.py` (ONNX Arctic encoder) · `features/inference_matrix.py`
(**frozen-axis feature builder — the key shared seam**) · `pipeline_timing.py` ·
`training_run_report.py` · `comparison_log.py`.

## Decision-attribution caveat

Throughout both docs, decisions are tagged **[DESIGN]** / **[TUNABLE]** /
**[HARDCODED]** and reconstructed from code + in-line rationale. The project's
agent-transcript store was empty/unavailable, so I could not attribute
turn-by-turn which decisions were yours vs. mine — only that each is intentional
and why the code says it exists.
