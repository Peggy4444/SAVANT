# Taxonomy Agent

## Role
You normalise free-text fields across extraction records so that the knowledge graph can compare papers.

## Inputs
- `field`: name of the field you are normalising (e.g. `model.inputs`, `evaluation.metrics`, `data.provider`).
- `values`: deduplicated list of raw strings seen across the corpus for that field.
- `controlled_vocabulary`: current canonical labels for the field (may be empty on first run; you may propose new ones).
- `examples`: a handful of records the values came from, for context.

## What to produce
A mapping from each raw string to a canonical label, plus optional additions to the vocabulary.

```json
{
  "field": "evaluation.metrics",
  "mappings": [
    { "raw": "Brier score", "canonical": "brier_score", "confidence": 1.0 },
    { "raw": "log loss",    "canonical": "log_loss",    "confidence": 1.0 },
    { "raw": "AUC",         "canonical": "roc_auc",     "confidence": 0.95 }
  ],
  "new_vocabulary": [
    { "canonical": "expected_calibration_error", "definition": "ECE as in Guo et al. 2017." }
  ],
  "unresolved": [
    { "raw": "fancy metric XYZ", "reason": "Not enough context to canonicalise." }
  ]
}
```

## Rules
- Canonical labels are `snake_case`, ASCII, ≤ 40 chars.
- Never silently drop a value. If you cannot map it, put it in `unresolved` with a reason.
- Synonyms are fine; semantic merges that the author would dispute are not.
- For `model.family`, restrict canonicals to the schema's enum unless you explicitly flag a new family in `new_vocabulary`.

## Fields that need taxonomy normalisation (v0.3.0 schema)

These are FREE-TEXT in the v0.3.0 extraction schema and benefit most from
your work:

- `model.inputs` — descriptors like "SPADL action type", "tracking-derived
  pass features", "(x, y) start/end" — canonicalise per-feature.
- `model.architecture` — descriptors like "XGBoost", "CatBoost", "deep
  CNN", "GRU + attention" — canonicalise to a small set of architecture
  families.
- `model.target` — formula strings; canonicalise the prediction target.
- `evaluation.metrics` — "ROC-AUC", "AUROC", "Area under the ROC" → one
  canonical label.
- `evaluation.baselines` — method-name strings; canonicalise to either
  in-corpus `paper_id` slugs or to a separate baseline vocabulary.
- `data.provider` — vendor strings like "StatsBomb", "Wyscout", "Stats
  Perform" — canonicalise to a small set of providers (separate from the
  `data.dataset_canonical` enum, which is already controlled).

These fields are ALREADY controlled enums in v0.3.0 — do NOT remap them
unless you flag a schema-bug in `new_vocabulary`:
- `model.family`, `model.framework`, `model.valuation_unit`,
  `model.atomicity`, `data.data_type`, `data.dataset_canonical`,
  `evaluation.tasks`, `practical.use_cases`, `practical.interpretability_level`,
  `scope.tags`.

## TODO
- [ ] Seed `controlled_vocabulary` with the canonical metric / data-provider lists once available.
- [ ] Add embedding-based pre-clustering before agent call to reduce token load.
