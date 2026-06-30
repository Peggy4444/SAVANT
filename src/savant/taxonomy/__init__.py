"""Taxonomy normalisation: canonicalise free-text fields across extraction records.

Implemented (B5):
- `TaxonomyRunner` — one-call agent runner that returns canonical mappings
  for one field at a time.
- `collect_raw_values(field_path, records)` — aggregate corpus-wide raw
  string values to feed the runner.
- `collect_examples(field_path, records, limit)` — small (paper_id, raw)
  list for the agent's context.
- `apply_taxonomy_to_record(record, field_path, mappings)` — patch a copy
  of a record with a `_canonical` sibling field without mutating the gold
  standard.

Still TODO:
- [ ] Embedding-based pre-clustering to reduce token load on large corpora.
- [ ] CLI script that loops over the v0.3.0 free-text fields and persists
      one mapping JSON per field (planned as part of B7).
"""

from .runner import (
    DEFAULT_TAXONOMY_DIR,
    TaxonomyRunner,
    apply_taxonomy_to_record,
    collect_examples,
    collect_raw_values,
    save_taxonomy,
)

__all__ = [
    "DEFAULT_TAXONOMY_DIR",
    "TaxonomyRunner",
    "apply_taxonomy_to_record",
    "collect_examples",
    "collect_raw_values",
    "save_taxonomy",
]
