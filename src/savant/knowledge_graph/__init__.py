"""Knowledge graph construction over the SAVANT extraction records.

See `src/savant/knowledge_graph/README.md` for the full schema design — node
types, edge types, attributes, and example queries.
"""

from .build import (
    EDGE_TYPES,
    KG_SCHEMA_VERSION,
    NODE_TYPES,
    build_kg,
    load_all_extractions,
    save_kg,
)
from .empirical import attach_benchmark_results, summarise_enrichment
from .integration import coverage_diff, gap_diff, nearest_neighbours

__all__ = [
    "EDGE_TYPES",
    "KG_SCHEMA_VERSION",
    "NODE_TYPES",
    "attach_benchmark_results",
    "build_kg",
    "coverage_diff",
    "gap_diff",
    "load_all_extractions",
    "nearest_neighbours",
    "save_kg",
    "summarise_enrichment",
]
