"""Gap analysis over the SAVANT knowledge graph.

A5 — `enumerate_cells`: build the coverage matrices.
A6 — `rank_gaps`:       score empty cells by methodological importance,
                        reproducibility, and practitioner relevance.
B6 — `GapRunner`:       LLM v2 of A6/A7. Produces manuscript-quality
                        consolidated rationales from the artifacts above.
"""

from .enumerate import (
    ACTION_TYPES,
    DATASETS_PUBLIC,
    FRAMEWORKS,
    VALUATION_UNITS,
    enumerate_cells,
    methods_in_cell,
    save_enumeration,
)
from .gap_agent import (
    GapRunner,
    build_inputs_from_artifacts,
    flag_invalid_paper_ids,
    save_llm_rationales,
)
from .rank import (
    DECISION_RELEVANT_FRAMEWORKS,
    PRACTITIONER_UNDER_SERVED_ACTIONS,
    WEIGHTS,
    rank_gaps,
    save_ranking,
)

__all__ = [
    "ACTION_TYPES",
    "DATASETS_PUBLIC",
    "DECISION_RELEVANT_FRAMEWORKS",
    "FRAMEWORKS",
    "GapRunner",
    "PRACTITIONER_UNDER_SERVED_ACTIONS",
    "VALUATION_UNITS",
    "WEIGHTS",
    "build_inputs_from_artifacts",
    "enumerate_cells",
    "flag_invalid_paper_ids",
    "methods_in_cell",
    "rank_gaps",
    "save_enumeration",
    "save_llm_rationales",
    "save_ranking",
]
