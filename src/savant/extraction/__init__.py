"""Agentic extraction: run extraction + critic agents over the paper corpus.

Implemented:
- `savant.agents.client` — LLMClient Protocol (Anthropic + Mock).
- `savant.agents.runner` — AgentRunner base.
- `savant.extraction.runner.ExtractionRunner` — produce a v0.3.0 record
  from a paper's chunks + bibliographic metadata.
- `savant.extraction.critic.CriticRunner` — re-verify a record against
  its source text.

Still TODO:
- [ ] PDF chunking pass that splits papers into proper section-aware chunks
      (today the UI hands a single chunk with the whole PDF text).
- [ ] Full critic loop: extract → critic → revise (≤ N rounds) → accept/reject.
"""

from .critic import (
    VALID_VERDICTS,
    CriticRunner,
    apply_verdict_to_record,
    critique_extraction,
)
from .runner import ExtractionRunner

__all__ = [
    "VALID_VERDICTS",
    "CriticRunner",
    "ExtractionRunner",
    "apply_verdict_to_record",
    "critique_extraction",
]
