"""Agentic-pipeline runtime: LLM client abstraction + agent runner base.

See `src/savant/agents/client.py` for the `LLMClient` Protocol and its two
implementations (`AnthropicClient`, `MockClient`). See `runner.py` for the
common runner pattern subclassed by the per-agent runners
(`CriticRunner`, `TaxonomyRunner`, `GapRunner` — in their own modules).
"""

from .client import (
    MODELS_WITHOUT_TEMPERATURE,
    AnthropicClient,
    LLMCallLog,
    LLMClient,
    MockClient,
    OfflineModeError,
    prompt_hash,
    supports_temperature,
)
from .runner import AgentRunner, AgentValidationError, format_input_block

__all__ = [
    "AgentRunner",
    "AgentValidationError",
    "AnthropicClient",
    "LLMCallLog",
    "LLMClient",
    "MODELS_WITHOUT_TEMPERATURE",
    "MockClient",
    "OfflineModeError",
    "format_input_block",
    "prompt_hash",
    "supports_temperature",
]
