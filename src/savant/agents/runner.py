"""Common runner pattern for every SAVANT agent.

`AgentRunner` is an abstract base class. Subclasses set `AGENT_NAME` (which
indexes their config in `configs/agents.yaml`) and implement
`_render_user_message`. They optionally override `_validate` to enforce an
output schema.

The runner:
1. Loads `agents.yaml` (defaults + per-agent overrides).
2. Loads the prompt from `prompts/{agent}_agent.md`.
3. Renders inputs into a user message (subclass-specific).
4. Calls `client.complete()` with the resolved model/temperature/max_tokens.
5. Retries on JSON-parse errors using the configured backoff.
6. Validates the response (subclass-specific).
7. Returns the dict.

Subclass quick example:

    class CriticRunner(AgentRunner):
        AGENT_NAME = "critic"
        def _render_user_message(self, *, record, source_excerpts):
            return f"## Record\n```json\n{json.dumps(record)}\n```\n\n## Source\n{source_excerpts}"
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml

from ..utils.config import repo_root
from .client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_AGENTS_YAML = repo_root() / "configs" / "agents.yaml"


class AgentValidationError(Exception):
    """Raised by `AgentRunner._validate` when the response is well-formed
    JSON but does not satisfy the agent's output contract. NOT retryable —
    surfaces immediately to the caller."""


def format_input_block(label: str, content: Any) -> str:
    """Uniformly-formatted input section for user messages.

    - Strings render as fenced text.
    - Dicts / lists render as fenced JSON.
    - Other types stringify.
    """
    if isinstance(content, (dict, list)):
        body = json.dumps(content, indent=2, ensure_ascii=False)
        fence = "```json"
    elif isinstance(content, str):
        body = content
        fence = "```"
    else:
        body = str(content)
        fence = "```"
    return f"## {label}\n\n{fence}\n{body}\n```\n"


class AgentRunner(ABC):
    """Base class for one-shot agent calls. Subclass per-agent."""

    #: Indexes into `configs/agents.yaml::agents`. Subclasses MUST set.
    AGENT_NAME: ClassVar[str]

    def __init__(self, client: LLMClient,
                 config_path: Path | None = None) -> None:
        self.client = client
        self.config_path = config_path or DEFAULT_AGENTS_YAML
        full_config = yaml.safe_load(self.config_path.read_text())
        self.defaults = full_config.get("defaults", {}) or {}
        self.agent_config = full_config["agents"][self.AGENT_NAME]
        self.prompt = self._load_prompt()

    # --- Subclass extension points ------------------------------------------

    @abstractmethod
    def _render_user_message(self, **inputs: Any) -> str:
        """Convert per-call inputs into the user-message string."""

    def _validate(self, response: dict[str, Any]) -> dict[str, Any]:
        """Optionally validate response. Default: no-op. Subclasses that
        validate should raise `AgentValidationError` on failure — that
        error is NOT caught by the retry loop and surfaces to the caller."""
        return response

    # --- Internals -----------------------------------------------------------

    def _load_prompt(self) -> str:
        rel = self.agent_config["prompt"]
        prompt_path = repo_root() / rel
        return prompt_path.read_text()

    @property
    def model(self) -> str:
        return self.agent_config.get("model") or self.defaults.get("model") or ""

    @property
    def temperature(self) -> float:
        if "temperature" in self.agent_config:
            return float(self.agent_config["temperature"])
        return float(self.defaults.get("temperature", 0.0))

    @property
    def max_tokens(self) -> int:
        if "max_tokens" in self.agent_config:
            return int(self.agent_config["max_tokens"])
        return int(self.defaults.get("max_tokens", 4096))

    def _retry_policy(self) -> tuple[int, list[int]]:
        retry = self.defaults.get("retry", {}) or {}
        return (int(retry.get("attempts", 1) or 1),
                list(retry.get("backoff_seconds", []) or []))

    # --- Main entry point ---------------------------------------------------

    def run(self, **inputs: Any) -> dict[str, Any]:
        """Render inputs → call LLM → validate → return dict.

        Retries on JSON-parse errors (`ValueError` from `parse_json_response`)
        per the `defaults.retry` policy in `agents.yaml`. Other exceptions
        propagate immediately."""
        user = self._render_user_message(**inputs)
        attempts, backoffs = self._retry_policy()
        last_exc: Exception | None = None
        for attempt_idx in range(attempts):
            try:
                response = self.client.complete(
                    system=self.prompt,
                    user=user,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    agent=self.AGENT_NAME,
                )
                return self._validate(response)
            except ValueError as exc:
                last_exc = exc
                if attempt_idx + 1 < attempts:
                    wait = backoffs[min(attempt_idx, len(backoffs) - 1)] \
                        if backoffs else 1
                    logger.warning(
                        "agent=%s json-parse failed on attempt %d/%d; sleeping %ds",
                        self.AGENT_NAME, attempt_idx + 1, attempts, wait,
                    )
                    time.sleep(wait)
                    continue
                # exhausted retries
                raise
        # Defensive — unreachable.
        assert last_exc is not None
        raise last_exc
