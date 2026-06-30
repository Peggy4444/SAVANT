"""LLM client abstraction.

CLAUDE.md §2 mandates "no silent network calls — LLM fetches go through a
single client module that logs every request and respects an offline-mode
flag." This module is that single seam.

Two implementations:
- `AnthropicClient`: wraps `anthropic.Anthropic`, parses JSON responses,
  logs every call to `data/outputs/llm_logs/{YYYY-MM-DD}.jsonl`.
- `MockClient`: queue-based stub for tests and dry runs. No network. No
  spend. Records every call so tests can assert on prompt content.

Both satisfy the `LLMClient` Protocol so downstream agent runners are
implementation-agnostic.

Offline mode (env var `SAVANT_OFFLINE=1`) makes `AnthropicClient.complete`
raise `OfflineModeError` — agent code paths that require an LLM should
catch this and produce a clear, actionable message ("set ANTHROPIC_API_KEY
or pass `--mock`").
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..utils.config import repo_root

logger = logging.getLogger(__name__)


class OfflineModeError(RuntimeError):
    """Raised when an Anthropic call is attempted while SAVANT_OFFLINE is set."""


# Models that REJECT the `temperature` parameter (typically the
# "extended thinking" / reasoning models). The SDK raises a 400
# 'temperature is deprecated for this model' if we send it.
MODELS_WITHOUT_TEMPERATURE: frozenset[str] = frozenset({
    "claude-opus-4-7",
    "claude-opus-4-7[1m]",
})


def supports_temperature(model: str) -> bool:
    """True if this model accepts the `temperature` kwarg on the
    `messages.create` endpoint. False for reasoning models that deprecate it."""
    return model not in MODELS_WITHOUT_TEMPERATURE


@dataclass(frozen=True)
class LLMCallLog:
    """One log line per LLM call. Persisted as JSONL for cost / provenance audit."""

    ts: str  # ISO-8601 UTC
    agent: str  # which SAVANT agent originated the call
    model: str
    prompt_hash: str
    input_tokens: int | None
    output_tokens: int | None
    duration_s: float
    response_kind: str  # "json" | "raw" | "error"
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def prompt_hash(*parts: str) -> str:
    """Stable hash of (system, user, ...) — first 16 hex chars of SHA-256.
    Used for cost-attribution and the `provenance.prompt_hash` field."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class LLMClient(Protocol):
    """Minimal LLM-call contract shared by Anthropic + Mock."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        agent: str = "unknown",
    ) -> dict[str, Any]:
        """Send (system, user) to the LLM, parse JSON response, return dict.

        Implementations MUST log the call. Implementations MUST raise on
        malformed JSON.
        """
        ...


# --- Helpers -----------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Strip leading/trailing ```json ... ``` fences so json.loads works."""
    return _FENCE_RE.sub("", text).strip()


def parse_json_response(raw: str) -> dict[str, Any]:
    """Tolerant JSON parser for model output. Strips code fences and
    leading prose. Raises ValueError with a useful preview on failure."""
    cleaned = _strip_code_fence(raw)
    # Some models prepend prose before the JSON object — find the first '{'.
    first = cleaned.find("{")
    if first > 0:
        cleaned = cleaned[first:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        preview = cleaned[:300].replace("\n", " ")
        raise ValueError(
            f"Model did not return valid JSON ({exc.msg}). First 300 chars: {preview!r}"
        ) from exc


def default_log_dir() -> Path:
    return repo_root() / "data" / "outputs" / "llm_logs"


def _append_log(log: LLMCallLog, log_dir: Path | None = None) -> None:
    log_dir = log_dir or default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    date_part = log.ts.split("T", 1)[0]
    path = log_dir / f"{date_part}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")


# --- Anthropic implementation ------------------------------------------------

class AnthropicClient:
    """Production LLM client. Uses the `anthropic` SDK + your API key.

    Set `ANTHROPIC_API_KEY` in the environment (or in `.env`; we load it via
    python-dotenv in calling scripts). Set `SAVANT_OFFLINE=1` to refuse all
    calls (useful in CI to ensure tests don't accidentally spend money).
    """

    def __init__(self, api_key: str | None = None,
                 log_dir: Path | None = None,
                 offline: bool | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._log_dir = log_dir
        self._offline = (
            bool(int(os.environ.get("SAVANT_OFFLINE", "0")))
            if offline is None else offline
        )
        self._sdk_client = None  # lazily constructed on first call

    def _ensure_sdk(self):
        if self._offline:
            raise OfflineModeError(
                "AnthropicClient.complete() refused: SAVANT_OFFLINE is set "
                "or offline=True was passed. Use MockClient for tests."
            )
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Either export it, put it in "
                "a .env file, or pass api_key= to AnthropicClient()."
            )
        if self._sdk_client is None:
            import anthropic
            self._sdk_client = anthropic.Anthropic(api_key=self._api_key)
        return self._sdk_client

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        agent: str = "unknown",
    ) -> dict[str, Any]:
        sdk = self._ensure_sdk()
        ph = prompt_hash(system, user)
        t0 = time.perf_counter()
        log_extras: dict[str, Any] = {}
        # Some models (e.g. claude-opus-4-7) deprecate the temperature kwarg.
        call_kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if supports_temperature(model):
            call_kwargs["temperature"] = temperature
        try:
            resp = sdk.messages.create(**call_kwargs)
        except Exception as exc:
            _append_log(LLMCallLog(
                ts=datetime.now(timezone.utc).isoformat(),
                agent=agent, model=model, prompt_hash=ph,
                input_tokens=None, output_tokens=None,
                duration_s=round(time.perf_counter() - t0, 4),
                response_kind="error", error=str(exc),
            ), self._log_dir)
            raise

        # Extract concatenated text from the response content blocks
        raw = "".join(getattr(block, "text", "") for block in resp.content)
        input_tokens = getattr(resp.usage, "input_tokens", None)
        output_tokens = getattr(resp.usage, "output_tokens", None)
        try:
            parsed = parse_json_response(raw)
            kind = "json"
        except ValueError as exc:
            log_extras["raw_preview"] = raw[:300]
            _append_log(LLMCallLog(
                ts=datetime.now(timezone.utc).isoformat(),
                agent=agent, model=model, prompt_hash=ph,
                input_tokens=input_tokens, output_tokens=output_tokens,
                duration_s=round(time.perf_counter() - t0, 4),
                response_kind="error", error=str(exc), extras=log_extras,
            ), self._log_dir)
            raise

        _append_log(LLMCallLog(
            ts=datetime.now(timezone.utc).isoformat(),
            agent=agent, model=model, prompt_hash=ph,
            input_tokens=input_tokens, output_tokens=output_tokens,
            duration_s=round(time.perf_counter() - t0, 4),
            response_kind=kind, extras=log_extras,
        ), self._log_dir)
        return parsed


# --- Mock implementation for tests ------------------------------------------

class MockClient:
    """Queue-based stub. No network, no spend. Useful for tests, dry runs,
    and CI. Construct with a list of canned dict responses; each `complete()`
    call pops one in FIFO order. Records every call to `.calls`.

    Example
    -------
    >>> client = MockClient(responses=[{"verdict": "accept"}])
    >>> client.complete(system="...", user="...", model="m", temperature=0,
    ...                 max_tokens=100, agent="critic")
    {'verdict': 'accept'}
    >>> client.calls[0]["agent"]
    'critic'
    """

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses: list[dict[str, Any]] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def queue(self, response: dict[str, Any]) -> None:
        self.responses.append(response)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        agent: str = "unknown",
    ) -> dict[str, Any]:
        if not self.responses:
            raise RuntimeError(
                f"MockClient: no canned response queued for agent={agent!r}. "
                f"Call .queue(response) before the agent runs."
            )
        self.calls.append({
            "agent": agent, "model": model, "temperature": temperature,
            "max_tokens": max_tokens, "system": system, "user": user,
            "prompt_hash": prompt_hash(system, user),
        })
        return self.responses.pop(0)
