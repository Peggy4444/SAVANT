"""LLM client tests (WP 2 — Step B1).

All tests use MockClient or assert constructor behaviour. No live API calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from savant.agents.client import (
    AnthropicClient,
    LLMCallLog,
    MockClient,
    OfflineModeError,
    _append_log,
    parse_json_response,
    prompt_hash,
)


# --- Pure-function tests -----------------------------------------------------

def test_prompt_hash_is_stable_and_short() -> None:
    assert prompt_hash("system", "user") == prompt_hash("system", "user")
    assert prompt_hash("a", "b") != prompt_hash("a", "c")
    assert len(prompt_hash("x")) == 16


def test_parse_json_strips_code_fences() -> None:
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('```\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_skips_leading_prose() -> None:
    """Models sometimes prepend explanation before the JSON."""
    response = "Here's the JSON you asked for:\n\n{\"verdict\": \"accept\"}"
    assert parse_json_response(response) == {"verdict": "accept"}


def test_parse_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError, match="did not return valid JSON"):
        parse_json_response("nope nope nope")


# --- MockClient tests --------------------------------------------------------

def test_mock_client_replays_responses_in_order() -> None:
    client = MockClient(responses=[{"verdict": "accept"}, {"verdict": "revise"}])
    a = client.complete(system="s", user="u", model="m", temperature=0,
                        max_tokens=100, agent="critic")
    b = client.complete(system="s", user="u2", model="m", temperature=0,
                        max_tokens=100, agent="critic")
    assert a == {"verdict": "accept"}
    assert b == {"verdict": "revise"}


def test_mock_client_records_call_metadata() -> None:
    client = MockClient(responses=[{"ok": True}])
    client.complete(system="role: critic", user="data here", model="claude-x",
                    temperature=0.1, max_tokens=200, agent="critic")
    assert len(client.calls) == 1
    c = client.calls[0]
    assert c["agent"] == "critic"
    assert c["model"] == "claude-x"
    assert c["temperature"] == 0.1
    assert c["max_tokens"] == 200
    assert c["system"] == "role: critic"
    assert c["user"] == "data here"
    assert len(c["prompt_hash"]) == 16


def test_mock_client_raises_when_queue_empty() -> None:
    client = MockClient()  # no responses queued
    with pytest.raises(RuntimeError, match="no canned response"):
        client.complete(system="s", user="u", model="m", temperature=0,
                        max_tokens=100, agent="critic")


def test_mock_client_supports_queue_method() -> None:
    client = MockClient()
    client.queue({"a": 1})
    client.queue({"a": 2})
    assert client.complete(system="s", user="u", model="m", temperature=0,
                            max_tokens=10, agent="x")["a"] == 1
    assert client.complete(system="s", user="u", model="m", temperature=0,
                            max_tokens=10, agent="x")["a"] == 2


# --- AnthropicClient: constructor + offline mode ----------------------------

def test_anthropic_client_offline_flag_blocks_call() -> None:
    client = AnthropicClient(api_key="dummy", offline=True)
    with pytest.raises(OfflineModeError):
        client.complete(system="s", user="u", model="m", temperature=0,
                        max_tokens=10, agent="x")


def test_anthropic_client_no_key_blocks_call(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(api_key=None, offline=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        client.complete(system="s", user="u", model="m", temperature=0,
                        max_tokens=10, agent="x")


def test_anthropic_client_env_offline_respected(monkeypatch) -> None:
    monkeypatch.setenv("SAVANT_OFFLINE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    client = AnthropicClient()
    with pytest.raises(OfflineModeError):
        client.complete(system="s", user="u", model="m", temperature=0,
                        max_tokens=10, agent="x")


# --- Logging round-trip ------------------------------------------------------

def test_log_lines_are_appended_to_dated_jsonl(tmp_path: Path) -> None:
    log = LLMCallLog(
        ts="2026-06-30T12:00:00+00:00",
        agent="critic", model="claude-x", prompt_hash="abc",
        input_tokens=100, output_tokens=200,
        duration_s=1.23, response_kind="json",
    )
    _append_log(log, tmp_path)
    f = tmp_path / "2026-06-30.jsonl"
    assert f.exists()
    line = f.read_text().strip()
    import json as _json
    rec = _json.loads(line)
    assert rec["agent"] == "critic"
    assert rec["input_tokens"] == 100
    assert rec["response_kind"] == "json"
