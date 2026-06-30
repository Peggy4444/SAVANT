"""Critic-agent runner tests (WP 2 — Step B4).

All tests use MockClient. No live API.
"""

from __future__ import annotations

import json

import pytest

from savant.agents import AgentValidationError, MockClient
from savant.extraction.critic import (
    CriticRunner,
    apply_verdict_to_record,
    critique_extraction,
)


_MINIMAL_RECORD: dict = {
    "schema_version": "0.3.0",
    "paper_id": "decroos_2019_vaep",
    "bibliographic": {"title": "VAEP", "authors": ["X"], "year": 2019, "venue": "KDD"},
    "model": {"name": "VAEP", "family": "VAEP", "valuation_unit": "action"},
    "provenance": {
        "extracted_at": "2026-06-29T00:00:00Z",
        "agent": "extraction_agent",
        "model_id": "test/fixture",
        "prompt_hash": "abcdef",
        "source_chunks": [],
        "critic": {"verdict": "skipped", "model_id": "n/a", "notes": "Initial batch."},
    },
}

_GOOD_VERDICT = {
    "verdict": "accept",
    "schema_conformant": True,
    "factual_issues": [],
    "completeness_issues": [],
    "style_issues": [],
    "revision_instructions": "",
    "notes": "Looks good.",
}


# --- Runner smoke tests ----------------------------------------------------

def test_critic_runner_passes_record_and_source_to_client() -> None:
    client = MockClient(responses=[_GOOD_VERDICT])
    runner = CriticRunner(client)
    out = runner.run(
        record=_MINIMAL_RECORD,
        source="Section 4 of the paper reports ROC-AUC = 0.83.",
    )
    assert out == _GOOD_VERDICT
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["agent"] == "critic"
    # The serialised record + source must both appear in the user message
    assert "decroos_2019_vaep" in call["user"]
    assert "ROC-AUC" in call["user"]
    # The system prompt is the critic prompt
    assert "sceptical reviewer" in call["system"].lower()


def test_critique_extraction_helper_returns_dict() -> None:
    client = MockClient(responses=[_GOOD_VERDICT])
    out = critique_extraction(_MINIMAL_RECORD, "src", client)
    assert out["verdict"] == "accept"


# --- Validation tests ------------------------------------------------------

def test_runner_rejects_invalid_verdict() -> None:
    client = MockClient(responses=[{"verdict": "looks-fine"}])
    with pytest.raises(AgentValidationError, match="verdict"):
        critique_extraction(_MINIMAL_RECORD, "src", client)


def test_runner_rejects_non_boolean_schema_conformant() -> None:
    client = MockClient(responses=[{"verdict": "accept", "schema_conformant": "yes"}])
    with pytest.raises(AgentValidationError, match="schema_conformant"):
        critique_extraction(_MINIMAL_RECORD, "src", client)


def test_runner_accepts_minimal_valid_verdict() -> None:
    """Only `verdict` is required; absence of other fields is OK."""
    client = MockClient(responses=[{"verdict": "reject"}])
    out = critique_extraction(_MINIMAL_RECORD, "src", client)
    assert out == {"verdict": "reject"}


# --- apply_verdict_to_record -----------------------------------------------

def test_apply_verdict_patches_provenance_critic_only() -> None:
    original = dict(_MINIMAL_RECORD)
    patched = apply_verdict_to_record(
        _MINIMAL_RECORD,
        _GOOD_VERDICT,
        model_id="claude-test",
    )
    # Original is unchanged (deep copy)
    assert _MINIMAL_RECORD == original
    # Only provenance.critic changes
    assert patched["paper_id"] == _MINIMAL_RECORD["paper_id"]
    assert patched["provenance"]["critic"]["verdict"] == "accept"
    assert patched["provenance"]["critic"]["model_id"] == "claude-test"
    assert "Looks good" in patched["provenance"]["critic"]["notes"]


def test_apply_verdict_folds_structured_issues_into_notes() -> None:
    verdict = {
        "verdict": "revise",
        "factual_issues": [
            {"field": "evaluation.metrics", "issue": "Claims ROC-AUC but source shows accuracy."},
        ],
        "revision_instructions": "Re-read Section 4.",
    }
    patched = apply_verdict_to_record(_MINIMAL_RECORD, verdict, model_id="m")
    notes = patched["provenance"]["critic"]["notes"]
    assert "factual_issues" in notes
    assert "evaluation.metrics" in notes
    assert "revision_instructions" in notes
    assert "Re-read Section 4." in notes


def test_apply_verdict_handles_empty_extras() -> None:
    """A bare-minimum verdict with just {verdict: accept} should produce a
    valid patched record without spurious notes."""
    patched = apply_verdict_to_record(
        _MINIMAL_RECORD, {"verdict": "accept"}, model_id="m"
    )
    assert patched["provenance"]["critic"]["verdict"] == "accept"
    assert patched["provenance"]["critic"]["notes"] is None
