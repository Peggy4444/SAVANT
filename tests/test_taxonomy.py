"""Taxonomy-agent runner tests (WP 2 — Step B5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from savant.agents import AgentValidationError, MockClient
from savant.taxonomy import (
    TaxonomyRunner,
    apply_taxonomy_to_record,
    collect_examples,
    collect_raw_values,
    save_taxonomy,
)


_SAMPLE_RECORDS = [
    {
        "paper_id": "decroos_2019_vaep",
        "evaluation": {"metrics": ["ROC-AUC", "Brier score"]},
        "data": {"provider": "Wyscout"},
    },
    {
        "paper_id": "singh_2018_xt",
        "evaluation": {"metrics": []},
        "data": {"provider": "Opta/blog-undisclosed"},
    },
    {
        "paper_id": "brooks_2016_passrank",
        "evaluation": {"metrics": ["AUROC", "F1"]},
        "data": {"provider": "Opta"},
    },
]


_VALID_MAPPING = {
    "field": "evaluation.metrics",
    "mappings": [
        {"raw": "ROC-AUC",     "canonical": "roc_auc",     "confidence": 1.0},
        {"raw": "AUROC",       "canonical": "roc_auc",     "confidence": 1.0},
        {"raw": "Brier score", "canonical": "brier_score", "confidence": 1.0},
        {"raw": "F1",          "canonical": "f1",          "confidence": 0.95},
    ],
    "new_vocabulary": [],
    "unresolved": [],
}


# --- Runner: rendering + plumbing ------------------------------------------

def test_runner_passes_field_and_values_to_client() -> None:
    client = MockClient(responses=[_VALID_MAPPING])
    runner = TaxonomyRunner(client)
    out = runner.run(
        field="evaluation.metrics",
        values=["ROC-AUC", "AUROC", "Brier score", "F1"],
        controlled_vocabulary=[],
        examples=[],
    )
    assert out == _VALID_MAPPING
    call = client.calls[0]
    assert call["agent"] == "taxonomy"
    assert "evaluation.metrics" in call["user"]
    for v in ("ROC-AUC", "Brier score", "F1"):
        assert v in call["user"]


def test_runner_dedupes_values_before_sending() -> None:
    client = MockClient(responses=[_VALID_MAPPING])
    runner = TaxonomyRunner(client)
    runner.run(
        field="evaluation.metrics",
        values=["ROC-AUC", "ROC-AUC", "AUROC"],  # duplicate
        controlled_vocabulary=[],
        examples=[],
    )
    user_msg = client.calls[0]["user"]
    assert user_msg.count('"ROC-AUC"') == 1  # serialized once after dedup


# --- Validation ------------------------------------------------------------

def test_validates_missing_field() -> None:
    client = MockClient(responses=[{"mappings": []}])
    with pytest.raises(AgentValidationError, match="'field'"):
        TaxonomyRunner(client).run(
            field="x", values=[], controlled_vocabulary=[], examples=[]
        )


def test_validates_mappings_must_be_list_of_dicts() -> None:
    client = MockClient(responses=[{"field": "x", "mappings": "not a list"}])
    with pytest.raises(AgentValidationError, match="mappings"):
        TaxonomyRunner(client).run(
            field="x", values=[], controlled_vocabulary=[], examples=[]
        )


def test_validates_each_mapping_has_required_keys() -> None:
    client = MockClient(responses=[{
        "field": "x",
        "mappings": [{"raw": "Foo"}],  # missing 'canonical'
    }])
    with pytest.raises(AgentValidationError, match="canonical"):
        TaxonomyRunner(client).run(
            field="x", values=[], controlled_vocabulary=[], examples=[]
        )


def test_validates_canonical_label_style() -> None:
    """Canonical labels must be ASCII alphanumeric + _/- only."""
    client = MockClient(responses=[{
        "field": "x",
        "mappings": [{"raw": "Foo", "canonical": "has spaces and !chars"}],
    }])
    with pytest.raises(AgentValidationError, match="ASCII"):
        TaxonomyRunner(client).run(
            field="x", values=[], controlled_vocabulary=[], examples=[]
        )


# --- collect_raw_values / collect_examples ---------------------------------

def test_collect_raw_values_flattens_lists_and_dedupes() -> None:
    out = collect_raw_values("evaluation.metrics", _SAMPLE_RECORDS)
    # ROC-AUC, Brier score, AUROC, F1 — order of first sight
    assert out == ["ROC-AUC", "Brier score", "AUROC", "F1"]


def test_collect_raw_values_handles_scalar_strings() -> None:
    out = collect_raw_values("data.provider", _SAMPLE_RECORDS)
    assert set(out) == {"Wyscout", "Opta/blog-undisclosed", "Opta"}


def test_collect_raw_values_ignores_missing_paths() -> None:
    out = collect_raw_values("nonexistent.path", _SAMPLE_RECORDS)
    assert out == []


def test_collect_examples_returns_paper_id_and_raw_pairs() -> None:
    out = collect_examples("data.provider", _SAMPLE_RECORDS, limit=2)
    assert len(out) == 2
    assert all("paper_id" in e and "raw" in e for e in out)


# --- apply_taxonomy_to_record ----------------------------------------------

def test_apply_to_record_writes_canonical_sibling_for_list() -> None:
    mappings = {"ROC-AUC": "roc_auc", "Brier score": "brier_score"}
    record = {"paper_id": "x", "evaluation": {"metrics": ["ROC-AUC", "Brier score", "unmapped"]}}
    patched = apply_taxonomy_to_record(record, "evaluation.metrics", mappings)
    # Original untouched
    assert record["evaluation"]["metrics"] == ["ROC-AUC", "Brier score", "unmapped"]
    # Canonical sibling created; unmapped values dropped per design
    assert patched["evaluation"]["metrics_canonical"] == ["roc_auc", "brier_score"]


def test_apply_to_record_writes_canonical_sibling_for_scalar() -> None:
    mappings = {"Wyscout": "wyscout"}
    record = {"data": {"provider": "Wyscout"}}
    patched = apply_taxonomy_to_record(record, "data.provider", mappings)
    assert patched["data"]["provider"] == "Wyscout"
    assert patched["data"]["provider_canonical"] == "wyscout"


def test_apply_to_record_no_op_on_missing_path() -> None:
    record = {"paper_id": "x"}
    patched = apply_taxonomy_to_record(record, "evaluation.metrics", {"a": "b"})
    assert patched == record


# --- save_taxonomy ---------------------------------------------------------

def test_save_taxonomy_writes_field_slug_json(tmp_path: Path) -> None:
    path = save_taxonomy(_VALID_MAPPING, tmp_path)
    assert path.name == "evaluation_metrics.json"
    import json as _json
    loaded = _json.loads(path.read_text())
    assert loaded["field"] == "evaluation.metrics"
    assert len(loaded["mappings"]) == 4
