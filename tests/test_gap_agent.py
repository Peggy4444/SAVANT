"""Gap-agent runner tests (WP 2 — Step B6).

All tests use MockClient. The most interesting test exercises
`build_inputs_from_artifacts` against the real KG + enumeration + ranking
artifacts so we know the integration shape works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from savant.agents import AgentValidationError, MockClient
from savant.gap_analysis import (
    GapRunner,
    build_inputs_from_artifacts,
    enumerate_cells,
    flag_invalid_paper_ids,
    rank_gaps,
    save_llm_rationales,
)
from savant.knowledge_graph import build_kg, load_all_extractions


# --- Canned LLM response (matches prompts/gap_agent.md output schema) ------

_VALID_GAP = {
    "id": "gap-001",
    "title": "Per-action MDP value methods are absent from the corpus",
    "consolidated_auto_ranks": [
        {"framework": "markov-decision-process", "valuation_unit": "action",
         "action_type": "defensive_actions", "auto_rank": 1, "auto_score": 2.79},
    ],
    "the_gap": "No MDP framework method in the corpus produces a per-action value.",
    "scientific_motivation": "MDPs natively support Q(s,a) decomposition, but every MDP paper in the corpus (Rudd 2011, Van Roy 2021 etc.) values STATES.",
    "practitioner_motivation": "A per-action MDP would let coaches answer counterfactual decision questions with uncertainty bounds.",
    "closest_existing_work": ["decroos_2019_vaep", "vanroy_2021_markov_reward"],
    "suggested_first_study": "Reuse Van Roy 2021's StatsBomb-open MDP; add a per-action Q-head trained via policy iteration.",
    "novelty_score": 0.75,
    "feasibility_score": 0.85,
    "confidence": 0.8,
}

_VALID_RESPONSE = {
    "gaps": [_VALID_GAP],
    "agent_notes": "Consolidated the top six auto-ranked MDP × action cells.",
}


# --- Runner: rendering ----------------------------------------------------

def test_runner_passes_all_inputs_to_client() -> None:
    client = MockClient(responses=[_VALID_RESPONSE])
    runner = GapRunner(client)
    out = runner.run(
        coverage_summary={"corpus_size": 24},
        algorithmic_ranking=[{"rank": 1, "framework": "X"}],
        corpus_papers=[{"paper_id": "p1"}],
        dataset_facts={"n_public_datasets_used_in_corpus": 1},
    )
    assert out == _VALID_RESPONSE
    call = client.calls[0]
    assert call["agent"] == "gap"
    # Every input block must reach the user message
    for marker in ("Coverage summary", "Algorithmic ranking",
                    "Corpus papers", "Dataset facts"):
        assert marker in call["user"]
    assert "manuscript-quality" in call["system"].lower() or "gap" in call["system"].lower()


def test_runner_optionally_includes_open_questions() -> None:
    client = MockClient(responses=[_VALID_RESPONSE])
    GapRunner(client).run(
        coverage_summary={}, algorithmic_ranking=[], corpus_papers=[],
        dataset_facts={}, author_open_questions=["q1", "q2"],
    )
    user = client.calls[0]["user"]
    assert "Author open questions" in user
    assert "q1" in user


# --- Validation -----------------------------------------------------------

def test_validates_missing_gaps_list() -> None:
    client = MockClient(responses=[{"agent_notes": "nope"}])
    with pytest.raises(AgentValidationError, match="'gaps' list"):
        GapRunner(client).run(
            coverage_summary={}, algorithmic_ranking=[],
            corpus_papers=[], dataset_facts={},
        )


def test_validates_empty_gaps_list() -> None:
    client = MockClient(responses=[{"gaps": []}])
    with pytest.raises(AgentValidationError, match="empty 'gaps' list"):
        GapRunner(client).run(
            coverage_summary={}, algorithmic_ranking=[],
            corpus_papers=[], dataset_facts={},
        )


def test_validates_missing_required_gap_keys() -> None:
    incomplete = dict(_VALID_GAP)
    del incomplete["confidence"]
    client = MockClient(responses=[{"gaps": [incomplete]}])
    with pytest.raises(AgentValidationError, match="missing required keys"):
        GapRunner(client).run(
            coverage_summary={}, algorithmic_ranking=[],
            corpus_papers=[], dataset_facts={},
        )


def test_validates_score_in_unit_interval() -> None:
    bad = dict(_VALID_GAP); bad["novelty_score"] = 1.5
    client = MockClient(responses=[{"gaps": [bad]}])
    with pytest.raises(AgentValidationError, match="novelty_score"):
        GapRunner(client).run(
            coverage_summary={}, algorithmic_ranking=[],
            corpus_papers=[], dataset_facts={},
        )


def test_validates_closest_existing_work_is_list() -> None:
    bad = dict(_VALID_GAP); bad["closest_existing_work"] = "decroos_2019_vaep"
    client = MockClient(responses=[{"gaps": [bad]}])
    with pytest.raises(AgentValidationError, match="closest_existing_work"):
        GapRunner(client).run(
            coverage_summary={}, algorithmic_ranking=[],
            corpus_papers=[], dataset_facts={},
        )


# --- Integration: build_inputs_from_artifacts against the real corpus -----

@pytest.fixture(scope="module")
def real_artifacts():
    g = build_kg(load_all_extractions())
    enum = enumerate_cells(g)
    ranking = rank_gaps(enum)
    return g, enum, ranking, load_all_extractions()


def test_build_inputs_against_real_corpus(real_artifacts) -> None:
    g, enum, ranking, records = real_artifacts
    inputs = build_inputs_from_artifacts(g, enum, ranking,
                                          extraction_records=records, top_k=10)
    # Every required key present
    for key in ("coverage_summary", "algorithmic_ranking",
                 "corpus_papers", "dataset_facts", "author_open_questions"):
        assert key in inputs
    # Coverage summary captures the 24-paper corpus
    assert inputs["coverage_summary"]["corpus_size"] == 24
    fw_counts = inputs["coverage_summary"]["frameworks_with_method_counts"]
    assert "expectation-based" in fw_counts and fw_counts["expectation-based"] > 0
    # Top-K respected
    assert len(inputs["algorithmic_ranking"]) == 10
    # Corpus papers: 24 candidate papers visible (held-out stubs excluded)
    assert len(inputs["corpus_papers"]) == 24
    pids = {p["paper_id"] for p in inputs["corpus_papers"]}
    for canonical in ("decroos_2019_vaep", "singh_2018_xt", "fernandez_2019_epv"):
        assert canonical in pids
    # Dataset facts include both "distinct public datasets" and
    # "methods using a public dataset". With the current corpus the only
    # public-data use is Van Roy 2021 markov_reward (other-public / scraped),
    # giving 1 distinct public dataset and 1 method using it.
    facts = inputs["dataset_facts"]
    assert facts["n_corpus_methods"] == 24
    assert facts["n_public_datasets_used_in_corpus"] == 1
    assert facts["n_methods_using_public_dataset"] == 1
    # Roughly known from A4: only a handful of methods released code
    assert 0 <= facts["n_methods_with_code_released"] <= 24


# --- Persistence + paper_id cross-reference -------------------------------

def test_save_llm_rationales_writes_json(tmp_path: Path) -> None:
    path = save_llm_rationales(_VALID_RESPONSE, tmp_path / "out.json")
    assert path.exists()
    import json as _json
    loaded = _json.loads(path.read_text())
    assert len(loaded["gaps"]) == 1


def test_flag_invalid_paper_ids_finds_unknown_slugs() -> None:
    valid = {"decroos_2019_vaep", "singh_2018_xt"}
    response = {"gaps": [{
        "id": "gap-1",
        "closest_existing_work": ["decroos_2019_vaep", "made_up_2099_xyz"],
    }]}
    flagged = flag_invalid_paper_ids(response, valid)
    assert len(flagged) == 1
    assert flagged[0]["gap_id"] == "gap-1"
    assert flagged[0]["invalid_paper_ids"] == ["made_up_2099_xyz"]


def test_flag_invalid_paper_ids_accepts_sentinel() -> None:
    """'no exact match in corpus' is the allowed escape hatch per the prompt."""
    flagged = flag_invalid_paper_ids(
        {"gaps": [{"id": "g", "closest_existing_work": ["no exact match in corpus"]}]},
        valid_paper_ids={"decroos_2019_vaep"},
    )
    assert flagged == []
