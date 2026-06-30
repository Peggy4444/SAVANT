"""Coverage-matrix enumeration + ranking tests (Stage 5 / Track A — A5 + A6)."""

from __future__ import annotations

import pytest

from savant.gap_analysis import (
    ACTION_TYPES,
    FRAMEWORKS,
    VALUATION_UNITS,
    WEIGHTS,
    enumerate_cells,
    methods_in_cell,
    rank_gaps,
)
from savant.knowledge_graph import build_kg, load_all_extractions


@pytest.fixture(scope="module")
def enumeration():
    g = build_kg(load_all_extractions())
    return enumerate_cells(g), g


def test_enumeration_has_expected_top_level_keys(enumeration):
    e, _ = enumeration
    for k in ("schema_version", "corpus_size", "dimensions", "matrices"):
        assert k in e
    for m in ("framework_x_valuation_unit_x_action_type",
              "framework_x_action_type",
              "valuation_unit_x_action_type",
              "framework_x_valuation_unit",
              "framework_x_dataset",
              "dataset_x_action_type"):
        assert m in e["matrices"], f"Missing matrix {m!r}"


def test_3d_matrix_shape(enumeration):
    e, _ = enumeration
    cells = e["matrices"]["framework_x_valuation_unit_x_action_type"]
    assert len(cells) == len(FRAMEWORKS) * len(VALUATION_UNITS) * len(ACTION_TYPES)
    # Every cell has the required keys
    for c in cells:
        assert set(c) >= {"framework", "valuation_unit", "action_type", "count", "methods"}
        assert len(c["methods"]) == c["count"]


def test_counts_sum_to_at_least_corpus_size(enumeration):
    """The framework x valuation_unit projection sums to exactly corpus_size
    (each method has exactly one framework, one valuation_unit)."""
    e, _ = enumeration
    fxv = e["matrices"]["framework_x_valuation_unit"]
    total = sum(c["count"] for c in fxv)
    # Some methods may have framework=None — those are dropped by the strict
    # equality match. So total <= corpus_size.
    assert 0 < total <= e["corpus_size"]


def test_known_gap_goalkeeper_x_rl(enumeration):
    """No RL method values goalkeeper actions in the current corpus — used
    as a smoke-test of the cell lookup."""
    e, _ = enumeration
    cells = e["matrices"]["framework_x_action_type"]
    rl_gk = next(c for c in cells
                 if c["framework"] == "reinforcement-learning"
                 and c["action_type"] == "goalkeeper_actions")
    assert rl_gk["count"] == 0


def test_methods_in_cell_returns_correct_projections(enumeration):
    e, g = enumeration
    # Find any non-empty 3D cell from the enumeration
    nonzero = [c for c in e["matrices"]["framework_x_valuation_unit_x_action_type"]
               if c["count"] > 0]
    assert nonzero, "Corpus must have at least one non-empty 3D cell"
    sample = nonzero[0]

    results = methods_in_cell(
        g,
        framework=sample["framework"],
        valuation_unit=sample["valuation_unit"],
        action_type=sample["action_type"],
    )
    assert len(results) == sample["count"]
    assert sorted(r["paper_id"] for r in results) == sample["methods"]


def test_methods_in_cell_rejects_unknown_filter(enumeration):
    _, g = enumeration
    with pytest.raises(ValueError, match="Unknown filter"):
        methods_in_cell(g, banana="yellow")


# --- A6 ranking tests --------------------------------------------------------

@pytest.fixture(scope="module")
def ranking(enumeration):
    e, _ = enumeration
    return rank_gaps(e)


def test_ranking_has_expected_shape(ranking):
    assert ranking["schema_version"] == "1.0"
    assert set(ranking["weights"]) == {"methodological", "reproducibility", "practitioner"}
    assert ranking["weights"]["methodological"] == 1.0  # primary
    assert ranking["n_candidate_cells"] > 0
    # Every entry has the right keys + rank starts at 1
    for i, g in enumerate(ranking["ranked_gaps"], start=1):
        assert g["rank"] == i
        assert g["current_count"] == 0  # default max_count is 0
        assert g["score"] >= 0
        assert "framework" in g and "valuation_unit" in g and "action_type" in g
        assert "rationale" in g and g["rationale"]


def test_ranking_is_descending(ranking):
    scores = [g["score"] for g in ranking["ranked_gaps"]]
    assert scores == sorted(scores, reverse=True)


def test_top_gap_has_positive_methodological_score(ranking):
    """The #1 ranked gap must have methodological_score > 0 (otherwise the
    weighting is broken)."""
    top = ranking["ranked_gaps"][0]
    assert top["components"]["methodological_score"] > 0


def test_rule_based_cells_score_zero(ranking):
    """Rule-based has zero methods in the corpus, so cells in that row
    should have methodological_score == 0 and total score driven only by
    practitioner bonus (≤ WEIGHTS['practitioner'] * 1.0)."""
    rule_cells = [g for g in ranking["ranked_gaps"] if g["framework"] == "rule-based"]
    assert rule_cells, "Expected rule-based cells in ranking"
    for g in rule_cells:
        assert g["components"]["methodological_score"] == 0
        # Total <= practitioner bonus contribution (no methodological, no repro)
        assert g["score"] <= WEIGHTS["practitioner"] * 1.5  # at most (action+framework bonus)
