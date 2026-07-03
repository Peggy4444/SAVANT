"""BM-16 KG empirical-enrichment tests. Synthetic bench dirs, no network."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from savant.knowledge_graph.empirical import (
    _wrapper_to_corpus_paper_id, attach_benchmark_results,
    summarise_enrichment,
)


# --- Fixture: minimal KG stub with one Method node per corpus paper --


@pytest.fixture
def stub_kg() -> nx.MultiDiGraph:
    """Two papers with one Method node each. Mirrors the real KG schema
    but skips the surrounding Author / Venue / Framework nodes."""
    g = nx.MultiDiGraph()
    # singh_2018_xt paper + method
    g.add_node("paper:singh_2018_xt", node_type="paper", paper_id="singh_2018_xt")
    g.add_node("method:singh_2018_xt::xt_singh",
               node_type="method", name="xT (Singh)")
    g.add_edge("paper:singh_2018_xt",
               "method:singh_2018_xt::xt_singh",
               key="PROPOSES", edge_type="PROPOSES")
    # decroos_2019_vaep paper + method
    g.add_node("paper:decroos_2019_vaep",
               node_type="paper", paper_id="decroos_2019_vaep")
    g.add_node("method:decroos_2019_vaep::vaep",
               node_type="method", name="VAEP")
    g.add_edge("paper:decroos_2019_vaep",
               "method:decroos_2019_vaep::vaep",
               key="PROPOSES", edge_type="PROPOSES")
    return g


@pytest.fixture
def synthetic_bench_dir(tmp_path: Path) -> Path:
    """Minimal per-task CSVs matching the real evaluator's schema."""
    d = tmp_path / "bench"
    (d / "per_task").mkdir(parents=True)

    pd.DataFrame([
        {"model_id": "xt_state_singh",
         "pearson_rho_point": 0.52, "pearson_rho_lo": 0.48, "pearson_rho_hi": 0.57,
         "spearman_rho_point": 0.51, "spearman_rho_lo": 0.45, "spearman_rho_hi": 0.55,
         "mae_point": 1.00, "mae_lo": 0.9, "mae_hi": 1.1},
        {"model_id": "xt_action_sumpter",
         "pearson_rho_point": 0.41, "pearson_rho_lo": 0.36, "pearson_rho_hi": 0.47,
         "spearman_rho_point": 0.41, "spearman_rho_lo": 0.35, "spearman_rho_hi": 0.47,
         "mae_point": 54.3, "mae_lo": 48, "mae_hi": 58},
        {"model_id": "vaep_socceraction",
         "pearson_rho_point": 0.45, "pearson_rho_lo": 0.41, "pearson_rho_hi": 0.50,
         "spearman_rho_point": 0.44, "spearman_rho_lo": 0.4, "spearman_rho_hi": 0.5,
         "mae_point": 1.60, "mae_lo": 1.4, "mae_hi": 1.8},
    ]).to_csv(d / "per_task" / "t3_pearson_bootstrap.csv", index=False)

    pd.DataFrame([
        {"model_id": "vaep_socceraction",
         "brier_point": 0.010, "brier_lo": 0.009, "brier_hi": 0.011,
         "log_loss_binary_point": 0.054, "log_loss_binary_lo": 0.05, "log_loss_binary_hi": 0.06,
         "ece_point": 0.0004, "ece_lo": 0.0003, "ece_hi": 0.0005,
         "roc_auc_point": 0.643, "roc_auc_lo": 0.6, "roc_auc_hi": 0.69},
    ]).to_csv(d / "per_task" / "t1_brier_bootstrap.csv", index=False)

    pd.DataFrame([
        {"model_id": "vaep_socceraction",
         "log_loss_multiclass_point": 1.35,
         "log_loss_multiclass_lo": 1.35, "log_loss_multiclass_hi": 1.36,
         "top1_accuracy_point": 0.34,
         "top1_accuracy_lo": 0.33, "top1_accuracy_hi": 0.35,
         "ece_multiclass_point": 0.11,
         "ece_multiclass_lo": 0.10, "ece_multiclass_hi": 0.12},
    ]).to_csv(d / "per_task" / "t2_multiclass_bootstrap.csv", index=False)

    pd.DataFrame([
        {"model_id": "vaep_socceraction",
         "pearson_rho_delta": -0.092,
         "spearman_rho_delta": -0.09,
         "mae_delta": -0.05},
    ]).to_csv(d / "per_task" / "t4_delta.csv", index=False)

    # Manifests aren't required — the enricher tolerates their absence.
    return d


# --- Wrapper → paper_id mapping ------------------------------------


def test_wrapper_to_corpus_paper_id_covers_all_registered_methods():
    mapping = _wrapper_to_corpus_paper_id()
    # Sanity: every wrapper is in METHODS_BY_ID and maps to a paper_id.
    from savant.benchmarking import METHODS_BY_ID
    assert set(mapping.keys()) == set(METHODS_BY_ID.keys())
    for wid, pid in mapping.items():
        assert isinstance(pid, str) and pid, f"empty paper_id for {wid}"


def test_singh_xt_variants_share_a_paper_id():
    """The two xT wrappers test the model.valuation_unit axis and MUST
    map to the same corpus paper — otherwise our multi-variant handling
    is untested."""
    mapping = _wrapper_to_corpus_paper_id()
    assert mapping["xt_state_singh"] == mapping["xt_action_sumpter"]


# --- Enrichment core -----------------------------------------------


def test_attach_benchmark_results_populates_variants(stub_kg, synthetic_bench_dir):
    attach_benchmark_results(stub_kg, synthetic_bench_dir)

    xt_node = stub_kg.nodes["method:singh_2018_xt::xt_singh"]
    assert "empirical_variants" in xt_node
    variants = xt_node["empirical_variants"]
    assert len(variants) == 2
    vids = {v["variant_id"] for v in variants}
    assert vids == {"xt_state_singh", "xt_action_sumpter"}

    vaep_node = stub_kg.nodes["method:decroos_2019_vaep::vaep"]
    assert "empirical_variants" in vaep_node
    assert len(vaep_node["empirical_variants"]) == 1
    assert vaep_node["empirical_variants"][0]["variant_id"] == "vaep_socceraction"


def test_all_four_tasks_land_on_the_node(stub_kg, synthetic_bench_dir):
    """VAEP row must carry T1 + T2 + T3 + T4 fields."""
    attach_benchmark_results(stub_kg, synthetic_bench_dir)
    row = stub_kg.nodes["method:decroos_2019_vaep::vaep"]["empirical_variants"][0]
    for k in ("t3_pearson_rho", "t3_pearson_rho_lo", "t3_pearson_rho_hi",
              "t3_spearman_rho", "t3_mae",
              "t1_brier", "t1_log_loss", "t1_ece", "t1_roc_auc",
              "t2_log_loss_multiclass", "t2_top1_accuracy", "t2_ece_multiclass",
              "t4_delta_pearson_rho"):
        assert k in row, f"missing metric {k!r}"


def test_attach_is_noop_when_bench_dir_empty(stub_kg, tmp_path):
    empty = tmp_path / "empty"
    (empty / "per_task").mkdir(parents=True)
    attach_benchmark_results(stub_kg, empty)
    for _, d in stub_kg.nodes(data=True):
        if d.get("node_type") == "method":
            assert "empirical_variants" not in d


def test_method_node_without_benchmark_row_is_untouched(stub_kg, tmp_path):
    """Adding an extra Method node whose paper has no benchmark rows —
    it must NOT gain empirical_variants."""
    bench = tmp_path / "bench2"
    (bench / "per_task").mkdir(parents=True)
    pd.DataFrame([{
        "model_id": "vaep_socceraction",
        "pearson_rho_point": 0.4, "pearson_rho_lo": 0.3, "pearson_rho_hi": 0.5,
        "spearman_rho_point": 0.4, "spearman_rho_lo": 0.3, "spearman_rho_hi": 0.5,
        "mae_point": 1.0, "mae_lo": 0.9, "mae_hi": 1.1,
    }]).to_csv(bench / "per_task" / "t3_pearson_bootstrap.csv", index=False)

    # Untouched paper.
    stub_kg.add_node("paper:some_other_paper", node_type="paper",
                     paper_id="some_other_paper")
    stub_kg.add_node("method:some_other_paper::x",
                     node_type="method", name="Other")
    stub_kg.add_edge("paper:some_other_paper",
                     "method:some_other_paper::x",
                     key="PROPOSES", edge_type="PROPOSES")

    attach_benchmark_results(stub_kg, bench)
    assert "empirical_variants" not in stub_kg.nodes["method:some_other_paper::x"]


# --- Summary + edge cases ------------------------------------------


def test_summarise_enrichment_counts_correctly(stub_kg, synthetic_bench_dir):
    attach_benchmark_results(stub_kg, synthetic_bench_dir)
    summary = summarise_enrichment(stub_kg)
    assert summary["n_methods_total"] == 2
    assert summary["n_methods_enriched"] == 2
    assert summary["coverage_pct"] == pytest.approx(100.0)
    assert "singh_2018_xt" in summary["variants_by_paper"]
    assert set(summary["variants_by_paper"]["singh_2018_xt"]) == {
        "xt_state_singh", "xt_action_sumpter",
    }


def test_summarise_handles_partial_enrichment(stub_kg, tmp_path):
    """Only VAEP row present → coverage = 1/2 = 50%."""
    bench = tmp_path / "partial"
    (bench / "per_task").mkdir(parents=True)
    pd.DataFrame([{
        "model_id": "vaep_socceraction",
        "pearson_rho_point": 0.4, "pearson_rho_lo": 0.3, "pearson_rho_hi": 0.5,
        "spearman_rho_point": 0.4, "spearman_rho_lo": 0.3, "spearman_rho_hi": 0.5,
        "mae_point": 1.0, "mae_lo": 0.9, "mae_hi": 1.1,
    }]).to_csv(bench / "per_task" / "t3_pearson_bootstrap.csv", index=False)

    attach_benchmark_results(stub_kg, bench)
    summary = summarise_enrichment(stub_kg)
    assert summary["n_methods_enriched"] == 1
    assert summary["coverage_pct"] == pytest.approx(50.0)


# --- Gap-agent input surfaces empirical results -------------------


def test_gap_agent_input_includes_empirical_when_kg_enriched(
    stub_kg, synthetic_bench_dir,
):
    """After enrichment, `build_inputs_from_artifacts` must expose
    `empirical_results` so the gap-agent prompt sees measured metrics."""
    from savant.gap_analysis import build_inputs_from_artifacts

    attach_benchmark_results(stub_kg, synthetic_bench_dir)

    # Minimal enumeration + ranking to satisfy the builder.
    enumeration = {
        "corpus_size": 2,
        "matrices": {"framework_x_valuation_unit_x_action_type": []},
        "dimensions": {
            "datasets_in_corpus": [],
            "datasets_public": [],
        },
    }
    ranking = {"ranked_gaps": []}

    inputs = build_inputs_from_artifacts(stub_kg, enumeration, ranking)
    assert "empirical_results" in inputs
    empirical = inputs["empirical_results"]
    assert set(empirical.keys()) == {"singh_2018_xt", "decroos_2019_vaep"}
    assert len(empirical["singh_2018_xt"]["variants"]) == 2
    assert empirical["decroos_2019_vaep"]["variants"][0]["variant_id"] == "vaep_socceraction"


def test_gap_agent_input_empirical_is_empty_when_kg_not_enriched(stub_kg):
    """No enrichment → the field is an empty dict, not missing."""
    from savant.gap_analysis import build_inputs_from_artifacts

    enumeration = {
        "corpus_size": 2,
        "matrices": {"framework_x_valuation_unit_x_action_type": []},
        "dimensions": {
            "datasets_in_corpus": [],
            "datasets_public": [],
        },
    }
    ranking = {"ranked_gaps": []}
    inputs = build_inputs_from_artifacts(stub_kg, enumeration, ranking)
    assert "empirical_results" in inputs
    assert inputs["empirical_results"] == {}


# --- Serialisation preserves structure (GraphML round-trip) -------


def test_save_kg_survives_enriched_variants(stub_kg, synthetic_bench_dir, tmp_path):
    """JSON keeps the structured list; GraphML JSON-encodes on write.
    Both artefacts must be produceable without exceptions."""
    from savant.knowledge_graph import save_kg
    attach_benchmark_results(stub_kg, synthetic_bench_dir)
    paths = save_kg(stub_kg, tmp_path)
    assert paths["json"].exists()
    assert paths["graphml"].exists()

    # JSON round-trip: variants preserved as a list of dicts.
    data = json.loads(paths["json"].read_text())
    vaep_data = next(n for n in data["nodes"]
                     if n["id"] == "method:decroos_2019_vaep::vaep")
    assert isinstance(vaep_data["empirical_variants"], list)
    assert vaep_data["empirical_variants"][0]["variant_id"] == "vaep_socceraction"

    # GraphML: variants land as a JSON-encoded string (schema requirement).
    graphml_bytes = paths["graphml"].read_bytes()
    assert b"empirical_variants" in graphml_bytes
    assert b"vaep_socceraction" in graphml_bytes
