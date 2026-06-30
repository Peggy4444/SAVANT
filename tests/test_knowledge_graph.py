"""Knowledge graph builder tests (Stage 4 / Track A — Step A3)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from savant.knowledge_graph import (
    EDGE_TYPES,
    KG_SCHEMA_VERSION,
    NODE_TYPES,
    build_kg,
    load_all_extractions,
    save_kg,
)


@pytest.fixture(scope="module")
def kg() -> nx.MultiDiGraph:
    return build_kg(load_all_extractions())


def test_kg_builds_and_has_schema_version(kg: nx.MultiDiGraph) -> None:
    assert kg.graph["kg_schema_version"] == KG_SCHEMA_VERSION
    assert kg.number_of_nodes() > 0
    assert kg.number_of_edges() > 0


def test_kg_has_expected_node_types(kg: nx.MultiDiGraph) -> None:
    seen = {data.get("node_type") for _, data in kg.nodes(data=True)}
    for nt in NODE_TYPES:
        assert nt in seen, f"Node type {nt!r} missing from KG"


def test_kg_has_expected_edge_types(kg: nx.MultiDiGraph) -> None:
    seen = {data.get("edge_type") for _, _, data in kg.edges(data=True)}
    for et in (
        "AUTHORED_BY", "PUBLISHED_IN", "PROPOSES",
        "HAS_FRAMEWORK", "HAS_VALUATION_UNIT", "VALUES",
        "USES_DATASET", "EVALUATED_ON_TASK", "SUITABLE_FOR",
    ):
        assert et in seen, f"Edge type {et!r} missing from KG"
    p2p_seen = {e for e in seen if e in ("EXTENDS", "IMPROVES_UPON",
                                          "COMPARES_AGAINST", "USES_DATASET_OF",
                                          "CRITIQUES")}
    assert len(p2p_seen) >= 3, f"Paper-to-paper edges underrepresented: {p2p_seen}"


def test_one_method_per_candidate_paper(kg: nx.MultiDiGraph) -> None:
    paper_nodes = [n for n, d in kg.nodes(data=True)
                   if d.get("node_type") == "paper" and not d.get("_stub")]
    method_nodes = [n for n, d in kg.nodes(data=True) if d.get("node_type") == "method"]
    assert len(paper_nodes) == 24
    assert len(method_nodes) == 24


def test_every_method_has_required_practitioner_attributes(kg: nx.MultiDiGraph) -> None:
    for n, d in kg.nodes(data=True):
        if d.get("node_type") != "method":
            continue
        for required in ("valuation_unit", "atomicity", "interpretability_level"):
            assert required in d, f"Method {n} missing {required}"
        assert isinstance(d["code_released"], bool)


def test_action_coverage_is_consistent_with_values_edges(kg: nx.MultiDiGraph) -> None:
    method_to_action_types: dict[str, set[str]] = {}
    for u, v, d in kg.edges(data=True):
        if d.get("edge_type") != "VALUES":
            continue
        method_to_action_types.setdefault(u, set()).add(kg.nodes[v]["name"])
    method_nodes = [n for n, d in kg.nodes(data=True) if d.get("node_type") == "method"]
    for m in method_nodes:
        assert m in method_to_action_types, f"Method {m} has no VALUES edges"


def test_paper_to_paper_edges_resolve(kg: nx.MultiDiGraph) -> None:
    p2p = {"EXTENDS", "IMPROVES_UPON", "COMPARES_AGAINST",
           "USES_DATASET_OF", "CRITIQUES"}
    for u, v, d in kg.edges(data=True):
        if d.get("edge_type") not in p2p:
            continue
        assert kg.nodes[u].get("node_type") == "paper", f"Source {u} not a paper"
        assert kg.nodes[v].get("node_type") == "paper", f"Target {v} not a paper"


def test_build_is_deterministic() -> None:
    g1 = build_kg(load_all_extractions())
    g2 = build_kg(load_all_extractions())
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()
    assert set(g1.nodes()) == set(g2.nodes())


def test_save_kg_writes_both_formats(tmp_path: Path, kg: nx.MultiDiGraph) -> None:
    paths = save_kg(kg, tmp_path)
    assert paths["json"].exists() and paths["json"].stat().st_size > 0
    assert paths["graphml"].exists() and paths["graphml"].stat().st_size > 0
    import json as _json
    data = _json.loads(paths["json"].read_text())
    assert len(data["nodes"]) == kg.number_of_nodes()
