"""Build a NetworkX MultiDiGraph from the 24 candidate extraction JSON records.

Implements the schema in `src/savant/knowledge_graph/README.md`. The output
is deterministic given the same input JSONs: node ids are derived from
content, not generated.

Public API:
    load_all_extractions(dir) -> list[dict]
    build_kg(records) -> MultiDiGraph
    save_kg(graph, out_dir) -> {"json": Path, "graphml": Path}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from ..utils.config import repo_root

# NetworkX 3.1's GraphML writer references the removed `np.float_` alias.
# Shim it back to `np.float64` so writing succeeds on NumPy >= 2.0.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

KG_SCHEMA_VERSION = "1.0"

NODE_TYPES: tuple[str, ...] = (
    "paper",
    "method",
    "author",
    "venue",
    "framework",
    "valuation_unit",
    "action_type",
    "dataset",
    "use_case",
    "evaluation_task",
)

EDGE_TYPES: tuple[str, ...] = (
    # bibliographic
    "AUTHORED_BY",
    "PUBLISHED_IN",
    # method-structural
    "PROPOSES",
    "HAS_FRAMEWORK",
    "HAS_VALUATION_UNIT",
    "VALUES",
    "USES_DATASET",
    "EVALUATED_ON_TASK",
    "SUITABLE_FOR",
    # paper-to-paper lineage
    "EXTENDS",
    "IMPROVES_UPON",
    "COMPARES_AGAINST",
    "USES_DATASET_OF",
    "CRITIQUES",
)

# Public-data datasets (used to set Dataset.is_public attribute).
_PUBLIC_DATASETS: frozenset[str] = frozenset({
    "statsbomb-open",
    "statsbomb-360-open",
    "wyscout-public-2017-18",
    "skillcorner-public",
    "metrica-sports-public",
    "other-public",
})


# --- Node-id helpers (stable, content-derived) --------------------------------

def _paper_id(paper_id: str) -> str:
    return f"paper:{paper_id}"


def _method_id(paper_id: str, method_name: str) -> str:
    return f"method:{paper_id}::{method_name}"


def _author_id(name: str) -> str:
    return f"author:{name.strip()}"


def _venue_id(name: str) -> str:
    return f"venue:{name.strip()}"


def _framework_id(name: str) -> str:
    return f"framework:{name}"


def _valuation_unit_id(name: str) -> str:
    return f"valuation_unit:{name}"


def _action_type_id(name: str) -> str:
    return f"action_type:{name}"


def _dataset_id(name: str) -> str:
    return f"dataset:{name}"


def _use_case_id(name: str) -> str:
    return f"use_case:{name}"


def _eval_task_id(name: str) -> str:
    return f"evaluation_task:{name}"


# --- Loader -------------------------------------------------------------------

def load_all_extractions(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Load every extraction JSON in `directory` (default:
    repo_root/data/outputs/extractions/), sorted by paper_id for determinism."""
    if directory is None:
        directory = repo_root() / "data" / "outputs" / "extractions"
    d = Path(directory)
    records: list[dict[str, Any]] = []
    for p in sorted(d.glob("*.json")):
        records.append(json.loads(p.read_text()))
    return records


# --- Builder ------------------------------------------------------------------

def _add_paper_and_method(g: nx.MultiDiGraph, rec: dict[str, Any]) -> tuple[str, str]:
    pid = rec["paper_id"]
    bib = rec["bibliographic"]
    model = rec["model"]
    practical = rec["practical"]
    code = rec.get("code_release", {}) or {}

    paper_node = _paper_id(pid)
    g.add_node(
        paper_node,
        node_type="paper",
        paper_id=pid,
        title=bib.get("title"),
        year=bib.get("year"),
        venue_type=bib.get("venue_type"),
        doi=bib.get("doi"),
        url=bib.get("url"),
        open_access=bib.get("open_access"),
    )

    method_node = _method_id(pid, model["name"])
    g.add_node(
        method_node,
        node_type="method",
        method_id=method_node.removeprefix("method:"),
        name=model["name"],
        family=model.get("family"),
        framework=model.get("framework"),
        valuation_unit=model["valuation_unit"],
        atomicity=model["atomicity"],
        supervision=model.get("supervision"),
        value_scale=model.get("outputs", {}).get("value_scale"),
        inference_realtime_capable=practical.get("inference_realtime_capable"),
        interpretability_level=practical.get("interpretability_level"),
        compute_intensity=practical.get("compute_intensity"),
        code_released=bool(code.get("released")),
        code_maintenance_status=code.get("maintenance_status"),
    )
    g.add_edge(paper_node, method_node, key="PROPOSES", edge_type="PROPOSES")
    return paper_node, method_node


def _add_authors(g: nx.MultiDiGraph, paper_node: str, authors: list[str]) -> None:
    for i, name in enumerate(authors or []):
        if not name:
            continue
        an = _author_id(name)
        if an not in g:
            g.add_node(an, node_type="author", name=name.strip())
        g.add_edge(paper_node, an, key="AUTHORED_BY", edge_type="AUTHORED_BY", position=i)


def _add_venue(g: nx.MultiDiGraph, paper_node: str, bib: dict[str, Any]) -> None:
    name = bib.get("venue")
    if not name:
        return
    vn = _venue_id(name)
    if vn not in g:
        g.add_node(vn, node_type="venue", name=name, venue_type=bib.get("venue_type"))
    g.add_edge(paper_node, vn, key="PUBLISHED_IN", edge_type="PUBLISHED_IN")


def _add_framework(g: nx.MultiDiGraph, method_node: str, framework: str | None) -> None:
    if not framework:
        return
    fn = _framework_id(framework)
    if fn not in g:
        g.add_node(fn, node_type="framework", name=framework)
    g.add_edge(method_node, fn, key="HAS_FRAMEWORK", edge_type="HAS_FRAMEWORK")


def _add_valuation_unit(g: nx.MultiDiGraph, method_node: str, vu: str) -> None:
    n = _valuation_unit_id(vu)
    if n not in g:
        g.add_node(n, node_type="valuation_unit", name=vu)
    g.add_edge(method_node, n, key="HAS_VALUATION_UNIT", edge_type="HAS_VALUATION_UNIT")


def _add_action_coverage(g: nx.MultiDiGraph, method_node: str, action_coverage: dict[str, bool]) -> None:
    for action_type, covered in (action_coverage or {}).items():
        if not covered:
            continue
        n = _action_type_id(action_type)
        if n not in g:
            g.add_node(n, node_type="action_type", name=action_type)
        g.add_edge(method_node, n, key="VALUES", edge_type="VALUES")


def _add_datasets(g: nx.MultiDiGraph, method_node: str, datasets: list[str]) -> None:
    for ds in datasets or []:
        if not ds:
            continue
        n = _dataset_id(ds)
        if n not in g:
            g.add_node(
                n,
                node_type="dataset",
                name=ds,
                is_public=(ds in _PUBLIC_DATASETS),
            )
        g.add_edge(method_node, n, key="USES_DATASET", edge_type="USES_DATASET")


def _add_eval_tasks(g: nx.MultiDiGraph, method_node: str, tasks: list[str]) -> None:
    for t in tasks or []:
        if not t:
            continue
        n = _eval_task_id(t)
        if n not in g:
            g.add_node(n, node_type="evaluation_task", name=t)
        g.add_edge(method_node, n, key="EVALUATED_ON_TASK", edge_type="EVALUATED_ON_TASK")


def _add_use_cases(g: nx.MultiDiGraph, method_node: str, use_cases: list[str]) -> None:
    for uc in use_cases or []:
        if not uc:
            continue
        n = _use_case_id(uc)
        if n not in g:
            g.add_node(n, node_type="use_case", name=uc)
        g.add_edge(method_node, n, key="SUITABLE_FOR", edge_type="SUITABLE_FOR")


def _add_paper_relationships(g: nx.MultiDiGraph, paper_node: str, relationships: list[dict]) -> None:
    for rel in relationships or []:
        rtype = rel.get("type")
        target_pid = rel.get("target_paper_id")
        if not rtype or not target_pid:
            continue
        # Target paper node may not exist yet (held-out) — add a stub
        target_node = _paper_id(target_pid)
        if target_node not in g:
            g.add_node(target_node, node_type="paper", paper_id=target_pid, _stub=True)
        rkey = rtype.replace("-", "_").upper()  # 'compares-against' -> 'COMPARES_AGAINST'
        if rkey not in EDGE_TYPES:
            raise ValueError(f"Unknown relationship type {rtype!r} (mapped to {rkey!r})")
        g.add_edge(paper_node, target_node, key=rkey, edge_type=rkey)


def build_kg(records: Iterable[dict[str, Any]]) -> nx.MultiDiGraph:
    """Build the full SAVANT knowledge graph from a list of extraction JSONs.

    Returns a NetworkX MultiDiGraph. The graph carries a graph-level attribute
    `kg_schema_version` that downstream consumers (gap_analysis, visualization)
    can check.
    """
    g = nx.MultiDiGraph()
    g.graph["kg_schema_version"] = KG_SCHEMA_VERSION

    records = list(records)
    for rec in records:
        paper_node, method_node = _add_paper_and_method(g, rec)
        _add_authors(g, paper_node, rec["bibliographic"].get("authors") or [])
        _add_venue(g, paper_node, rec["bibliographic"])
        _add_framework(g, method_node, rec["model"].get("framework"))
        _add_valuation_unit(g, method_node, rec["model"]["valuation_unit"])
        _add_action_coverage(g, method_node, rec["model"].get("action_coverage", {}))
        _add_datasets(g, method_node, rec["data"].get("dataset_canonical") or [])
        _add_eval_tasks(g, method_node, rec["evaluation"].get("tasks") or [])
        _add_use_cases(g, method_node, rec["practical"].get("use_cases") or [])

    # Second pass for paper-to-paper edges (so targets in the corpus already exist).
    for rec in records:
        _add_paper_relationships(g, _paper_id(rec["paper_id"]), rec.get("relationships") or [])

    return g


# --- Serialization ------------------------------------------------------------

def _scrub_for_serialization(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """GraphML and JSON node-link can't carry None — convert to empty strings
    (GraphML) or leave (JSON, which handles null fine)."""
    return g  # node-link JSON handles None; GraphML conversion handled inline


def save_kg(g: nx.MultiDiGraph, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Persist the KG as both node-link JSON and GraphML."""
    if out_dir is None:
        out_dir = repo_root() / "data" / "outputs" / "kg"
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    json_path = od / "savant_kg.json"
    graphml_path = od / "savant_kg.graphml"

    # Node-link JSON: clean and machine-readable.
    data = nx.node_link_data(g)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")

    # GraphML can't serialize None or booleans-as-Python-bools cleanly across
    # all tools; cast None to empty strings on a copy.
    gg = g.copy()
    for _, attrs in gg.nodes(data=True):
        for k, v in list(attrs.items()):
            if v is None:
                attrs[k] = ""
    for _, _, attrs in gg.edges(data=True):
        for k, v in list(attrs.items()):
            if v is None:
                attrs[k] = ""
    nx.write_graphml(gg, graphml_path)

    return {"json": json_path, "graphml": graphml_path}
