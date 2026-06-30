"""Enumerate the multi-dimensional coverage matrices from the KG.

For each method in the corpus, the KG tells us its `framework`,
`valuation_unit`, the set of `action_type`s it values, and the set of
`dataset_canonical` entries it was trained / evaluated on. From these
projections we can build several coverage matrices:

    - framework x valuation_unit x action_type  (primary 3-D matrix, 126 cells)
    - framework x action_type                   (2-D projection)
    - valuation_unit x action_type              (2-D projection)
    - framework x valuation_unit                (2-D projection — corpus shape)
    - framework x dataset                       (dataset-usage map)
    - dataset x action_type                     (reproducibility surface)

Each cell value is the COUNT of methods whose attributes intersect that
combination. A cell with count 0 is a "gap" — A6 then ranks gaps by
interestingness.

Output of `enumerate_cells(g)` is a JSON-serialisable dict:
    {
      "schema_version": "1.0",
      "corpus_size": 24,
      "dimensions": { "frameworks": [...], "valuation_units": [...], ... },
      "matrices": {
        "framework_x_valuation_unit_x_action_type": [
          {"framework": "...", "valuation_unit": "...", "action_type": "...",
           "count": int, "methods": [...]},
          ...
        ],
        ...
      }
    }

The matrices are flat lists (instead of nested dicts) for ease of pandas /
spreadsheet round-trip.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import networkx as nx

from ..utils.config import repo_root

ENUMERATION_SCHEMA_VERSION = "1.0"

# Canonical dimension value lists. Frozen as constants so the enumeration is
# deterministic even when the corpus shrinks/grows. (Cells that are empty
# because the dimension value isn't in the corpus AT ALL are still emitted —
# that's where the most important gaps live.)
FRAMEWORKS: tuple[str, ...] = (
    "expectation-based",
    "markov-decision-process",
    "reinforcement-learning",
    "physics-based",
    "rule-based",
    "hybrid",
)

VALUATION_UNITS: tuple[str, ...] = ("action", "state", "both")

ACTION_TYPES: tuple[str, ...] = (
    "passes",
    "shots",
    "dribbles_carries",
    "defensive_actions",
    "off_ball_actions",
    "set_pieces",
    "goalkeeper_actions",
)

# Public-data identifiers (mirrored from knowledge_graph.build._PUBLIC_DATASETS).
DATASETS_PUBLIC: frozenset[str] = frozenset({
    "statsbomb-open",
    "statsbomb-360-open",
    "wyscout-public-2017-18",
    "skillcorner-public",
    "metrica-sports-public",
    "other-public",
})


# --- Per-method projections from the KG --------------------------------------

def _method_nodes(g: nx.MultiDiGraph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("node_type") == "method"]


def _proposing_paper(g: nx.MultiDiGraph, method_node: str) -> str | None:
    for u, _, ed in g.in_edges(method_node, data=True):
        if ed.get("edge_type") == "PROPOSES":
            return u
    return None


def _outgoing_names(g: nx.MultiDiGraph, method_node: str, edge_type: str) -> list[str]:
    """All neighbours of `method_node` reached via edges of `edge_type`,
    returned as the neighbour's `name` attribute."""
    out = []
    for _, v, ed in g.out_edges(method_node, data=True):
        if ed.get("edge_type") == edge_type:
            name = g.nodes[v].get("name")
            if name is not None:
                out.append(name)
    return out


def _method_projection(g: nx.MultiDiGraph, method_node: str) -> dict[str, Any]:
    """Return the method's framework, valuation_unit, action_types and
    datasets in a uniform dict."""
    d = g.nodes[method_node]
    paper_node = _proposing_paper(g, method_node)
    paper_id = g.nodes[paper_node].get("paper_id") if paper_node else None
    return {
        "method_node": method_node,
        "method_name": d.get("name"),
        "paper_id": paper_id,
        "framework": d.get("framework"),
        "valuation_unit": d.get("valuation_unit"),
        "action_types": _outgoing_names(g, method_node, "VALUES"),
        "datasets":     _outgoing_names(g, method_node, "USES_DATASET"),
    }


# --- Coverage-matrix builders ------------------------------------------------

def _three_d_matrix(projections: list[dict]) -> list[dict]:
    """framework x valuation_unit x action_type."""
    cells: list[dict] = []
    for fw, vu, at in itertools.product(FRAMEWORKS, VALUATION_UNITS, ACTION_TYPES):
        methods = [
            p["paper_id"] for p in projections
            if p["framework"] == fw
            and p["valuation_unit"] == vu
            and at in p["action_types"]
        ]
        cells.append({
            "framework": fw, "valuation_unit": vu, "action_type": at,
            "count": len(methods), "methods": sorted(methods),
        })
    return cells


def _two_d_matrix(projections: list[dict], dim_a: str, dim_b: str,
                  values_a: tuple[str, ...], values_b: tuple[str, ...],
                  method_attr_a: str, method_attr_b: str,
                  list_b: bool = False) -> list[dict]:
    """Generic 2-D builder. `list_b=True` means dim_b is a multi-valued attr
    (e.g. action_types) so a method can appear in multiple b-buckets."""
    cells: list[dict] = []
    for a, b in itertools.product(values_a, values_b):
        methods = []
        for p in projections:
            if p[method_attr_a] != a:
                continue
            attr_b = p[method_attr_b]
            if (b in attr_b) if list_b else (attr_b == b):
                methods.append(p["paper_id"])
        cells.append({
            dim_a: a, dim_b: b,
            "count": len(methods), "methods": sorted(methods),
        })
    return cells


def _dataset_universe(projections: list[dict]) -> tuple[str, ...]:
    """Distinct datasets seen across projections, sorted (public first)."""
    ds = {d for p in projections for d in p["datasets"]}
    return tuple(sorted(ds, key=lambda x: (0 if x in DATASETS_PUBLIC else 1, x)))


# --- Public API --------------------------------------------------------------

def enumerate_cells(g: nx.MultiDiGraph) -> dict[str, Any]:
    """Compute every coverage matrix and return as one JSON-ready dict."""
    methods = _method_nodes(g)
    projections = [_method_projection(g, m) for m in methods]
    datasets_in_corpus = _dataset_universe(projections)

    matrices: dict[str, list[dict]] = {
        "framework_x_valuation_unit_x_action_type":
            _three_d_matrix(projections),
        "framework_x_action_type": _two_d_matrix(
            projections, "framework", "action_type",
            FRAMEWORKS, ACTION_TYPES,
            "framework", "action_types", list_b=True),
        "valuation_unit_x_action_type": _two_d_matrix(
            projections, "valuation_unit", "action_type",
            VALUATION_UNITS, ACTION_TYPES,
            "valuation_unit", "action_types", list_b=True),
        "framework_x_valuation_unit": _two_d_matrix(
            projections, "framework", "valuation_unit",
            FRAMEWORKS, VALUATION_UNITS,
            "framework", "valuation_unit", list_b=False),
        "framework_x_dataset": _two_d_matrix(
            projections, "framework", "dataset",
            FRAMEWORKS, datasets_in_corpus,
            "framework", "datasets", list_b=True),
        "dataset_x_action_type": _two_d_matrix(
            projections, "dataset", "action_type",
            datasets_in_corpus, ACTION_TYPES,
            "datasets", "action_types", list_b=True),
    }
    # In dataset_x_action_type we passed list_b=True for action_types, but
    # the source side is also a list — fix that with a dedicated builder.
    matrices["dataset_x_action_type"] = _build_dataset_x_action(projections, datasets_in_corpus)

    return {
        "schema_version": ENUMERATION_SCHEMA_VERSION,
        "corpus_size": len(methods),
        "dimensions": {
            "frameworks": list(FRAMEWORKS),
            "valuation_units": list(VALUATION_UNITS),
            "action_types": list(ACTION_TYPES),
            "datasets_in_corpus": list(datasets_in_corpus),
            "datasets_public": sorted(DATASETS_PUBLIC),
        },
        "matrices": matrices,
    }


def _build_dataset_x_action(projections: list[dict],
                            datasets: tuple[str, ...]) -> list[dict]:
    """Special-case 2-D builder where BOTH dimensions are multi-valued on
    the source method side."""
    cells: list[dict] = []
    for ds, at in itertools.product(datasets, ACTION_TYPES):
        methods = sorted(
            p["paper_id"] for p in projections
            if ds in p["datasets"] and at in p["action_types"]
        )
        cells.append({
            "dataset": ds, "action_type": at,
            "count": len(methods), "methods": methods,
        })
    return cells


def methods_in_cell(g: nx.MultiDiGraph, **filters: str) -> list[dict[str, Any]]:
    """Return all method projections that match every (attr=value) filter.

    Valid filter keys: framework, valuation_unit, action_type, dataset.

    Example
    -------
    >>> methods_in_cell(kg, framework="reinforcement-learning",
    ...                     action_type="goalkeeper_actions")
    []   # empty - this is one of the gaps
    """
    valid = {"framework", "valuation_unit", "action_type", "dataset"}
    bad = set(filters) - valid
    if bad:
        raise ValueError(f"Unknown filter keys: {bad}; valid: {valid}")

    projections = [_method_projection(g, m) for m in _method_nodes(g)]
    out: list[dict[str, Any]] = []
    for p in projections:
        if "framework" in filters and p["framework"] != filters["framework"]:
            continue
        if "valuation_unit" in filters and p["valuation_unit"] != filters["valuation_unit"]:
            continue
        if "action_type" in filters and filters["action_type"] not in p["action_types"]:
            continue
        if "dataset" in filters and filters["dataset"] not in p["datasets"]:
            continue
        out.append(p)
    return out


def save_enumeration(enumeration: dict[str, Any], out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = repo_root() / "data" / "outputs" / "gaps" / "enumeration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enumeration, indent=2, ensure_ascii=False) + "\n")
    return out_path
