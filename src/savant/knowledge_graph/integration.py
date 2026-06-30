"""Compare a newly-extracted record against the existing SAVANT KG.

Used by the UI's bring-your-own-paper flow (Option B of the Live Demo) to
turn a structured extraction into a *positioned* extraction — i.e. the
user gets back not just their record but where it sits in the corpus.

Three projections:
- `nearest_neighbours(new_rec, kg, top_k)` — methods most similar by
  framework + valuation_unit + action-type Jaccard.
- `coverage_diff(new_rec, kg)` — for each action_type, where the new
  method stands relative to corpus prevalence.
- `gap_diff(new_rec, enumeration)` — for each (framework × valuation_unit
  × action_type) cell the new method occupies, was that cell empty before?
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .build import _PUBLIC_DATASETS  # noqa: F401 (re-exported as convenience)


def _new_method_attrs(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the comparison-relevant fields from a v0.3.0 record."""
    m = record.get("model", {}) or {}
    ac = m.get("action_coverage", {}) or {}
    return {
        "name":           m.get("name") or "(unnamed)",
        "framework":      m.get("framework"),
        "valuation_unit": m.get("valuation_unit"),
        "action_types":   {k for k, v in ac.items() if v},
    }


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def nearest_neighbours(
    record: dict[str, Any],
    g: nx.MultiDiGraph,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Methods in the KG most similar to the new record.

    Similarity = 0.4·(framework match) + 0.3·(valuation_unit match) + 0.3·(action-type Jaccard).
    Returns up to `top_k` `{paper_id, method_name, similarity, components}` dicts.
    """
    new = _new_method_attrs(record)
    out: list[dict[str, Any]] = []
    for n, d in g.nodes(data=True):
        if d.get("node_type") != "method":
            continue
        their_ac: set[str] = set()
        for _, target, ed in g.out_edges(n, data=True):
            if ed.get("edge_type") == "VALUES":
                tname = g.nodes[target].get("name")
                if tname:
                    their_ac.add(tname)
        fw_match = 1.0 if d.get("framework") == new["framework"] and new["framework"] else 0.0
        vu_match = 1.0 if d.get("valuation_unit") == new["valuation_unit"] and new["valuation_unit"] else 0.0
        action_j = _jaccard(new["action_types"], their_ac)
        sim = 0.4 * fw_match + 0.3 * vu_match + 0.3 * action_j

        # Find the paper that proposes this method
        paper_id = None
        for u, _, ed in g.in_edges(n, data=True):
            if ed.get("edge_type") == "PROPOSES":
                paper_id = g.nodes[u].get("paper_id")
                break

        out.append({
            "paper_id": paper_id or n,
            "method_name": d.get("name"),
            "framework": d.get("framework"),
            "valuation_unit": d.get("valuation_unit"),
            "action_types_valued": sorted(their_ac),
            "similarity": round(sim, 3),
            "components": {
                "framework_match": fw_match,
                "valuation_unit_match": vu_match,
                "action_type_jaccard": round(action_j, 3),
            },
        })
    out.sort(key=lambda x: -x["similarity"])
    return out[:top_k]


def coverage_diff(record: dict[str, Any], g: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """For each action_type, how does the new method stand vs corpus prevalence?

    Returns a list of `{action_type, new_covers, corpus_n, corpus_pct,
    relation}` dicts where relation is one of:
    - 'fills-gap'  → corpus_n == 0 (new method does it, nobody else does)
    - 'rare'       → corpus_n ≤ 25% of corpus methods
    - 'common'     → corpus_n > 25%
    - 'absent'     → new method doesn't cover this action_type
    """
    new = _new_method_attrs(record)

    # Count methods covering each action_type
    n_methods = sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "method")
    counts: dict[str, int] = {}
    for n, d in g.nodes(data=True):
        if d.get("node_type") != "method":
            continue
        for _, target, ed in g.out_edges(n, data=True):
            if ed.get("edge_type") == "VALUES":
                tname = g.nodes[target].get("name")
                if tname:
                    counts[tname] = counts.get(tname, 0) + 1

    action_types = [
        "passes", "shots", "dribbles_carries",
        "defensive_actions", "off_ball_actions",
        "set_pieces", "goalkeeper_actions",
    ]
    out: list[dict[str, Any]] = []
    for at in action_types:
        corpus_n = counts.get(at, 0)
        pct = corpus_n / n_methods if n_methods else 0.0
        new_covers = at in new["action_types"]
        if not new_covers:
            relation = "absent"
        elif corpus_n == 0:
            relation = "fills-gap"
        elif pct <= 0.25:
            relation = "rare"
        else:
            relation = "common"
        out.append({
            "action_type": at,
            "new_covers": new_covers,
            "corpus_n": corpus_n,
            "corpus_pct": round(pct * 100, 1),
            "relation": relation,
        })
    return out


def gap_diff(record: dict[str, Any], enumeration: dict[str, Any]) -> list[dict[str, Any]]:
    """For each (framework × valuation_unit × action_type) cell the new
    method occupies, report whether that cell was empty in the corpus."""
    new = _new_method_attrs(record)
    if not new["framework"] or not new["valuation_unit"]:
        return []

    cells_3d = enumeration["matrices"]["framework_x_valuation_unit_x_action_type"]
    cell_lookup = {
        (c["framework"], c["valuation_unit"], c["action_type"]): c
        for c in cells_3d
    }

    out: list[dict[str, Any]] = []
    for at in new["action_types"]:
        key = (new["framework"], new["valuation_unit"], at)
        cell = cell_lookup.get(key)
        if cell is None:
            continue
        out.append({
            "framework": new["framework"],
            "valuation_unit": new["valuation_unit"],
            "action_type": at,
            "corpus_count_before": cell["count"],
            "would_fill_gap": cell["count"] == 0,
            "existing_methods_in_cell": cell["methods"],
        })
    return out
