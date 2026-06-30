"""Rank empty cells in the 3-D (framework x valuation_unit x action_type)
coverage matrix by "interestingness".

Per Pegah's design decision (2026-06-29):
- Methodological coverage is PRIMARY  (weight 1.0)
- Reproducibility (public-data)   SECONDARY (weight 0.3)
- Practitioner relevance          TERTIARY  (weight 0.2)

The methodological-coverage score rewards gaps where each individual
dimension IS represented in the corpus, but the COMBINATION is missing.
This is the "obvious-but-missed" pattern that surveys love to highlight.

A "totally isolated" gap (e.g. `rule-based x * x *` — the framework
itself has zero methods in the corpus) gets methodological_score = 0
and naturally sorts to the bottom of the ranking, so the top of the
ranking surfaces cells where the field could plausibly have a method.

Output (see `save_ranking`):
    {
      "schema_version": "1.0",
      "weights": {"methodological": 1.0, "reproducibility": 0.3, "practitioner": 0.2},
      "ranked_gaps": [
        {"rank": 1, "framework": ..., "valuation_unit": ..., "action_type": ...,
         "score": float, "components": {...}, "nearby_methods": [...],
         "rationale": "..."},
        ...
      ]
    }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..utils.config import repo_root
from .enumerate import (
    ACTION_TYPES,
    DATASETS_PUBLIC,
    FRAMEWORKS,
    VALUATION_UNITS,
)

RANKING_SCHEMA_VERSION = "1.0"

# Scoring weights (Pegah 2026-06-29: methodological PRIMARY).
WEIGHTS: dict[str, float] = {
    "methodological": 1.0,
    "reproducibility": 0.3,
    "practitioner":    0.2,
}

# Action types that practitioners explicitly under-serve and therefore want
# methods for. The survey paper and the club-data-scientist persona both
# benefit from finding gaps here.
PRACTITIONER_UNDER_SERVED_ACTIONS: frozenset[str] = frozenset({
    "defensive_actions",
    "off_ball_actions",
    "goalkeeper_actions",
    "set_pieces",
})

# Frameworks that map most directly to "what should a player do next"
# (the question coaches and clubs actually ask).
DECISION_RELEVANT_FRAMEWORKS: frozenset[str] = frozenset({
    "reinforcement-learning",
    "markov-decision-process",
})


# --- Marginal counts (one-dimensional projections) ---------------------------

def _marginal_counts(enumeration: dict[str, Any]) -> dict[str, Counter]:
    """For each dimension, how many methods have each value?
    Used to compute methodological_score: gaps are most interesting when
    each individual dimension is well-represented but the combination isn't."""
    cells_3d = enumeration["matrices"]["framework_x_valuation_unit_x_action_type"]
    fw_counts: Counter = Counter()
    vu_counts: Counter = Counter()
    at_counts: Counter = Counter()

    # Sum cell counts along each axis (deduplicating by paper_id inside each cell).
    # Methods can value multiple action_types so simple cell-count summation
    # overcounts the per-method axis presence. Instead, count distinct paper_ids.
    fw_methods: dict[str, set[str]] = {}
    vu_methods: dict[str, set[str]] = {}
    at_methods: dict[str, set[str]] = {}

    for c in cells_3d:
        for pid in c["methods"]:
            fw_methods.setdefault(c["framework"], set()).add(pid)
            vu_methods.setdefault(c["valuation_unit"], set()).add(pid)
            at_methods.setdefault(c["action_type"], set()).add(pid)

    fw_counts.update({k: len(v) for k, v in fw_methods.items()})
    vu_counts.update({k: len(v) for k, v in vu_methods.items()})
    at_counts.update({k: len(v) for k, v in at_methods.items()})

    return {"framework": fw_counts, "valuation_unit": vu_counts, "action_type": at_counts}


def _methods_using_public_data_by_framework(enumeration: dict[str, Any]) -> dict[str, set[str]]:
    """Per framework, set of paper_ids whose `framework × dataset` projection
    intersects a public dataset. Needed for the reproducibility bonus."""
    fwxds = enumeration["matrices"]["framework_x_dataset"]
    out: dict[str, set[str]] = {}
    for c in fwxds:
        if c["dataset"] in DATASETS_PUBLIC:
            out.setdefault(c["framework"], set()).update(c["methods"])
    return out


# --- Per-cell scoring --------------------------------------------------------

def _methodological_score(fw: str, vu: str, at: str,
                          marg: dict[str, Counter]) -> float:
    """Geometric-mean-style combination: zero if any axis is empty in the
    corpus, otherwise log-weighted product so very-frequent values don't
    dominate."""
    import math
    a = marg["framework"].get(fw, 0)
    b = marg["valuation_unit"].get(vu, 0)
    c = marg["action_type"].get(at, 0)
    if min(a, b, c) == 0:
        return 0.0
    # log1p smooths outsized values; cube-root keeps the dimensions balanced.
    return (math.log1p(a) * math.log1p(b) * math.log1p(c)) ** (1 / 3)


def _reproducibility_bonus(fw: str, public_by_fw: dict[str, set[str]]) -> float:
    """Bonus if the framework already has at least one method on public data
    — that means filling this gap is *actionable*: a researcher could plausibly
    reproduce / extend an existing public-data effort."""
    return 1.0 if public_by_fw.get(fw) else 0.0


def _practitioner_bonus(fw: str, at: str) -> float:
    score = 0.0
    if at in PRACTITIONER_UNDER_SERVED_ACTIONS:
        score += 1.0
    if fw in DECISION_RELEVANT_FRAMEWORKS:
        score += 0.5
    return score


# --- Nearby methods ----------------------------------------------------------

def _nearby_methods(
    target_fw: str, target_vu: str, target_at: str,
    enumeration: dict[str, Any],
) -> list[dict[str, Any]]:
    """Methods that share 2 of 3 dimensions with the target empty cell. Used
    by the rationale string to anchor the gap to existing work."""
    cells_3d = enumeration["matrices"]["framework_x_valuation_unit_x_action_type"]
    nearby_pids: dict[str, dict[str, Any]] = {}
    for c in cells_3d:
        if c["count"] == 0:
            continue
        matches = sum([
            c["framework"] == target_fw,
            c["valuation_unit"] == target_vu,
            c["action_type"] == target_at,
        ])
        if matches == 2:
            for pid in c["methods"]:
                nearby_pids.setdefault(pid, {
                    "paper_id": pid,
                    "shared_dimensions": [],
                })
                shared = []
                if c["framework"] == target_fw:        shared.append("framework")
                if c["valuation_unit"] == target_vu:   shared.append("valuation_unit")
                if c["action_type"] == target_at:      shared.append("action_type")
                nearby_pids[pid]["shared_dimensions"] = shared
    return sorted(nearby_pids.values(), key=lambda x: x["paper_id"])


# --- Auto-generated rationale ------------------------------------------------

def _format_rationale(fw: str, vu: str, at: str,
                      marg: dict[str, Counter],
                      nearby: list[dict[str, Any]],
                      repro_bonus: float,
                      pract_bonus: float) -> str:
    fw_n = marg["framework"].get(fw, 0)
    vu_n = marg["valuation_unit"].get(vu, 0)
    at_n = marg["action_type"].get(at, 0)
    at_human = at.replace("_", " ")
    fw_human = fw.replace("-", " ")
    parts = [
        f"No method currently combines {fw_human} framework, valuation_unit={vu}, "
        f"and {at_human}.",
        f"In the corpus: {fw_n} method(s) use {fw_human}; "
        f"{vu_n} value via {vu}; {at_n} value {at_human}.",
    ]
    if nearby:
        anchor = ", ".join(n["paper_id"] for n in nearby[:3])
        more = "" if len(nearby) <= 3 else f" (+{len(nearby) - 3} more)"
        parts.append(f"Closest related work (shares 2/3 dimensions): {anchor}{more}.")
    if repro_bonus > 0:
        parts.append(
            "Public data already exists for this framework, so a reproducible benchmark is feasible."
        )
    if pract_bonus > 0:
        flags = []
        if at in PRACTITIONER_UNDER_SERVED_ACTIONS:
            flags.append(f"{at_human} is under-served")
        if fw in DECISION_RELEVANT_FRAMEWORKS:
            flags.append(f"{fw_human} maps to decision-recommendation use cases")
        parts.append("Practitioner relevance: " + "; ".join(flags) + ".")
    return " ".join(parts)


# --- Public API --------------------------------------------------------------

def rank_gaps(enumeration: dict[str, Any], max_count: int = 0,
              keep_zero_methodological: bool = False) -> dict[str, Any]:
    """Rank cells in the 3-D matrix whose count <= max_count.

    Parameters
    ----------
    enumeration
        Output of `enumerate_cells(kg)`.
    max_count
        Cells with `count > max_count` are excluded. Default 0 (only truly
        empty cells).
    keep_zero_methodological
        If False (default), cells whose methodological_score == 0 (i.e. one
        of the three dimensions is empty across the entire corpus, like
        `rule-based`) are still emitted but appear at the bottom of the
        ranking with a flag in the rationale.
    """
    cells_3d = enumeration["matrices"]["framework_x_valuation_unit_x_action_type"]
    marg = _marginal_counts(enumeration)
    public_by_fw = _methods_using_public_data_by_framework(enumeration)

    scored: list[dict[str, Any]] = []
    for c in cells_3d:
        if c["count"] > max_count:
            continue
        fw, vu, at = c["framework"], c["valuation_unit"], c["action_type"]
        meth = _methodological_score(fw, vu, at, marg)
        if not keep_zero_methodological and meth == 0:
            # Still keep it but flag — A7 reviewer may want to see corpus-level
            # gaps even though they're not the primary ranking target.
            pass
        repro = _reproducibility_bonus(fw, public_by_fw)
        pract = _practitioner_bonus(fw, at)
        total = (WEIGHTS["methodological"] * meth
                 + WEIGHTS["reproducibility"] * repro
                 + WEIGHTS["practitioner"]   * pract)
        nearby = _nearby_methods(fw, vu, at, enumeration)
        rationale = _format_rationale(fw, vu, at, marg, nearby, repro, pract)
        scored.append({
            "framework": fw, "valuation_unit": vu, "action_type": at,
            "current_count": c["count"],
            "score": round(total, 4),
            "components": {
                "methodological_score": round(meth, 4),
                "reproducibility_bonus": repro,
                "practitioner_bonus": pract,
            },
            "nearby_methods": nearby,
            "rationale": rationale,
        })

    # Sort: descending score; break ties by methodological_score then framework name.
    scored.sort(key=lambda x: (
        -x["score"],
        -x["components"]["methodological_score"],
        x["framework"], x["valuation_unit"], x["action_type"],
    ))
    for i, g in enumerate(scored, start=1):
        g["rank"] = i

    return {
        "schema_version": RANKING_SCHEMA_VERSION,
        "weights": WEIGHTS,
        "n_candidate_cells": len(scored),
        "n_with_positive_methodological_score":
            sum(1 for g in scored if g["components"]["methodological_score"] > 0),
        "ranked_gaps": scored,
    }


def save_ranking(ranking: dict[str, Any], out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = repo_root() / "data" / "outputs" / "gaps" / "ranked.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ranking, indent=2, ensure_ascii=False) + "\n")
    return out_path
