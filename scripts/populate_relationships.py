"""Populate the `relationships` array in each candidate extraction JSON.

The 24 extraction-agent runs deliberately left `relationships: []` so a single
post-hoc pass could cross-reference baseline mentions against the corpus
paper_id set. This script encodes that resolution as a hand-curated table
derived from each paper's `evaluation.baselines`, `scope.rationale`, and the
extraction-agent critic notes.

Edges are CONSERVATIVE: included only when the cited paper is unambiguously
in the SAVANT corpus AND the relationship type is supported by explicit text
in the source paper (not inferred by the annotator).

Re-run after editing the RELATIONSHIPS table:
    PYTHONPATH=src python scripts/populate_relationships.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTRACTIONS_DIR = REPO / "data" / "outputs" / "extractions"

# (paper_id, relationship_type, target_paper_id, evidence) tuples.
# The evidence column is for traceability only — not stored in the JSON.
RELATIONSHIPS: list[tuple[str, str, str, str]] = [
    # ---- decroos_2019_vaep ----
    ("decroos_2019_vaep", "extends", "decroos_2017_starss",
        "Same group; VAEP generalizes the STARSS phase-based action rating to per-action goal/concede probabilities."),
    ("decroos_2019_vaep", "compares-against", "singh_2018_xt",
        "Sec. 4 of VAEP paper benchmarks against xT as the primary baseline."),

    # ---- bransen_2019_choke_or_shine ----
    ("bransen_2019_choke_or_shine", "extends", "decroos_2019_vaep",
        "Paper uses VAEP-style V(s_j) - V(s_i) contribution rating and conditions it on mental pressure."),

    # ---- bransen_2019_pass_value ----
    ("bransen_2019_pass_value", "compares-against", "gyarmati_2016_qpass",
        "QPass listed in baselines: 'QPass approximation' and 'QPass approximation xG'."),
    ("bransen_2019_pass_value", "improves-upon", "gyarmati_2016_qpass",
        "ECOM reports lower log-loss than QPass approximation on EPL 2017/18 outcome prediction."),

    # ---- dick_2021_actions ----
    ("dick_2021_actions", "extends", "dick_2019_positioning",
        "Same authors; rates ACTIONS via Delta V on the state-value function introduced in Dick & Brefeld 2019."),
    ("dick_2021_actions", "compares-against", "dick_2019_positioning",
        "Baselines explicitly list 'ConvNet value model of Dick and Brefeld (2019)'."),

    # ---- liu_2020_deep_soccer ----
    ("liu_2020_deep_soccer", "compares-against", "decroos_2019_vaep",
        "Baselines explicitly list 'VAEP (Decroos et al. 2019)'."),
    ("liu_2020_deep_soccer", "improves-upon", "decroos_2019_vaep",
        "Paired t-tests reported showing GIM outperforms VAEP on calibration and player-ranking proxies."),

    # ---- llana_2020_off_ball_advantage ----
    ("llana_2020_off_ball_advantage", "extends", "fernandez_2019_epv",
        "Off-ball advantage is layered on top of Fernandez 2019 EPV; effective value added is an EPV attribution."),

    # ---- merhej_2021_daxt ----
    ("merhej_2021_daxt", "extends", "singh_2018_xt",
        "DAxT = Defensive Action xT; model is a defensive specialisation of Singh's xT framework."),

    # ---- nakahara_2023_multiagent_rl ----
    ("nakahara_2023_multiagent_rl", "compares-against", "spearman_2018_beyondxg",
        "Baselines explicitly list 'OBSO (Off-Ball Scoring Opportunities) [Spearman 2018]'."),

    # ---- pappalardo_2019_playerank ----
    ("pappalardo_2019_playerank", "compares-against", "brooks_2016_passrank",
        "Baselines explicitly list 'Pass Shot Value (PSV) by Brooks et al. [6]'."),
    ("pappalardo_2019_playerank", "improves-upon", "brooks_2016_passrank",
        "Outperforms PSV on Wyscout-scout pair-rank agreement and majority-class baselines."),

    # ---- rahimian_2022_data_driven ----
    ("rahimian_2022_data_driven", "extends", "rahimian_2022_beyond_av",
        "Same lead author follow-up; generalizes the EPO-optimisation framework to BOTH offensive AND defensive teams."),

    # ---- vanroy_2020_xt_vs_vaep ----
    ("vanroy_2020_xt_vs_vaep", "compares-against", "singh_2018_xt",
        "Paper's title and Sec. 3 explicitly compare xT to VAEP."),
    ("vanroy_2020_xt_vs_vaep", "compares-against", "decroos_2019_vaep",
        "Paper's title and Sec. 3 explicitly compare xT to VAEP."),
    ("vanroy_2020_xt_vs_vaep", "critiques", "singh_2018_xt",
        "Sec. 4 identifies xT's structural limitations (location-only state, no risk component, ball-progression-only scope)."),
    ("vanroy_2020_xt_vs_vaep", "critiques", "decroos_2019_vaep",
        "Sec. 4 identifies VAEP limitations (action-sequence-bound horizon, complex feature engineering, opaque to interpretation)."),

    # ---- vanroy_2021_markov_reward ----
    ("vanroy_2021_markov_reward", "compares-against", "vanroy_2020_xt_vs_vaep",
        "Critic notes for the extraction record flag explicit citation of Van Roy et al. 2020 xT vs VAEP as a conceptual antecedent."),

    # ---- vanroy_2021_leaving_goals ----
    ("vanroy_2021_leaving_goals", "compares-against", "vanroy_2020_xt_vs_vaep",
        "Critic notes for the extraction record flag explicit citation of Van Roy et al. 2020 xT vs VAEP as a conceptual antecedent."),
]


def main() -> None:
    # Group relationships by source paper_id
    by_source: dict[str, list[dict[str, str]]] = {}
    for src, rel_type, target, _ in RELATIONSHIPS:
        by_source.setdefault(src, []).append(
            {"type": rel_type, "target_paper_id": target}
        )

    # Sanity: every source and target must exist in the extractions dir
    available = {p.stem for p in EXTRACTIONS_DIR.glob("*.json")}
    # Held-out paper_ids (referenced as targets but not present in extractions)
    held_out = {"gyarmati_2016_qpass", "overmeer_2025_epv_unet",
                "vanarem_2025_xt_tradeoff", "xarles_2025_av_survey"}

    for src, rel_type, target, _ in RELATIONSHIPS:
        assert src in available, f"Source {src!r} missing from extractions/"
        assert target in available or target in held_out, (
            f"Target {target!r} not in candidates or held-out"
        )

    # Patch each candidate JSON: replace relationships array
    updated = 0
    for path in sorted(EXTRACTIONS_DIR.glob("*.json")):
        rec = json.loads(path.read_text())
        new_rels = by_source.get(rec["paper_id"], [])
        if rec.get("relationships") != new_rels:
            rec["relationships"] = new_rels
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
            updated += 1
            print(f"  patched {rec['paper_id']:40s}  +{len(new_rels)} edges")

    print(f"\nTotal edges added: {len(RELATIONSHIPS)}  across {len(by_source)} source papers")
    print(f"JSONs updated: {updated}")


if __name__ == "__main__":
    main()
