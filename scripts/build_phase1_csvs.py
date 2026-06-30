"""Build papers.template.csv and annotations.template.csv from the 24 extraction
JSON records + 4 held-out bibliographic entries.

This is a one-off Phase-1 script. Once Stage 2 (PDF ingestion) and Stage 3
(extraction agent) are wired in, the CSVs will be regenerated from those
artefacts automatically.

Run from repo root:
    PYTHONPATH=src python scripts/build_phase1_csvs.py
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTRACTIONS_DIR = REPO / "data" / "outputs" / "extractions"
PAPERS_CSV = REPO / "data" / "papers" / "papers.template.csv"
ANNOTATIONS_CSV = REPO / "data" / "annotations" / "annotations.template.csv"

TODAY = date.today().isoformat()


# --- Map PDF filename -> file_path for each paper_id (file_path is relative to repo root) -------
FILE_PATHS: dict[str, str] = {
    "singh_2018_xt": "data/papers/Introducing Expected Threat (xT).html",
    "rudd_2011_markov": "data/papers/rudd-xT.pdf",
    "brooks_2016_passrank": "data/papers/Brooks.pdf",
    "decroos_2017_starss": "data/papers/starss-decroos.pdf",
    "spearman_2018_beyondxg": "data/papers/beyond-xG-spearman.pdf",
    "fernandez_2019_epv": "data/papers/EPV-sloan.pdf",
    "decroos_2019_vaep": "data/papers/VAEP.pdf",
    "power_2017_passes": "data/papers/not-all-passes-are-equal.pdf",
    "bransen_2018_pass_contributions": "data/papers/pass-value-bransen-and-vanhaaren.pdf",
    "bransen_2019_pass_value": "data/papers/pass-valuation-jqas.pdf",
    "bransen_2019_choke_or_shine": "data/papers/Choke-or-Shine-Quantifying-Soccer-Players-Abilities-to-Perform-Under-Mental-Pressure.pdf",
    "pappalardo_2019_playerank": "data/papers/playerank.pdf",
    "dick_2019_positioning": "data/papers/Brefeld-RL.pdf",
    "dick_2021_actions": "data/papers/dick-rating-actions.pdf",
    "liu_2020_deep_soccer": "data/papers/Liu-deepRL.pdf",
    "vanroy_2020_xt_vs_vaep": "data/papers/xt_vs_vaep.pdf",
    "vanroy_2021_markov_reward": "data/papers/VanRoy_MDP.pdf",
    "vanroy_2021_leaving_goals": "data/papers/VanRoy2021_SSAC_Shooting.pdf",
    "merhej_2021_daxt": "data/papers/defensive-xT.pdf",
    "nakahara_2023_multiagent_rl": "data/papers/off-ball-RL.pdf",
    "rahimian_2022_beyond_av": "data/papers/rahimian-RL.pdf",
    "rahimian_2022_data_driven": "data/papers/rahimian-RL1.pdf",
    "wu_2023_offball": "data/papers/off-the-ball.pdf",
    "llana_2020_off_ball_advantage": "data/papers/off-ball-right-place-at-the-right-time.pdf",
    # held-out
    "gyarmati_2016_qpass": "data/papers/QPass.pdf",
    "overmeer_2025_epv_unet": "data/papers/U-net.pdf",
    "vanarem_2025_xt_tradeoff": "data/papers/xT2.pdf",
    "xarles_2025_av_survey": "data/papers/survey.pdf",
}


# --- Picking the primary_tag (single most informative scope.tags value) -------
def pick_primary_tag(tags: list[str]) -> str:
    priority = [
        "VAEP", "atomic-VAEP", "expected-threat", "EPV", "possession-value",
        "defensive-action", "off-ball-action", "pass-value", "shot-value",
        "graph-neural-network", "reinforcement-learning", "markov-decision-process",
        "ranking-players", "interpretability", "calibration", "deep-learning",
        "tracking-data", "event-data", "tactical-analysis", "action-valuation",
    ]
    for t in priority:
        if t in tags:
            return t
    return "other"


def list_to_csv(values: list) -> str:
    return ";".join(str(v) for v in (values or []))


def fmt_bool(v) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return ""


def fmt_str(v) -> str:
    return "" if v is None else str(v)


def action_coverage_list(ac: dict) -> str:
    if not ac:
        return ""
    return ";".join(k for k, v in ac.items() if v is True)


def claims_to_str(claims: list) -> str:
    return ";".join(
        f"{c.get('evidence_location','')} | {c.get('statement','')}"
        for c in (claims or [])
    )


def relationships_to_str(rels: list) -> str:
    return ";".join(f"{r['type']}:{r['target_paper_id']}" for r in (rels or []))


# --- Build papers.csv row for each paper ----------------------------------------
def papers_row_from_record(rec: dict, corpus_pool: str) -> dict:
    paper_id = rec["paper_id"]
    bib = rec["bibliographic"]
    return {
        "paper_id": paper_id,
        "title": fmt_str(bib.get("title")),
        "authors": ";".join(bib.get("authors") or []),
        "year": fmt_str(bib.get("year")),
        "venue": fmt_str(bib.get("venue")),
        "venue_type": fmt_str(bib.get("venue_type")),
        "doi": fmt_str(bib.get("doi")),
        "url": fmt_str(bib.get("url")),
        "open_access": fmt_bool(bib.get("open_access")),
        "file_path": FILE_PATHS.get(paper_id, ""),
        "primary_tag": pick_primary_tag(rec.get("scope", {}).get("tags") or []),
        "screening_decision": "include" if rec.get("scope", {}).get("in_scope") else "exclude",
        "screening_decision_by": "pegah",
        "screening_decision_date": TODAY,
        "needs_full_text": "false",
        "pegah_verified": "true",
        "pegah_verified_date": TODAY,
        "david_verified": "false",
        "david_verified_date": "",
        "corpus_pool": corpus_pool,
        "notes": "Auto-extracted from PDF 2026-06-29; verified by Pegah, awaiting David's independent verification.",
    }


# --- Build annotations.csv row from a candidate extraction record --------------
def annotations_row(rec: dict) -> dict:
    m = rec["model"]
    d = rec["data"]
    e = rec["evaluation"]
    p = rec["practical"]
    c = rec.get("code_release", {})
    return {
        "paper_id": rec["paper_id"],
        "annotator": "pegah",
        "annotation_date": TODAY,
        "in_scope": fmt_bool(rec["scope"]["in_scope"]),
        "scope_rationale": fmt_str(rec["scope"].get("rationale")),
        "model_name": fmt_str(m.get("name")),
        "model_family": fmt_str(m.get("family")),
        "model_framework": fmt_str(m.get("framework")),
        "model_valuation_unit": fmt_str(m.get("valuation_unit")),
        "model_inputs": list_to_csv(m.get("inputs")),
        "model_target": fmt_str(m.get("target")),
        "model_supervision": fmt_str(m.get("supervision")),
        "model_architecture": fmt_str(m.get("architecture")),
        "model_spatial_resolution": fmt_str(m.get("spatial_resolution")),
        "model_temporal_horizon": fmt_str(m.get("temporal_horizon")),
        "model_atomicity": fmt_str(m.get("atomicity")),
        "action_coverage": action_coverage_list(m.get("action_coverage", {})),
        "outputs_value_scale": fmt_str(m.get("outputs", {}).get("value_scale")),
        "outputs_aggregation_levels": list_to_csv(m.get("outputs", {}).get("aggregation_levels")),
        "data_provider": fmt_str(d.get("provider")),
        "data_type": list_to_csv(d.get("data_type")),
        "data_dataset_canonical": list_to_csv(d.get("dataset_canonical")),
        "data_leagues": list_to_csv(d.get("leagues")),
        "data_seasons": list_to_csv(d.get("seasons")),
        "data_n_matches": fmt_str(d.get("n_matches")),
        "data_is_public": fmt_bool(d.get("is_public")),
        "evaluation_metrics": list_to_csv(e.get("metrics")),
        "evaluation_tasks": list_to_csv(e.get("tasks")),
        "evaluation_baselines": list_to_csv(e.get("baselines")),
        "calibration_reported": fmt_bool(e.get("calibration_reported")),
        "uncertainty_reported": fmt_bool(e.get("uncertainty_reported")),
        "evaluation_validated_against_expert_judgment": fmt_bool(e.get("validated_against_expert_judgment")),
        "evaluation_bootstrap_or_ci_reported": fmt_bool(e.get("bootstrap_or_ci_reported")),
        "splits": fmt_str(e.get("splits")),
        "practical_use_cases": list_to_csv(p.get("use_cases")),
        "practical_inference_realtime_capable": fmt_bool(p.get("inference_realtime_capable")),
        "practical_interpretability_level": fmt_str(p.get("interpretability_level")),
        "practical_compute_intensity": fmt_str(p.get("compute_intensity")),
        "key_claims": claims_to_str(rec.get("claims")),
        "limitations": ";".join(rec.get("limitations") or []),
        "open_questions": ";".join(rec.get("open_questions") or []),
        "code_released": fmt_bool(c.get("released")),
        "code_url": fmt_str(c.get("url")),
        "code_license": fmt_str(c.get("license")),
        "code_implementer": fmt_str(c.get("implementer")),
        "code_maintenance_status": fmt_str(c.get("maintenance_status")),
        "relationships": relationships_to_str(rec.get("relationships")),
        "verified_by_pegah": "true",
        "verified_by_pegah_date": TODAY,
        "verified_by_david": "false",
        "verified_by_david_date": "",
        "verification_notes": "",
    }


# --- Held-out rows (bibliographic only, no annotation) -------------------------
HELD_OUT: dict[str, dict] = {
    "gyarmati_2016_qpass": {
        "paper_id": "gyarmati_2016_qpass",
        "bibliographic": {
            "title": "QPass: a Merit-based Evaluation of Soccer Passes",
            "authors": ["Laszlo Gyarmati", "Rade Stanojevic"],
            "year": 2016,
            "venue": "arXiv:1608.03532",
            "venue_type": "preprint",
            "doi": None,
            "url": "https://arxiv.org/abs/1608.03532",
            "open_access": True,
        },
        "scope": {"in_scope": True, "tags": ["pass-value"]},
    },
    "overmeer_2025_epv_unet": {
        "paper_id": "overmeer_2025_epv_unet",
        "bibliographic": {
            "title": "Revisiting expected possession value in football: Introducing a benchmark, U-net architecture, and reward for passes",
            "authors": ["Thijs Overmeer", "Tim Janssen", "Wim P.M. Nuijten"],
            "year": 2025,
            "venue": "arXiv:2502.02565",
            "venue_type": "preprint",
            "doi": None,
            "url": "https://arxiv.org/abs/2502.02565",
            "open_access": True,
        },
        "scope": {"in_scope": True, "tags": ["EPV"]},
    },
    "vanarem_2025_xt_tradeoff": {
        "paper_id": "vanarem_2025_xt_tradeoff",
        "bibliographic": {
            "title": "The trade-off between model flexibility and accuracy of the Expected Threat model in football",
            "authors": ["Koen W. van Arem", "Jakob Söhl", "Mirjam Bruinsma", "Geurt Jongbloed"],
            "year": 2025,
            "venue": "arXiv:2511.09457",
            "venue_type": "preprint",
            "doi": None,
            "url": "https://arxiv.org/abs/2511.09457",
            "open_access": True,
        },
        "scope": {"in_scope": True, "tags": ["expected-threat"]},
    },
    "xarles_2025_av_survey": {
        "paper_id": "xarles_2025_av_survey",
        "bibliographic": {
            "title": "Action Valuation in Sports: A Survey",
            "authors": ["Artur Xarles", "Sergio Escalera", "Thomas B. Moeslund", "Albert Clapés"],
            "year": 2025,
            "venue": "arXiv:2504.06163",
            "venue_type": "preprint",
            "doi": None,
            "url": "https://arxiv.org/abs/2504.06163",
            "open_access": True,
        },
        "scope": {"in_scope": True, "tags": ["action-valuation"]},
    },
}


def main() -> None:
    # Load all 24 candidate extraction records.
    candidates = []
    for p in sorted(EXTRACTIONS_DIR.glob("*.json")):
        rec = json.loads(p.read_text())
        candidates.append(rec)
    assert len(candidates) == 24, f"Expected 24 candidate JSONs, got {len(candidates)}"

    # ------ papers.template.csv ------
    paper_cols = [
        "paper_id", "title", "authors", "year", "venue", "venue_type",
        "doi", "url", "open_access", "file_path", "primary_tag",
        "screening_decision", "screening_decision_by", "screening_decision_date",
        "needs_full_text", "pegah_verified", "pegah_verified_date",
        "david_verified", "david_verified_date", "corpus_pool", "notes",
    ]

    rows = []
    for rec in candidates:
        rows.append(papers_row_from_record(rec, corpus_pool="candidate"))
    for rec in HELD_OUT.values():
        r = papers_row_from_record(rec, corpus_pool="held-out")
        r["notes"] = "Held-out: bibliographic-only; annotation reserved for the Stage-3 extraction agent's test pool."
        rows.append(r)

    # Sort by year then paper_id for stable diffs
    rows.sort(key=lambda r: (int(r["year"]) if r["year"] else 0, r["paper_id"]))

    PAPERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PAPERS_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=paper_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {PAPERS_CSV} ({len(rows)} rows)")

    # ------ annotations.template.csv ------
    ann_cols = [
        "paper_id", "annotator", "annotation_date", "in_scope", "scope_rationale",
        "model_name", "model_family", "model_framework", "model_valuation_unit",
        "model_inputs", "model_target", "model_supervision", "model_architecture",
        "model_spatial_resolution", "model_temporal_horizon", "model_atomicity",
        "action_coverage", "outputs_value_scale", "outputs_aggregation_levels",
        "data_provider", "data_type", "data_dataset_canonical",
        "data_leagues", "data_seasons", "data_n_matches", "data_is_public",
        "evaluation_metrics", "evaluation_tasks", "evaluation_baselines",
        "calibration_reported", "uncertainty_reported",
        "evaluation_validated_against_expert_judgment",
        "evaluation_bootstrap_or_ci_reported", "splits",
        "practical_use_cases", "practical_inference_realtime_capable",
        "practical_interpretability_level", "practical_compute_intensity",
        "key_claims", "limitations", "open_questions",
        "code_released", "code_url", "code_license", "code_implementer",
        "code_maintenance_status", "relationships",
        "verified_by_pegah", "verified_by_pegah_date",
        "verified_by_david", "verified_by_david_date", "verification_notes",
    ]

    ann_rows = [annotations_row(rec) for rec in candidates]
    ann_rows.sort(key=lambda r: r["paper_id"])

    ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ANNOTATIONS_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ann_cols)
        w.writeheader()
        w.writerows(ann_rows)
    print(f"Wrote {ANNOTATIONS_CSV} ({len(ann_rows)} rows)")


if __name__ == "__main__":
    main()
