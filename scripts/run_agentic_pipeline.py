"""WP 2 orchestrator: critic → taxonomy → gap-agent end-to-end.

Two modes:

  --mode mock   (default, safe)  Uses MockClient with inline canned responses.
                                  No API calls. No spend. Useful for
                                  validating the pipeline structure, CI, and
                                  before committing to real LLM spend.

  --mode live                     Uses AnthropicClient. Requires
                                  ANTHROPIC_API_KEY env var. Real tokens.

Three stages, each can be skipped:

  --skip-critic    Skip the critic-agent pass over extraction records.
  --skip-taxonomy  Skip the taxonomy normalisation of free-text fields.
  --skip-gap       Skip the gap-agent's manuscript-quality rationale.

Safety:
  Critic verdicts modify `data/outputs/extractions/*.json` (patching
  provenance.critic). To prevent accidental corruption of the gold standard
  in mock mode, the critic stage is DRY-RUN by default — pass `--write` to
  actually save the patches.

  Taxonomy and gap stages always write their own NEW files
  (`data/outputs/taxonomy/` and `data/outputs/gaps/llm_rationales.json`),
  so they don't need a write flag.

Usage:
  PYTHONPATH=src python scripts/run_agentic_pipeline.py
  PYTHONPATH=src python scripts/run_agentic_pipeline.py --mode live --write
  PYTHONPATH=src python scripts/run_agentic_pipeline.py --skip-critic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from savant.agents import AnthropicClient, MockClient  # noqa: E402
from savant.extraction.critic import (  # noqa: E402
    CriticRunner,
    apply_verdict_to_record,
)
from savant.gap_analysis import (  # noqa: E402
    GapRunner,
    build_inputs_from_artifacts,
    enumerate_cells,
    flag_invalid_paper_ids,
    rank_gaps,
    save_llm_rationales,
)
from savant.knowledge_graph import build_kg, load_all_extractions  # noqa: E402
from savant.taxonomy import (  # noqa: E402
    TaxonomyRunner,
    collect_examples,
    collect_raw_values,
    save_taxonomy,
)

# Fields the taxonomy pass normalises in mock mode (per prompts/taxonomy_agent.md).
DEFAULT_TAXONOMY_FIELDS: tuple[str, ...] = (
    "evaluation.metrics",
    "evaluation.baselines",
    "model.architecture",
    "data.provider",
)

# --- Mock-mode canned-response builders -------------------------------------

def _to_snake(s: str) -> str:
    """Quick-and-dirty snake_case canonical for the mock taxonomy responses."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "unknown"


def _stub_critic_response(record: dict) -> dict:
    return {
        "verdict": "accept",
        "schema_conformant": True,
        "factual_issues": [],
        "completeness_issues": [],
        "style_issues": [],
        "revision_instructions": "",
        "notes": f"[mock] Auto-accept for {record.get('paper_id')}.",
    }


def _stub_taxonomy_response(field: str, raw_values: list[str]) -> dict:
    return {
        "field": field,
        "mappings": [
            {"raw": v, "canonical": _to_snake(v), "confidence": 0.5}
            for v in raw_values
        ],
        "new_vocabulary": [],
        "unresolved": [],
    }


def _stub_gap_response(top_k_ranking: list[dict]) -> dict:
    """One demonstrative gap that consolidates the top-ranked MDP × action
    cells. Schema-conformant; intentionally generic so this works on any
    corpus state."""
    md_cells = [
        {
            "framework": g["framework"],
            "valuation_unit": g["valuation_unit"],
            "action_type": g["action_type"],
            "auto_rank": g["rank"],
            "auto_score": g["score"],
        }
        for g in top_k_ranking[:6]
    ]
    return {
        "gaps": [
            {
                "id": "gap-001-mock",
                "title": "Mock-mode placeholder gap",
                "consolidated_auto_ranks": md_cells,
                "the_gap": ("[mock] This is a placeholder gap produced by the "
                            "orchestrator in --mode mock. Run with --mode live "
                            "to get a real LLM-authored rationale."),
                "scientific_motivation": "[mock] Placeholder text.",
                "practitioner_motivation": "[mock] Placeholder text.",
                "closest_existing_work": ["no exact match in corpus"],
                "suggested_first_study": "[mock] Run with --mode live for real output.",
                "novelty_score": 0.5,
                "feasibility_score": 0.5,
                "confidence": 0.0,
            }
        ],
        "agent_notes": "Mock-mode response. Inspect with --mode live for real content.",
    }


def _build_mock_client(records, taxonomy_fields, ranking, *,
                       skip_critic: bool, skip_taxonomy: bool,
                       skip_gap: bool):
    """Pre-queue canned responses for every call the orchestrator will make
    in mock mode. Order MUST match the call order AND must NOT queue
    responses for skipped stages (otherwise MockClient will pop the wrong
    response for a later stage)."""
    responses = []
    if not skip_critic:
        for r in records:
            responses.append(_stub_critic_response(r))
    if not skip_taxonomy:
        for field in taxonomy_fields:
            raw = collect_raw_values(field, records)
            responses.append(_stub_taxonomy_response(field, raw))
    if not skip_gap:
        responses.append(_stub_gap_response(ranking["ranked_gaps"]))
    return MockClient(responses=responses)


# --- Logging helpers --------------------------------------------------------

def _log(stage: str, msg: str, elapsed: float | None = None) -> None:
    suffix = f"  [{elapsed:5.2f}s]" if elapsed is not None else ""
    print(f"  {stage}  {msg}{suffix}")


# --- Stage runners ----------------------------------------------------------

def run_critic_stage(client, records, *, write: bool,
                     records_dir: Path, mock_mode: bool, model_id: str) -> dict:
    runner = CriticRunner(client)
    counts = {"accept": 0, "revise": 0, "reject": 0}
    for record in records:
        verdict = runner.run(record=record, source="(source chunking not yet implemented)")
        counts[verdict["verdict"]] = counts.get(verdict["verdict"], 0) + 1
        if write:
            patched = apply_verdict_to_record(record, verdict, model_id=model_id)
            (records_dir / f"{record['paper_id']}.json").write_text(
                json.dumps(patched, indent=2, ensure_ascii=False) + "\n"
            )
    return counts


def run_taxonomy_stage(client, records, fields, *, out_dir: Path) -> list[Path]:
    runner = TaxonomyRunner(client)
    out_paths: list[Path] = []
    for field in fields:
        raw = collect_raw_values(field, records)
        examples = collect_examples(field, records, limit=5)
        result = runner.run(
            field=field, values=raw,
            controlled_vocabulary=[], examples=examples,
        )
        out_paths.append(save_taxonomy(result, out_dir=out_dir))
    return out_paths


def run_gap_stage(client, *, g, enumeration, ranking, records,
                  out_path: Path, top_k: int = 20) -> dict:
    inputs = build_inputs_from_artifacts(
        g, enumeration, ranking,
        extraction_records=records, top_k=top_k,
    )
    runner = GapRunner(client)
    result = runner.run(**inputs)
    save_llm_rationales(result, out_path=out_path)
    valid_pids = {r["paper_id"] for r in records}
    invalid = flag_invalid_paper_ids(result, valid_pids)
    if invalid:
        result["_post_hoc_invalid_paper_ids"] = invalid
    return {"n_gaps": len(result["gaps"]), "invalid_paper_ids": invalid}


# --- Main entry point -------------------------------------------------------

def run(
    *,
    mode: str = "mock",
    skip_critic: bool = False,
    skip_taxonomy: bool = False,
    skip_gap: bool = False,
    write: bool = False,
    extractions_dir: Path | None = None,
    taxonomy_out_dir: Path | None = None,
    gap_out_path: Path | None = None,
    taxonomy_fields: tuple[str, ...] = DEFAULT_TAXONOMY_FIELDS,
    top_k_gaps: int = 20,
) -> dict[str, Any]:
    """Programmatic entry point. Returns a summary dict for tests + scripting."""
    t_total = time.perf_counter()
    print(f"\n=== SAVANT WP 2 — Agentic pipeline (mode={mode}, write={write}) ===\n")

    extractions_dir = extractions_dir or (REPO / "data" / "outputs" / "extractions")
    taxonomy_out_dir = taxonomy_out_dir or (REPO / "data" / "outputs" / "taxonomy")
    gap_out_path = gap_out_path or (REPO / "data" / "outputs" / "gaps" / "llm_rationales.json")

    # Load the artifacts every stage needs.
    records = load_all_extractions(extractions_dir)
    g = build_kg(records)
    enumeration = enumerate_cells(g)
    ranking = rank_gaps(enumeration)

    # Build the client.
    if mode == "live":
        client = AnthropicClient()
        model_id_for_provenance = "claude/agentic-pipeline-live"
    else:
        client = _build_mock_client(
            records, taxonomy_fields, ranking,
            skip_critic=skip_critic, skip_taxonomy=skip_taxonomy,
            skip_gap=skip_gap,
        )
        model_id_for_provenance = "mock/agentic-pipeline"

    summary: dict[str, Any] = {"mode": mode, "write": write}

    # Critic
    if not skip_critic:
        t0 = time.perf_counter()
        counts = run_critic_stage(
            client, records, write=write,
            records_dir=extractions_dir, mock_mode=(mode == "mock"),
            model_id=model_id_for_provenance,
        )
        summary["critic"] = counts
        verdict_summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        _log("critic", f"{len(records)} records → {verdict_summary or 'no verdicts'}",
             time.perf_counter() - t0)
        if not write:
            _log("critic", "(dry-run; pass --write to patch provenance.critic in records)")
    else:
        _log("critic", "skipped")

    # Taxonomy
    if not skip_taxonomy:
        t0 = time.perf_counter()
        paths = run_taxonomy_stage(client, records, taxonomy_fields,
                                    out_dir=taxonomy_out_dir)
        summary["taxonomy_files"] = [str(p) for p in paths]
        _log("taxonomy", f"normalised {len(paths)} fields", time.perf_counter() - t0)
    else:
        _log("taxonomy", "skipped")

    # Gap
    if not skip_gap:
        t0 = time.perf_counter()
        info = run_gap_stage(
            client, g=g, enumeration=enumeration, ranking=ranking,
            records=records, out_path=gap_out_path, top_k=top_k_gaps,
        )
        summary["gap"] = info
        _log("gap", f"produced {info['n_gaps']} consolidated rationales → {gap_out_path.name}",
             time.perf_counter() - t0)
        if info["invalid_paper_ids"]:
            print(f"    WARNING: gap-agent referenced unknown paper_ids: {info['invalid_paper_ids']}")
    else:
        _log("gap", "skipped")

    elapsed = time.perf_counter() - t_total
    print(f"\n=== Done in {elapsed:.2f}s ===\n")
    summary["elapsed_s"] = round(elapsed, 3)
    return summary


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--skip-critic", action="store_true")
    ap.add_argument("--skip-taxonomy", action="store_true")
    ap.add_argument("--skip-gap", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="actually save critic verdicts back to extraction JSONs "
                         "(default: dry-run for safety)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="top-K ranked gaps passed to the gap-agent")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        mode=args.mode,
        skip_critic=args.skip_critic,
        skip_taxonomy=args.skip_taxonomy,
        skip_gap=args.skip_gap,
        write=args.write,
        top_k_gaps=args.top_k,
    )
