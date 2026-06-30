"""Gap-agent runner — LLM v2 of A6 (ranking) + A7 (rationale).

Reads the enumeration + algorithmic ranking + a compact corpus summary,
calls the LLM with `prompts/gap_agent.md`, and returns a consolidated list
of manuscript-quality rationales.

Complements (does NOT replace) `data/outputs/gaps/top5_rationale.md` —
which is the hand-written ground truth. The LLM output goes to
`data/outputs/gaps/llm_rationales.json` so the two can be compared as part
of WP 6 validation.

Typical usage:

    g = build_kg(load_all_extractions())
    enum = enumerate_cells(g)
    ranking = rank_gaps(enum)
    inputs = build_inputs_from_artifacts(g, enum, ranking)
    result = GapRunner(client).run(**inputs)
    save_llm_rationales(result)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from ..agents import AgentRunner, AgentValidationError, LLMClient, format_input_block
from ..utils.config import repo_root
from .enumerate import DATASETS_PUBLIC

DEFAULT_LLM_RATIONALES_PATH = (
    repo_root() / "data" / "outputs" / "gaps" / "llm_rationales.json"
)

# Per-gap required keys (per prompts/gap_agent.md output schema)
_REQUIRED_GAP_KEYS: frozenset[str] = frozenset({
    "id",
    "title",
    "the_gap",
    "scientific_motivation",
    "practitioner_motivation",
    "closest_existing_work",
    "suggested_first_study",
    "novelty_score",
    "feasibility_score",
    "confidence",
})


class GapRunner(AgentRunner):
    """One pass over the gap-analysis artifacts → consolidated rationales."""

    AGENT_NAME = "gap"

    def _render_user_message(
        self,
        *,
        coverage_summary: dict[str, Any],
        algorithmic_ranking: list[dict[str, Any]],
        corpus_papers: list[dict[str, Any]],
        dataset_facts: dict[str, Any],
        author_open_questions: list[str] | None = None,
    ) -> str:
        blocks = [
            format_input_block("Coverage summary", coverage_summary),
            format_input_block("Algorithmic ranking (top-K)", algorithmic_ranking),
            format_input_block("Corpus papers", corpus_papers),
            format_input_block("Dataset facts", dataset_facts),
        ]
        if author_open_questions:
            blocks.append(format_input_block(
                "Author open questions (inspiration only)",
                author_open_questions,
            ))
        return "\n".join(blocks)

    def _validate(self, response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response.get("gaps"), list):
            raise AgentValidationError(
                "gap-agent response must contain a 'gaps' list"
            )
        if not response["gaps"]:
            raise AgentValidationError(
                "gap-agent response has empty 'gaps' list — prompt requires 5–8 entries"
            )
        for i, gap in enumerate(response["gaps"]):
            missing = _REQUIRED_GAP_KEYS - gap.keys()
            if missing:
                raise AgentValidationError(
                    f"gaps[{i}] missing required keys: {sorted(missing)}"
                )
            if not isinstance(gap["closest_existing_work"], list):
                raise AgentValidationError(
                    f"gaps[{i}].closest_existing_work must be a list of paper_id strings"
                )
            for score_field in ("novelty_score", "feasibility_score", "confidence"):
                v = gap[score_field]
                if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                    raise AgentValidationError(
                        f"gaps[{i}].{score_field} must be a number in [0, 1]; got {v!r}"
                    )
        return response


# --- Input construction from artifacts ---------------------------------------

def _coverage_summary(enumeration: dict[str, Any]) -> dict[str, Any]:
    """Compact per-dimension summary the LLM can quote directly."""
    cells_3d = enumeration["matrices"]["framework_x_valuation_unit_x_action_type"]
    fw_methods: dict[str, set[str]] = {}
    vu_methods: dict[str, set[str]] = {}
    at_methods: dict[str, set[str]] = {}
    for c in cells_3d:
        for pid in c["methods"]:
            fw_methods.setdefault(c["framework"], set()).add(pid)
            vu_methods.setdefault(c["valuation_unit"], set()).add(pid)
            at_methods.setdefault(c["action_type"], set()).add(pid)
    return {
        "corpus_size": enumeration["corpus_size"],
        "frameworks_with_method_counts":
            {k: len(v) for k, v in sorted(fw_methods.items())},
        "valuation_units_with_method_counts":
            {k: len(v) for k, v in sorted(vu_methods.items())},
        "action_types_with_method_counts":
            {k: len(v) for k, v in sorted(at_methods.items())},
        "datasets_in_corpus": enumeration["dimensions"]["datasets_in_corpus"],
        "datasets_public_universe": enumeration["dimensions"]["datasets_public"],
    }


def _corpus_papers(g: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """One {paper_id, title, year, method_name, framework, valuation_unit}
    dict per candidate paper. Stable order by year, paper_id."""
    out: list[dict[str, Any]] = []
    for n, d in g.nodes(data=True):
        if d.get("node_type") != "paper" or d.get("_stub"):
            continue
        method_name = framework = valuation_unit = None
        for _, mnode, edata in g.out_edges(n, data=True):
            if edata.get("edge_type") == "PROPOSES":
                m = g.nodes[mnode]
                method_name = m.get("name")
                framework = m.get("framework")
                valuation_unit = m.get("valuation_unit")
                break
        out.append({
            "paper_id": d.get("paper_id") or n.split(":", 1)[-1],
            "title": d.get("title"),
            "year": d.get("year"),
            "method_name": method_name,
            "framework": framework,
            "valuation_unit": valuation_unit,
        })
    out.sort(key=lambda x: (x["year"] or 0, x["paper_id"]))
    return out


def _dataset_facts(g: nx.MultiDiGraph,
                   extraction_records: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Headline-cite-able single numbers about the corpus."""
    datasets_in_corpus = {n for n, d in g.nodes(data=True)
                          if d.get("node_type") == "dataset"}
    public_used = {n for n in datasets_in_corpus
                   if g.nodes[n].get("name") in DATASETS_PUBLIC}
    n_methods_with_code = 0
    n_methods_realtime = 0
    for _, d in g.nodes(data=True):
        if d.get("node_type") != "method":
            continue
        if d.get("code_released"):
            n_methods_with_code += 1
        if d.get("inference_realtime_capable") is True:
            n_methods_realtime += 1
    # Count methods that USE at least one public dataset — the practical
    # reproducibility headline number, distinct from "how many distinct
    # public datasets exist in the corpus".
    n_methods_on_public_data = 0
    for n, d in g.nodes(data=True):
        if d.get("node_type") != "method":
            continue
        for _, target, ed in g.out_edges(n, data=True):
            if ed.get("edge_type") == "USES_DATASET" and target in public_used:
                n_methods_on_public_data += 1
                break

    facts = {
        "n_corpus_methods": sum(1 for _, d in g.nodes(data=True)
                                if d.get("node_type") == "method"),
        "n_datasets_used_in_corpus": len(datasets_in_corpus),
        "n_public_datasets_used_in_corpus": len(public_used),
        "n_methods_using_public_dataset": n_methods_on_public_data,
        "n_methods_with_code_released": n_methods_with_code,
        "n_methods_inference_realtime_capable": n_methods_realtime,
    }
    if extraction_records is not None:
        n_validated = sum(
            1 for r in extraction_records
            if (r.get("evaluation") or {}).get("validated_against_expert_judgment")
        )
        facts["n_methods_validated_against_expert_judgment"] = n_validated
    return facts


def _author_open_questions(extraction_records: list[dict[str, Any]]) -> list[str]:
    """Flat list of all `open_questions` strings across the corpus, prefixed
    with paper_id for traceability. Capped at 80 entries to bound tokens."""
    out: list[str] = []
    for r in extraction_records:
        pid = r.get("paper_id", "?")
        for q in (r.get("open_questions") or []):
            if isinstance(q, str) and q.strip():
                out.append(f"[{pid}] {q.strip()}")
                if len(out) >= 80:
                    return out
    return out


def build_inputs_from_artifacts(
    g: nx.MultiDiGraph,
    enumeration: dict[str, Any],
    ranking: dict[str, Any],
    extraction_records: list[dict[str, Any]] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Assemble the dict passed to `GapRunner.run(**inputs)` from the
    artifacts produced by A3 + A5 + A6."""
    return {
        "coverage_summary": _coverage_summary(enumeration),
        "algorithmic_ranking": ranking["ranked_gaps"][:top_k],
        "corpus_papers": _corpus_papers(g),
        "dataset_facts": _dataset_facts(g, extraction_records),
        "author_open_questions": (
            _author_open_questions(extraction_records)
            if extraction_records is not None else []
        ),
    }


# --- Persistence + cross-reference helpers ----------------------------------

def save_llm_rationales(result: dict[str, Any],
                        out_path: Path | None = None) -> Path:
    out_path = out_path or DEFAULT_LLM_RATIONALES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return out_path


def flag_invalid_paper_ids(
    result: dict[str, Any],
    valid_paper_ids: set[str],
) -> list[dict[str, Any]]:
    """Post-hoc check (NOT a validator failure): list every gap whose
    `closest_existing_work` contains a paper_id NOT in the corpus.

    Returns a list of `{gap_id, invalid_paper_ids}` dicts — empty if all good.
    The prompt forbids invented paper_ids, but if the LLM does it anyway we
    surface it for human review rather than crashing.
    """
    out: list[dict[str, Any]] = []
    sentinel = "no exact match in corpus"
    for gap in result.get("gaps", []):
        invalid = [
            p for p in gap.get("closest_existing_work", [])
            if isinstance(p, str) and p != sentinel and p not in valid_paper_ids
        ]
        if invalid:
            out.append({"gap_id": gap.get("id"), "invalid_paper_ids": invalid})
    return out
