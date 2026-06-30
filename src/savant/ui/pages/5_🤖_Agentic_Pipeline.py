"""SAVANT UI — Agentic pipeline.

Demystifies the LLM-agent layer:
- Mermaid architecture diagram of the agent flow
- Per-agent prompt + input shape + output schema (3 tabs)
- "Run mock pipeline" button that executes the orchestrator end-to-end
- "Under the hood" pointers to the code
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from savant.ui.components import (
    apply_theme, footer, hero, section_eyebrow, stat_grid, stat_tile,
)
from savant.ui.data import REPO

st.set_page_config(
    page_title="SAVANT — Agentic pipeline",
    page_icon="🤖",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Stage 2 · Agentic loop",
    title="How SAVANT actually reads the literature.",
    subtitle=(
        "An LLM extraction agent parses each PDF, a critic agent re-verifies "
        "the result, a taxonomy agent canonicalises free-text fields, and a "
        "gap agent turns the algorithmic ranking into manuscript-quality "
        "prose. Every call is logged, validated against a JSON schema, and "
        "re-runnable in mock mode without spending tokens."
    ),
)


# --- Mermaid flow diagram --------------------------------------------------
section_eyebrow("The flow at a glance")

mermaid_diagram = """
flowchart LR
    PDFS[Paper PDFs<br/><i>24 candidate + 4 held-out</i>]:::data
    EX([Extraction agent]):::agent
    REC[Structured JSON<br/>extraction_schema.json v0.3.0]:::data
    CRIT([Critic agent]):::agent
    KG[Knowledge graph<br/>165 nodes / 476 edges]:::data
    TAX([Taxonomy agent]):::agent
    CANON[Canonical fields<br/>data/outputs/taxonomy/]:::data
    ENUM[Coverage matrices<br/>enumeration.json]:::data
    RANK[Algorithmic ranker<br/>ranked.json · 62 actionable gaps]:::data
    GAP([Gap agent]):::agent
    OUT[Manuscript rationales<br/>llm_rationales.json]:::data
    PDFS --> EX
    EX --> REC
    REC --> CRIT
    CRIT -->|accept| KG
    CRIT -->|revise| EX
    KG --> ENUM
    KG --> TAX
    TAX --> CANON
    ENUM --> RANK
    RANK --> GAP
    KG -.context.-> GAP
    GAP --> OUT
    classDef data fill:#FFFFFF,stroke:#06A77D,stroke-width:2px,color:#1A1A1A;
    classDef agent fill:#06A77D,stroke:#048864,color:#FFFFFF,stroke-width:2px;
"""

components.html(
    f"""
    <div id="mermaid-host" style="
        background:#FAFAFA;
        padding:1.4rem 1rem;
        border:1px solid #E5E7EB;
        border-radius:14px;">
      <div class="mermaid" style="text-align:center;">{mermaid_diagram}</div>
    </div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{
        startOnLoad: true,
        theme: "default",
        fontFamily: "Inter, sans-serif",
        themeVariables: {{
          primaryColor: "#06A77D",
          primaryTextColor: "#FFFFFF",
          primaryBorderColor: "#048864",
          lineColor: "#404040",
          textColor: "#1A1A1A",
          fontSize: "14px",
        }},
        flowchart: {{ htmlLabels: true, curve: "basis", nodeSpacing: 50, rankSpacing: 60 }},
      }});
    </script>
    """,
    height=380,
    scrolling=False,
)


# --- Headline stats -------------------------------------------------------
n_prompts = len(list((REPO / "prompts").glob("*.md")))
stat_grid([
    stat_tile("3", "Agents implemented",
              sub="critic · taxonomy · gap"),
    stat_tile(n_prompts, "Prompt files",
              sub="prompts/*.md — source of truth"),
    stat_tile("0", "API tokens to try",
              sub="mock mode is free + deterministic"),
    stat_tile("0.02s", "Mock-mode runtime",
              sub="full pipeline end-to-end"),
])


# --- Per-agent details (3 tabs) -------------------------------------------
st.markdown("&nbsp;")
section_eyebrow("Per-agent details")

CRITIC_INPUT = {
    "agent": "extraction",
    "output": "{the extraction record JSON}",
    "source": "{PDF excerpt or chunk reference}",
    "schema": "configs/extraction_schema.json v0.3.0",
}
CRITIC_OUTPUT = {
    "verdict": "accept | revise | reject",
    "schema_conformant": True,
    "factual_issues": [{"field": "...", "issue": "...", "evidence": "..."}],
    "completeness_issues": [{"field": "...", "issue": "..."}],
    "style_issues": [],
    "revision_instructions": "<one paragraph>",
    "notes": "<optional commentary>",
}

TAXONOMY_INPUT = {
    "field": "evaluation.metrics",
    "values": ["ROC-AUC", "AUROC", "Brier score", "F1"],
    "controlled_vocabulary": ["roc_auc", "brier_score"],
    "examples": [{"paper_id": "decroos_2019_vaep", "raw": "ROC-AUC"}],
}
TAXONOMY_OUTPUT = {
    "field": "evaluation.metrics",
    "mappings": [
        {"raw": "ROC-AUC", "canonical": "roc_auc",     "confidence": 1.0},
        {"raw": "AUROC",   "canonical": "roc_auc",     "confidence": 1.0},
        {"raw": "Brier score", "canonical": "brier_score", "confidence": 1.0},
    ],
    "new_vocabulary": [{"canonical": "expected_calibration_error", "definition": "..."}],
    "unresolved": [{"raw": "weird metric", "reason": "not enough context"}],
}

GAP_INPUT = {
    "coverage_summary": "{corpus_size, frameworks_with_method_counts, …}",
    "algorithmic_ranking": "[top-K gaps from ranked.json]",
    "corpus_papers": "[{paper_id, title, year, method_name, framework, valuation_unit}]",
    "dataset_facts": "{n_methods_using_public_dataset, n_methods_with_code_released, …}",
    "author_open_questions": "[paper-id-tagged future-work statements]",
}
GAP_OUTPUT = {
    "gaps": [
        {
            "id": "gap-001",
            "title": "Per-action MDP value methods are absent",
            "consolidated_auto_ranks": [{"framework": "...", "auto_rank": 1, "auto_score": 2.79}],
            "the_gap": "…",
            "scientific_motivation": "…",
            "practitioner_motivation": "…",
            "closest_existing_work": ["decroos_2019_vaep"],
            "suggested_first_study": "Reuse Van Roy 2021's StatsBomb-open MDP … (must be feasible on PUBLIC data)",
            "novelty_score": 0.75,
            "feasibility_score": 0.85,
            "confidence": 0.8,
        }
    ],
    "agent_notes": "Consolidated the top six MDP × action cells into one gap.",
}


def _render_agent_tab(
    *, prompt_filename: str, role: str, when_invoked: str,
    input_shape: dict, output_shape: dict,
) -> None:
    st.markdown(f"**Role.** {role}")
    st.markdown(f"**When invoked.** {when_invoked}")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Input shape**")
        st.code(json.dumps(input_shape, indent=2), language="json")
    with c2:
        st.markdown("**Output shape**")
        st.code(json.dumps(output_shape, indent=2), language="json")

    prompt_path = REPO / "prompts" / prompt_filename
    if prompt_path.exists():
        with st.expander(f"Full prompt — `prompts/{prompt_filename}`", expanded=False):
            st.markdown(prompt_path.read_text())
    else:
        st.warning(f"Prompt file `{prompt_filename}` not found.")


tab_crit, tab_tax, tab_gap = st.tabs([
    "🔍  Critic agent",
    "🏷  Taxonomy agent",
    "🎯  Gap agent",
])

with tab_crit:
    _render_agent_tab(
        prompt_filename="critic_agent.md",
        role=(
            "Sceptical reviewer. Given an extraction record and its source "
            "PDF excerpt, decides whether the record is faithful, complete, "
            "and schema-conformant, then returns a structured verdict + "
            "actionable issues."
        ),
        when_invoked=(
            "After every extraction-agent run, before the record is "
            "committed to `data/outputs/extractions/`. Patches "
            "`provenance.critic` in the record."
        ),
        input_shape=CRITIC_INPUT,
        output_shape=CRITIC_OUTPUT,
    )

with tab_tax:
    _render_agent_tab(
        prompt_filename="taxonomy_agent.md",
        role=(
            "Normaliser. Given a deduplicated list of free-text values for "
            "one field across the corpus (e.g. all `evaluation.metrics` strings), "
            "returns a canonical mapping plus optional new vocabulary."
        ),
        when_invoked=(
            "Once per controlled-vocabulary-eligible field (evaluation.metrics, "
            "model.architecture, evaluation.baselines, data.provider) after "
            "extraction is done. Outputs land in `data/outputs/taxonomy/`."
        ),
        input_shape=TAXONOMY_INPUT,
        output_shape=TAXONOMY_OUTPUT,
    )

with tab_gap:
    _render_agent_tab(
        prompt_filename="gap_agent.md",
        role=(
            "Survey author. Reads the coverage statistics + algorithmic ranking "
            "+ corpus summary, then produces a manuscript-quality consolidated "
            "list of gaps (replacing or complementing the hand-written rationale)."
        ),
        when_invoked=(
            "Once per pipeline run, after the algorithmic ranker (A6). "
            "Output is `data/outputs/gaps/llm_rationales.json` — the LLM "
            "v2 of `top5_rationale.md`."
        ),
        input_shape=GAP_INPUT,
        output_shape=GAP_OUTPUT,
    )


# --- Run-it-live (mock mode) -----------------------------------------------
st.markdown("&nbsp;")
section_eyebrow("Run the whole pipeline in mock mode")

st.markdown(
    "<div style='color:#404040; line-height:1.55; margin-bottom:0.8rem;'>"
    "Mock mode uses canned canonical responses — no Anthropic API key, no "
    "tokens, no spend. It validates the orchestrator end-to-end and writes "
    "real output files. For the actual LLM, head to the "
    "<strong>Live demo</strong> page.</div>",
    unsafe_allow_html=True,
)


def _load_orchestrator():
    """Dynamically load scripts/run_agentic_pipeline.py as a module."""
    script_path = REPO / "scripts" / "run_agentic_pipeline.py"
    if "run_agentic_pipeline_mod" in sys.modules:
        return sys.modules["run_agentic_pipeline_mod"]
    spec = importlib.util.spec_from_file_location(
        "run_agentic_pipeline_mod", script_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_agentic_pipeline_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


if st.button("▶  Run mock pipeline", type="primary"):
    with st.spinner("Running critic → taxonomy → gap-agent in mock mode…"):
        t0 = time.perf_counter()
        try:
            orch = _load_orchestrator()
            summary = orch.run(
                mode="mock",
                write=False,  # never modify gold standard from the UI
                taxonomy_out_dir=REPO / "data" / "outputs" / "taxonomy",
                gap_out_path=REPO / "data" / "outputs" / "gaps" / "llm_rationales.json",
            )
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    st.success(f"✓ Pipeline completed in {elapsed:.2f}s")

    cc1, cc2, cc3 = st.columns(3, gap="medium")
    with cc1:
        critic = summary.get("critic", {})
        st.markdown(
            f"<div class='savant-card'><div class='savant-card-title'>Critic</div>"
            f"<h3>{sum(critic.values())} verdicts</h3>"
            f"<div style='color:#6B7280;'>accept={critic.get('accept',0)} · "
            f"revise={critic.get('revise',0)} · reject={critic.get('reject',0)}</div>"
            f"<div style='color:#6B7280; font-size:0.82rem; margin-top:0.4rem;'>"
            f"(dry-run — extraction JSONs NOT modified)</div></div>",
            unsafe_allow_html=True,
        )
    with cc2:
        n_taxonomy = len(summary.get("taxonomy_files", []))
        st.markdown(
            f"<div class='savant-card'><div class='savant-card-title'>Taxonomy</div>"
            f"<h3>{n_taxonomy} field maps</h3>"
            f"<div style='color:#6B7280;'>written to data/outputs/taxonomy/</div></div>",
            unsafe_allow_html=True,
        )
    with cc3:
        gap_info = summary.get("gap", {})
        st.markdown(
            f"<div class='savant-card'><div class='savant-card-title'>Gap</div>"
            f"<h3>{gap_info.get('n_gaps','—')} consolidated</h3>"
            f"<div style='color:#6B7280;'>written to llm_rationales.json</div></div>",
            unsafe_allow_html=True,
        )

    # Show one sample taxonomy output
    sample_paths = [Path(p) for p in summary.get("taxonomy_files", [])][:1]
    if sample_paths:
        with st.expander(f"Sample taxonomy output — `{sample_paths[0].name}`",
                         expanded=False):
            st.code(sample_paths[0].read_text()[:3000] + " …", language="json")

    # Show the gap output
    gap_path = REPO / "data" / "outputs" / "gaps" / "llm_rationales.json"
    if gap_path.exists():
        with st.expander(f"LLM gap rationale output — `llm_rationales.json`",
                         expanded=False):
            st.code(gap_path.read_text(), language="json")


# --- Under the hood -------------------------------------------------------
st.markdown("&nbsp;")
section_eyebrow("Under the hood")

st.markdown(
    """
    | Concern | Where it lives |
    |---|---|
    | Agent prompts | `prompts/critic_agent.md`, `taxonomy_agent.md`, `gap_agent.md` |
    | Per-agent model + temperature + retry | `configs/agents.yaml` |
    | LLM client (Protocol + Anthropic + Mock) | `src/savant/agents/client.py` |
    | Generic agent-runner base | `src/savant/agents/runner.py` |
    | Critic runner | `src/savant/extraction/critic.py` |
    | Taxonomy runner | `src/savant/taxonomy/runner.py` |
    | Gap-agent runner | `src/savant/gap_analysis/gap_agent.py` |
    | Orchestrator CLI | `scripts/run_agentic_pipeline.py` |
    | Call logs (JSONL) | `data/outputs/llm_logs/{YYYY-MM-DD}.jsonl` |
    """
)

footer()
