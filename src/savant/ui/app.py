"""SAVANT — Streamlit UI entry point (Home / Overview).

Launch with:
    PYTHONPATH=src streamlit run src/savant/ui/app.py
or use scripts/run_ui.sh.

Streamlit auto-discovers files under `src/savant/ui/pages/` as additional
pages in the sidebar nav. This file is the Home page.
"""

from __future__ import annotations

import streamlit as st

from savant.ui.components import (
    apply_theme, badges, card, footer, hero, section_eyebrow,
    stat_grid, stat_tile,
)
from savant.ui.data import corpus_facts

# --- Page config (MUST be the first Streamlit call) ----------------------
st.set_page_config(
    page_title="SAVANT — Soccer Action Valuation",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# --- Sidebar branding -----------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.4rem 0 1.2rem;">
          <div style="font-size: 1.4rem; font-weight: 700;
                      letter-spacing: -0.02em; color: #06A77D;">
            SAVANT
          </div>
          <div style="font-size: 0.78rem; color: #6B7280;
                      margin-top: 0.2rem; line-height: 1.4;">
            Soccer Action Valuation —<br/>an agentic research system
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Hero -----------------------------------------------------------------
hero(
    eyebrow="An Agentic Framework for Action Valuation in Soccer",
    title="A knowledge graph + benchmark for every action-valuation method that matters.",
    subtitle=(
        "SAVANT is a reproducible research system that reviews the soccer "
        "action-valuation literature, builds a queryable knowledge graph, "
        "identifies under-explored research gaps, and benchmarks methods on "
        "open data — with every claim grounded in a citable paper."
    ),
)

# --- Headline stats -------------------------------------------------------
facts = corpus_facts()
stat_grid([
    stat_tile(facts["n_papers_total"], "Papers in the corpus",
              sub=f"{facts['n_candidates']} candidate · {facts['n_heldout']} held-out"),
    stat_tile(facts["n_methods"], "Methods catalogued",
              sub=f"one per candidate paper, {facts['years_min']}–{facts['years_max']}"),
    stat_tile(facts["n_kg_nodes"], "Knowledge-graph nodes",
              sub=f"{facts['n_kg_edges']} typed edges"),
    stat_tile(facts["n_actionable_gaps"], "Actionable research gaps",
              sub=f"of {facts['n_ranked_gaps']} empty cells in the 3-D matrix"),
    stat_tile(facts["n_methods_on_public_data"], "Methods on public data",
              sub=f"vs {facts['n_methods'] - facts['n_methods_on_public_data']} private-only"),
    stat_tile(facts["n_methods_with_code"], "Methods with code",
              sub=f"of {facts['n_methods']} total"),
])

# --- What SAVANT does (three cards in a row) ------------------------------
section_eyebrow("What this system does")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    card(
        title="Reads the literature",
        eyebrow="Stage 1 — Synthesis",
        body_md=(
            "A multi-agent extraction pipeline parses every paper into a "
            "structured record with **52 fields** covering model design, "
            "data, evaluation, action coverage, code availability, and "
            "use cases. Every record is schema-validated and "
            "citation-grounded — no field is set unless the paper "
            "actually says so."
        ),
    )
with c2:
    card(
        title="Finds the gaps",
        eyebrow="Stage 2 — Discovery",
        body_md=(
            "A knowledge graph of **165 nodes + 476 edges** projects the "
            "literature into framework × valuation-unit × action-type "
            "cells. An algorithmic ranker scores the empty cells by "
            "methodological reach, reproducibility, and practitioner "
            "relevance — surfacing the gaps worth working on."
        ),
    )
with c3:
    card(
        title="Benchmarks methods",
        eyebrow="Stage 3 — Validation",
        body_md=(
            "Reference implementations of xT, VAEP, EPV, and other "
            "methods are evaluated on **StatsBomb Open / 360** under a "
            "shared protocol designed by an agent that translates the "
            "ranked gaps into executable benchmark tasks. Calibration "
            "and uncertainty are first-class."
        ),
    )

# --- For your audience (two columns) --------------------------------------
section_eyebrow("Built for two audiences at once")

a1, a2 = st.columns(2, gap="large")
with a1:
    card(
        title="If you're a researcher",
        eyebrow="Academic use",
        body_md=(
            "- Drop in to find the **methodological lineage** of any "
            "soccer AV method (xT → defensive-xT → multi-agent variants).\n"
            "- Spot a gap that *actually has a public-data path* and turn "
            "it into a publishable study.\n"
            "- Compare two methods on a shared evaluation protocol — "
            "no more apples-to-oranges paper claims."
        ),
    )
with a2:
    card(
        title="If you're a club data scientist",
        eyebrow="Industrial use",
        body_md=(
            "- Filter methods by what data you actually have (event-only? "
            "tracking? StatsBomb-Open?).\n"
            "- Compare interpretability, real-time capability, and code "
            "release before investing engineer-weeks.\n"
            "- See which method covers the action types you care about "
            "(defensive, off-ball, set-pieces, goalkeeper)."
        ),
    )

# --- Quick tour callout ---------------------------------------------------
st.markdown(
    f"""
    <div class="savant-card" style="
        background: linear-gradient(135deg, #F5F7F6 0%, #FFFFFF 100%);
        border-color: rgba(6,167,125,0.25); margin-top: 1rem;
    ">
      <div class="savant-card-title">Take the tour</div>
      <h3 style="margin-bottom: 0.6rem;">Start anywhere — the sidebar maps to the pipeline.</h3>
      <p style="color: #404040; line-height: 1.6; margin-bottom: 0.8rem;">
        Every page below shows the <em>actual artifact</em> SAVANT produces at
        that stage. Browse the corpus to see what's in. Walk the knowledge
        graph to feel the structure. Open the gap analysis to see where to
        invest research effort. Chat with the corpus to ask anything in plain
        English. Run the live demo to watch the critic agent in real time.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
badges([
    ("Schema v0.3.0", "primary"),
    ("Open data", "primary"),
    ("Python 3.11", "muted"),
    ("MIT-licensed", "muted"),
    ("Reproducible", "accent"),
])

footer()
