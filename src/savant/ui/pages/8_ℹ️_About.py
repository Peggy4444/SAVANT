"""SAVANT UI — About / Methods.

Final page: what the system is, how the schema is structured, how to cite,
and where to find the code.
"""

from __future__ import annotations

import streamlit as st

from savant.ui.components import (
    apply_theme, badges, footer, hero, section_eyebrow, stat_grid, stat_tile,
)
from savant.ui.data import corpus_facts

st.set_page_config(
    page_title="SAVANT — About",
    page_icon="ℹ️",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="The project",
    title="About SAVANT.",
    subtitle=(
        "What the system is, how it's built, what schema each record satisfies, "
        "how to reproduce the artifacts, and how to cite the work in your own paper."
    ),
)


# --- Headline -------------------------------------------------------------
facts = corpus_facts()
stat_grid([
    stat_tile(facts["n_papers_total"], "Papers in the corpus",
              sub=f"{facts['n_candidates']} candidate · {facts['n_heldout']} held-out"),
    stat_tile("v0.3.0", "Schema version",
              sub="52 fields per record"),
    stat_tile("4", "Agents implemented",
              sub="extraction · critic · taxonomy · gap"),
    stat_tile("3.11", "Python",
              sub="MIT-licensed, open source"),
])


# --- What is SAVANT -------------------------------------------------------
section_eyebrow("The project")

st.markdown(
    """
    <div class='savant-prose' style='max-width:880px;'>

    <strong>SAVANT</strong> — <strong>S</strong>occer <strong>A</strong>ction
    <strong>V</strong>aluation: an <strong>A</strong>gentic framework for
    k<strong>N</strong>owledge synthesis, gap discovery, and benchmarking of
    <strong>T</strong>ransparent action-value models.

    <p>
      Soccer action valuation (xT, VAEP, EPV, possession-value models, etc.)
      has exploded over the last decade, but the literature is fragmented
      across analytics blogs, conference proceedings, and journal papers.
      There is no single, machine-readable map of <em>what has been tried,
      on what data, under what assumptions, and how the methods compare on
      a common benchmark</em>.
    </p>

    <p>
      SAVANT tries to fix that — with a reproducible, agentic research
      pipeline built around three coupled ideas:
    </p>

    <ol>
      <li><strong>Agentic literature synthesis.</strong> A team of LLM agents
      (screening, extraction, critic, taxonomy) ingests a curated paper corpus
      and produces a structured database describing each model: inputs,
      target, supervision signal, evaluation, dataset, claims, limitations.</li>

      <li><strong>Knowledge graph + gap analysis.</strong> Extracted records
      are projected into a knowledge graph linking methods → assumptions →
      evaluation protocols → datasets. A gap-analysis agent walks the graph
      to surface under-explored regions ("no published method values
      goalkeeper actions via reinforcement learning").</li>

      <li><strong>Reproducible empirical benchmark</strong> <em>(planned)</em>.
      A reference implementation of xT, VAEP, EPV, and other methods will
      be evaluated on StatsBomb Open Data (and StatsBomb 360 where licence
      permits) under a shared evaluation protocol designed by the
      benchmark agent.</li>
    </ol>

    </div>
    """,
    unsafe_allow_html=True,
)


# --- Schema ---------------------------------------------------------------
section_eyebrow("Schema v0.3.0 — what we capture per paper")

st.markdown(
    """
    Each paper is parsed into a record validated against
    `configs/extraction_schema.json` (v0.3.0). The schema groups fields
    by audience, so the same record serves both a survey author and a
    club data scientist:
    """
)

sc1, sc2 = st.columns(2, gap="large")
with sc1:
    st.markdown(
        """
        **Method block** *(the methodology core)*
        - `family`, `framework`, `valuation_unit` (action / state / both)
        - `inputs`, `target`, `supervision`, `architecture`
        - `spatial_resolution`, `temporal_horizon`, `atomicity`
        - **`action_coverage`** — 7 booleans for passes / shots / dribbles /
          defensive / off-ball / set-pieces / goalkeeper
        - **`outputs`** — `value_scale` + `aggregation_levels`

        **Data block**
        - `provider`, `data_type`, **`dataset_canonical`** (controlled enum)
        - `leagues`, `seasons`, `n_matches`, `is_public`
        """
    )
with sc2:
    st.markdown(
        """
        **Evaluation block**
        - `metrics`, `tasks`, `baselines`, `splits`
        - `calibration_reported`, `uncertainty_reported`
        - **`validated_against_expert_judgment`**, **`bootstrap_or_ci_reported`**

        **Practical block** *(the club-data-scientist lens)*
        - `use_cases`, `interpretability_level`, `compute_intensity`
        - `inference_realtime_capable`

        **Knowledge-graph block**
        - **`relationships`** — typed edges (`extends` / `compares-against` /
          `improves-upon` / `uses-dataset-of` / `critiques`)
        - `claims` — each with `evidence_location` (section / table)
        - `limitations`, `open_questions`, `code_release`
        """
    )

st.markdown("&nbsp;")
st.markdown(
    """
    > **Why v0.3.0 matters.** Earlier schemas (v0.1.0 / v0.2.0) lacked the
    > `action_coverage` granularity, the `practical` block, and explicit
    > typed `relationships`. Without those, gap analysis cannot say
    > *"no MDP method values goalkeeper actions"* and the KG cannot render
    > a lineage diagram. v0.3.0 is the version that makes both possible.
    """
)


# --- How to use ----------------------------------------------------------
section_eyebrow("How to use this system")

st.markdown(
    """
    | If you want to… | Go to |
    |---|---|
    | Browse the corpus, inspect any paper's full record | **📚 Corpus** |
    | See the methodological structure of the field | **🕸 Knowledge Graph** |
    | Spot a gap visually | **🔥 Coverage** |
    | Pick a research direction from the ranked gap list | **🎯 Gaps** |
    | Understand the LLM-agent layer | **🤖 Agentic Pipeline** |
    | Run the agents on your own paper, your own data, or your own dime | **⚡ Live Demo** |
    | Ask plain-English questions, get citation-grounded answers | **💬 Chat** |
    """
)


# --- Reproducibility -----------------------------------------------------
section_eyebrow("Reproducibility")

st.markdown(
    """
    Every artifact in this UI is regenerable from the 24 candidate JSON
    records in `data/outputs/extractions/` via deterministic scripts.

    | What | Script |
    |---|---|
    | Knowledge graph (`savant_kg.json`, `.graphml`, `.html`), 3 figures, enumeration, ranking | `scripts/run_gap_analysis.py` |
    | Agentic pipeline (critic / taxonomy / gap-agent in mock or live mode) | `scripts/run_agentic_pipeline.py` |
    | Build Phase-1 CSVs from extraction JSONs | `scripts/build_phase1_csvs.py` |
    | Populate paper-to-paper relationships in extractions | `scripts/populate_relationships.py` |
    | Single-paper live LLM demo | `scripts/demo_live_critic.py` |

    All scripts read from `configs/` (paths, agents, benchmark tasks,
    extraction schema). Models, temperatures, and retry policies live in
    `configs/agents.yaml` — never hardcoded.
    """
)


# --- Cite this work ------------------------------------------------------
section_eyebrow("Cite this work")

st.markdown(
    """
    SAVANT is being submitted to **Machine Learning with Applications**.
    For now, please cite the repository:
    """
)

st.code(
    """@misc{rahimian2026savant,
  title  = {{SAVANT}: An Agentic Framework for Knowledge Synthesis,
            Gap Discovery, and Benchmarking of Soccer Action Valuation Models},
  author = {Rahimian, Pegah and others},
  year   = {2026},
  note   = {Manuscript in preparation.
            See \\url{https://github.com/<your-username>/savant-action-valuation}}
}""",
    language="bibtex",
)


# --- Acknowledgements ----------------------------------------------------
section_eyebrow("Acknowledgements")

st.markdown(
    """
    SAVANT is being developed by:

    - **Pegah Rahimian** — research lead, schema design, corpus curation,
      knowledge graph + gap analysis methodology, manuscript lead.
    - **Axum** — first-pass paper annotator for the literature-review database.
    - **David Sumpter** — methodological second-reader and verifier for
      the gold-standard subset.

    Built on top of the open soccer-analytics community's prior work — see
    the corpus for the 24+ papers we synthesised. We particularly acknowledge
    the maintainers of the [`socceraction`](https://github.com/ML-KULeuven/socceraction)
    library, StatsBomb's [Open Data](https://github.com/statsbomb/open-data),
    [SkillCorner](https://github.com/SkillCorner/opendata), and
    [Metrica Sports](https://github.com/metrica-sports/sample-data) for
    making AV research reproducible.
    """
)


# --- Links + footer ------------------------------------------------------
section_eyebrow("Links")

badges([
    ("GitHub repo", "primary"),
    ("Manuscript draft (in prep.)", "muted"),
    ("MIT licence", "muted"),
    ("Schema v0.3.0", "primary"),
    ("Python 3.11", "muted"),
])

st.caption(
    "GitHub URL goes here once the repo is public. "
    "Contact: pegah dot rahimian at (institution) for collaboration."
)

footer()
