"""SAVANT UI — Knowledge graph.

Embeds the interactive pyvis HTML + the 3 static manuscript figures + a
KG-stats panel that mirrors the A3 build summary.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from savant.ui.components import (
    apply_theme, footer, hero, section_eyebrow, stat_grid, stat_tile,
)
from savant.ui.data import (
    REPO, corpus_facts, figure_paths, kg_html_path, load_kg,
)

st.set_page_config(
    page_title="SAVANT — Knowledge Graph",
    page_icon="🕸",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Stage 2 · The knowledge graph",
    title="Walk the structure of the field — methods, papers, datasets, frameworks.",
    subtitle=(
        "Every paper in the corpus is wired into a typed graph: papers cite "
        "papers, methods value action types, methods use datasets, methods "
        "fit use cases. Drag nodes, click for full attributes, filter "
        "groups in the legend — the structure is queryable."
    ),
)


# --- KG headline stats ----------------------------------------------------
facts = corpus_facts()
stat_grid([
    stat_tile(facts["n_kg_nodes"], "Nodes",
              sub=f"{facts['n_methods']} methods · {facts['n_authors']} authors"),
    stat_tile(facts["n_kg_edges"], "Typed edges",
              sub="14 distinct edge types"),
    stat_tile(facts["n_papers_total"], "Paper nodes",
              sub=f"{facts['n_candidates']} candidate · {facts['n_heldout']} held-out"),
    stat_tile("v1.0", "Schema",
              sub="see src/savant/knowledge_graph/README.md"),
])


# --- Tabs: interactive embed first, static figures second -----------------
tab_interactive, tab_figures, tab_breakdowns = st.tabs([
    "🎮  Interactive graph",
    "🖼  Manuscript figures",
    "📊  Node / edge breakdown",
])


# --- TAB: interactive embed ----------------------------------------------
with tab_interactive:
    section_eyebrow("Click · drag · zoom · filter")
    html_path = kg_html_path()
    if not html_path.exists():
        st.error(
            f"Interactive graph not found at `{html_path}`. Run "
            "`PYTHONPATH=src python scripts/run_gap_analysis.py` to regenerate."
        )
    else:
        st.markdown(
            '<div style="color:#404040; margin-bottom:0.8rem; font-size:1.02rem; line-height:1.55;">'
            'The legend (top-left) toggles node groups. Authors and venues '
            'are hidden by default to reduce visual crowding — re-enable '
            'them with their checkbox.'
            '</div>',
            unsafe_allow_html=True,
        )
        html_text = html_path.read_text()
        components.html(html_text, height=820, scrolling=False)

        st.download_button(
            "Download the interactive HTML (self-contained, share by email)",
            data=html_text,
            file_name="savant_kg.html",
            mime="text/html",
        )


# --- TAB: static figures --------------------------------------------------
with tab_figures:
    paths = figure_paths()

    section_eyebrow("Three manuscript figures")
    st.markdown(
        '<div class="savant-prose" style="margin-bottom:1rem;">'
        'These are the three figures used in the manuscript. Each is '
        'a deterministic render of the same knowledge graph above. '
        '300 dpi; safe to drop into LaTeX or slides.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Figure 1 — Lineage
    st.markdown("### Lineage of soccer AV methods")
    st.caption("Y axis = publication year. Edges colour-coded by relationship type. "
               "Only papers participating in at least one in-corpus citation edge are shown.")
    if paths["lineage"].exists():
        st.image(str(paths["lineage"]), use_container_width=True)
        with paths["lineage"].open("rb") as fh:
            st.download_button(
                "Download lineage figure (PNG)", data=fh.read(),
                file_name="kg_lineage.png", mime="image/png",
                key="dl_lineage",
            )
    else:
        st.warning("kg_lineage.png missing — run scripts/run_gap_analysis.py")

    st.markdown("---")

    # Figure 2 — Coverage heatmap
    st.markdown("### Action coverage by method")
    st.caption("Rows grouped by valuation_unit [A]/[S]/[B]. Cell colour = methodological framework. "
               "The empty cells in the goalkeeper / off-ball / set-pieces columns are the gap-analysis input.")
    if paths["coverage"].exists():
        st.image(str(paths["coverage"]), use_container_width=True)
        with paths["coverage"].open("rb") as fh:
            st.download_button(
                "Download coverage heatmap (PNG)", data=fh.read(),
                file_name="kg_coverage_heatmap.png", mime="image/png",
                key="dl_coverage",
            )

    st.markdown("---")

    # Figure 3 — Dataset overlay
    st.markdown("### Datasets used by method")
    st.caption("Green = public dataset (reproducible). Grey = private / proprietary. "
               "The near-monochrome grey panel is the reproducibility wall — almost the entire corpus depends on private data.")
    if paths["dataset"].exists():
        st.image(str(paths["dataset"]), use_container_width=True)
        with paths["dataset"].open("rb") as fh:
            st.download_button(
                "Download dataset overlay (PNG)", data=fh.read(),
                file_name="kg_dataset_overlay.png", mime="image/png",
                key="dl_dataset",
            )


# --- TAB: breakdowns ------------------------------------------------------
with tab_breakdowns:
    section_eyebrow("What's inside")

    g = load_kg()
    node_counts = Counter(
        d.get("node_type") for _, d in g.nodes(data=True) if d.get("node_type")
    )
    edge_counts = Counter(
        d.get("edge_type") for _, _, d in g.edges(data=True) if d.get("edge_type")
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("#### Nodes by type")
        node_df = pd.DataFrame(
            sorted(node_counts.items(), key=lambda x: -x[1]),
            columns=["node_type", "count"],
        )
        st.dataframe(node_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Edges by type")
        edge_df = pd.DataFrame(
            sorted(edge_counts.items(), key=lambda x: -x[1]),
            columns=["edge_type", "count"],
        )
        st.dataframe(edge_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="savant-card" style="margin-top:1.5rem;">'
        '<div class="savant-card-title">For developers</div>'
        '<h3>Where does this graph come from?</h3>'
        '<p style="color:#404040; line-height:1.6;">'
        'The KG is built deterministically from the 24 candidate '
        'extraction JSONs under <code>data/outputs/extractions/</code> '
        'by <code>savant.knowledge_graph.build.build_kg()</code>. The '
        '10 node types and 14 edge types are documented in '
        '<code>src/savant/knowledge_graph/README.md</code>. The '
        'paper-to-paper relationship edges (EXTENDS / IMPROVES_UPON / '
        'COMPARES_AGAINST / CRITIQUES) were resolved by hand from each '
        "paper&#39;s baselines list — see "
        '<code>scripts/populate_relationships.py</code>.'
        '</p></div>',
        unsafe_allow_html=True,
    )

footer()
