"""SAVANT UI — Gap analysis.

Three views over the gap-analysis artifacts:
1. Algorithmic ranking (A6 — `ranked.json`)
2. Manuscript-quality consolidated rationale (A7 — `top5_rationale.md`)
3. Cross-reference: which auto-ranked cells each manuscript gap covers
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from savant.gap_analysis.enumerate import (
    ACTION_TYPES, FRAMEWORKS, VALUATION_UNITS,
)
from savant.ui.components import (
    apply_theme, badges, footer, hero, section_eyebrow, stat_grid, stat_tile,
)
from savant.ui.data import load_ranking, load_top5_rationale_md

st.set_page_config(
    page_title="SAVANT — Gap analysis",
    page_icon="🎯",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Stage 2 · Gap discovery",
    title="The 62 research gaps that are actually worth working on.",
    subtitle=(
        "An algorithmic ranker scores every empty cell in the 3-D coverage "
        "matrix by methodological reach, reproducibility, and practitioner "
        "relevance. The hand-written rationale consolidates the top "
        "auto-ranked cells into manuscript-ready text — each with a "
        "feasible public-data study plan."
    ),
)


ranking = load_ranking()
gaps = ranking["ranked_gaps"]


# --- Headline stats -------------------------------------------------------
weights = ranking["weights"]
stat_grid([
    stat_tile(ranking["n_candidate_cells"], "Empty 3-D cells",
              sub="full pool of candidate gaps"),
    stat_tile(ranking["n_with_positive_methodological_score"],
              "Actionable gaps",
              sub="every axis is represented in corpus"),
    stat_tile(f"{weights['methodological']:.1f}",
              "Methodological weight",
              sub="primary — Pegah's design call"),
    stat_tile(f"{weights['reproducibility']:.1f}",
              "Reproducibility weight",
              sub="secondary — public-data bonus"),
    stat_tile(f"{weights['practitioner']:.1f}",
              "Practitioner weight",
              sub="tertiary — club-decision relevance"),
])


# --- Mapping: A7 manuscript gaps → A6 auto-ranks they consolidate --------
# (Extracted from data/outputs/gaps/top5_rationale.md.)
MANUSCRIPT_GAPS: list[dict] = [
    {
        "id": 1,
        "title": "MDP frameworks have not yet produced per-action values",
        "auto_ranks": [1, 2, 3, 4, 5, 6],
        "axis": "methodological",
    },
    {
        "id": 2,
        "title": "Expectation-based methods are structurally blind to off-ball actions",
        "auto_ranks": [9],
        "axis": "methodological",
    },
    {
        "id": 3,
        "title": "Goalkeepers are invisible to every framework except VAEP",
        "auto_ranks": [19],  # plus broader pattern at ranks 35–40
        "axis": "practitioner",
    },
    {
        "id": 4,
        "title": "Physics-based methods are confined to off-ball offence",
        "auto_ranks": [],  # ranks 39, 46, 52 — outside the table's default page
        "axis": "methodological",
    },
    {
        "id": 5,
        "title": "Reproducibility wall — tracking-dependent methods don't use public tracking",
        "auto_ranks": [],  # data-axis finding, not a 3-D cell
        "axis": "reproducibility",
    },
]


# --- Tabs ----------------------------------------------------------------
tab_alg, tab_ms, tab_xref = st.tabs([
    "📋  Algorithmic ranking (A6)",
    "📝  Manuscript rationale (A7)",
    "🔗  Cross-reference",
])


# ========================================================================
# TAB 1 — Algorithmic ranking
# ========================================================================
with tab_alg:
    section_eyebrow("Filter the 83 ranked empty cells")

    fcols = st.columns([1.2, 1.2, 1.2, 1.4], gap="small")
    with fcols[0]:
        fw_filter = st.multiselect(
            "Framework", options=list(FRAMEWORKS),
            default=[], key="alg_fw",
        )
    with fcols[1]:
        vu_filter = st.multiselect(
            "Valuation unit", options=list(VALUATION_UNITS),
            default=[], key="alg_vu",
        )
    with fcols[2]:
        at_filter = st.multiselect(
            "Action type", options=list(ACTION_TYPES),
            default=[], key="alg_at",
        )
    with fcols[3]:
        only_actionable = st.toggle(
            "Hide methodologically-empty cells (score=0)",
            value=True,
        )

    min_score = st.slider(
        "Minimum score", min_value=0.0,
        max_value=float(max(g["score"] for g in gaps)),
        value=0.0, step=0.1,
    )

    # Apply filters
    rows = []
    for g in gaps:
        if only_actionable and g["components"]["methodological_score"] == 0:
            continue
        if fw_filter and g["framework"] not in fw_filter: continue
        if vu_filter and g["valuation_unit"] not in vu_filter: continue
        if at_filter and g["action_type"] not in at_filter: continue
        if g["score"] < min_score: continue
        rows.append({
            "Rank": g["rank"],
            "Score": round(g["score"], 3),
            "Framework": g["framework"],
            "Valuation unit": g["valuation_unit"],
            "Action type": g["action_type"],
            "Methodological": round(g["components"]["methodological_score"], 2),
            "Reproducibility": int(g["components"]["reproducibility_bonus"]),
            "Practitioner": float(g["components"]["practitioner_bonus"]),
            "Nearby methods": len(g.get("nearby_methods") or []),
        })

    df = pd.DataFrame(rows)

    st.markdown(
        f"<div style='color:#06A77D; font-weight:600; margin: 0.4rem 0 0.6rem;'>"
        f"{len(df)} of {len(gaps)} gaps match</div>",
        unsafe_allow_html=True,
    )

    if len(df) == 0:
        st.info("No gaps match the current filters.")
    else:
        sel = st.dataframe(
            df, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "Rank": st.column_config.NumberColumn(width="small", format="%d"),
                "Score": st.column_config.NumberColumn(width="small", format="%.2f"),
                "Methodological": st.column_config.NumberColumn(width="small", format="%.2f"),
                "Reproducibility": st.column_config.NumberColumn(width="small", format="%d"),
                "Practitioner": st.column_config.NumberColumn(width="small", format="%.1f"),
                "Nearby methods": st.column_config.NumberColumn(width="small", format="%d"),
            },
            height=420,
        )

        # Drill-down
        st.markdown("---")
        if sel.selection.rows:
            picked_rank = int(df.iloc[sel.selection.rows[0]]["Rank"])
            gap = next(g for g in gaps if g["rank"] == picked_rank)
            section_eyebrow(f"#{gap['rank']} — score {gap['score']:.3f}")
            st.markdown(
                f"### `{gap['framework']}`  ×  `{gap['valuation_unit']}`  ×  `{gap['action_type']}`"
            )

            cscores = gap["components"]
            cc = st.columns(3, gap="small")
            with cc[0]:
                st.markdown(
                    f'<div class="savant-card"><div class="savant-card-title">Methodological</div>'
                    f'<h3>{cscores["methodological_score"]:.2f}</h3>'
                    f'<div style="color:#6B7280; font-size:0.85rem;">primary axis</div></div>',
                    unsafe_allow_html=True,
                )
            with cc[1]:
                st.markdown(
                    f'<div class="savant-card"><div class="savant-card-title">Reproducibility</div>'
                    f'<h3>{int(cscores["reproducibility_bonus"])}</h3>'
                    f'<div style="color:#6B7280; font-size:0.85rem;">public-data bonus</div></div>',
                    unsafe_allow_html=True,
                )
            with cc[2]:
                st.markdown(
                    f'<div class="savant-card"><div class="savant-card-title">Practitioner</div>'
                    f'<h3>{cscores["practitioner_bonus"]:.1f}</h3>'
                    f'<div style="color:#6B7280; font-size:0.85rem;">club-decision bonus</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("&nbsp;")
            st.markdown(f"**Auto-generated rationale.**")
            st.markdown(
                f"<div class='savant-prose'>{gap['rationale']}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("&nbsp;")
            nearby = gap.get("nearby_methods") or []
            if nearby:
                st.markdown(
                    f"**Closest existing work** (methods sharing 2 of 3 dimensions, "
                    f"{len(nearby)} total):"
                )
                badges([
                    (f"{n['paper_id']}  (matches {' + '.join(n['shared_dimensions'])})",
                     "primary")
                    for n in nearby
                ])
        else:
            st.info("Click any row in the table to see the full rationale + score breakdown.")


# ========================================================================
# TAB 2 — Manuscript rationale (A7)
# ========================================================================
with tab_ms:
    section_eyebrow("Hand-written manuscript draft")
    st.caption(
        "Five consolidated gaps for the manuscript's Open Problems section. "
        "Each picks one auto-ranked cell (or a cluster) and adds the scientific "
        "+ practitioner motivation plus a feasible public-data study suggestion."
    )

    md = load_top5_rationale_md()
    if md:
        st.markdown(md)
    else:
        st.warning(
            "`data/outputs/gaps/top5_rationale.md` not found. This is hand-curated "
            "and not auto-regenerated by the pipeline."
        )


# ========================================================================
# TAB 3 — Cross-reference
# ========================================================================
with tab_xref:
    section_eyebrow("Manuscript gap ↔ auto-ranked cells")

    st.markdown(
        '<div class="savant-prose" style="margin-bottom:1rem;">'
        'The hand-written rationale (tab 2) consolidates clusters of auto-ranked cells '
        '(tab 1) into one narrative gap each. This table makes the mapping explicit '
        'so anyone reading the manuscript can trace a claim back to the algorithmic '
        'ranking.'
        '</div>',
        unsafe_allow_html=True,
    )

    for mg in MANUSCRIPT_GAPS:
        consolidated_cells = [g for g in gaps if g["rank"] in mg["auto_ranks"]]

        with st.container():
            st.markdown(
                f'<div class="savant-card">'
                f'<div class="savant-card-title">Manuscript gap #{mg["id"]} · '
                f'<span style="color:#F18F01">{mg["axis"]} axis</span></div>'
                f'<h3>{mg["title"]}</h3>',
                unsafe_allow_html=True,
            )
            if consolidated_cells:
                cell_rows = pd.DataFrame([
                    {
                        "Auto rank": g["rank"],
                        "Score": round(g["score"], 2),
                        "Framework": g["framework"],
                        "Valuation unit": g["valuation_unit"],
                        "Action type": g["action_type"],
                    }
                    for g in consolidated_cells
                ])
                st.dataframe(cell_rows, hide_index=True, use_container_width=True,
                             height=min(280, 40 + 38 * len(cell_rows)))
            elif mg["axis"] == "reproducibility":
                st.info(
                    "This gap is on the data-availability axis — it surfaces from the "
                    "`dataset × action_type` projection, not the 3-D framework matrix, "
                    "so it has no entries in `ranked.json`."
                )
            else:
                st.info(
                    "This gap covers cells outside the table's top page "
                    "(ranks 35+, with methodological_score in the 1.6–2.1 range). "
                    "Adjust the filter on tab 1 to see them: set 'Hide methodologically-empty' off "
                    "and minimum score to 1.5."
                )
            st.markdown("</div>", unsafe_allow_html=True)

footer()
