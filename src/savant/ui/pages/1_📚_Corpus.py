"""SAVANT UI — Corpus browser.

28 papers (24 candidate + 4 held-out) with filter / search / row-click to
drill into the full extraction record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from savant.ui.components import (
    apply_theme, badges, card, footer, hero, section_eyebrow,
)
from savant.ui.data import (
    REPO, corpus_facts, load_extractions, load_papers_csv,
)

st.set_page_config(
    page_title="SAVANT — Corpus",
    page_icon="📚",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Stage 1 · Synthesis",
    title="The corpus — 28 soccer action-valuation methods, indexed.",
    subtitle=(
        "Every paper has been read and structured against the v0.3.0 SAVANT "
        "schema. Filter the list, click a row, and inspect every field we "
        "captured — model design, data provenance, evaluation, action "
        "coverage, code release, and downstream relationships."
    ),
)

facts = corpus_facts()
papers = load_papers_csv()
extractions = {r["paper_id"]: r for r in load_extractions()}


# --- Build the table (one row per paper, including held-out) --------------
def _row_for_paper(p: dict) -> dict:
    pid = p["paper_id"]
    ext = extractions.get(pid)
    framework = vu = action_cov = "—"
    if ext:
        framework = (ext.get("model") or {}).get("framework") or "—"
        vu = (ext.get("model") or {}).get("valuation_unit") or "—"
        ac = (ext.get("model") or {}).get("action_coverage") or {}
        action_cov = ", ".join(k for k, v in ac.items() if v) or "—"
    return {
        "paper_id":   pid,
        "year":       int(p["year"]) if p.get("year") else 0,
        "title":      p.get("title") or "—",
        "venue_type": p.get("venue_type") or "—",
        "primary_tag": p.get("primary_tag") or "—",
        "framework":  framework,
        "valuation_unit": vu,
        "action_types_valued": action_cov,
        "pool":       p.get("corpus_pool") or "—",
        "code_released": (
            "✓" if ext and (ext.get("code_release") or {}).get("released") else "✗"
        ),
    }

all_rows = [_row_for_paper(p) for p in papers]
df = pd.DataFrame(all_rows)


# --- Filter bar -----------------------------------------------------------
section_eyebrow("Filter & search")

col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.2, 1.4], gap="small")
with col_a:
    pool_filter = st.multiselect(
        "Corpus pool", options=sorted(df["pool"].unique()),
        default=sorted(df["pool"].unique()),
    )
with col_b:
    fw_filter = st.multiselect(
        "Framework", options=sorted(x for x in df["framework"].unique() if x != "—"),
        default=[],
    )
with col_c:
    vu_filter = st.multiselect(
        "Valuation unit", options=sorted(x for x in df["valuation_unit"].unique() if x != "—"),
        default=[],
    )
with col_d:
    year_min, year_max = int(df["year"].min() or 0), int(df["year"].max() or 2025)
    year_filter = st.slider(
        "Year range",
        min_value=year_min, max_value=year_max,
        value=(year_min, year_max),
    )

col_q1, col_q2 = st.columns([3, 1], gap="small")
with col_q1:
    q = st.text_input("Search title / author / paper_id",
                       placeholder="e.g. VAEP, Decroos, off-ball, …")

# --- Apply filters --------------------------------------------------------
mask = df["pool"].isin(pool_filter) & df["year"].between(*year_filter)
if fw_filter:
    mask &= df["framework"].isin(fw_filter)
if vu_filter:
    mask &= df["valuation_unit"].isin(vu_filter)
if q:
    ql = q.lower()
    paper_authors_lookup = {p["paper_id"]: (p.get("authors") or "").lower() for p in papers}
    df["_searchable"] = df.apply(
        lambda r: f"{r['paper_id']} {r['title']} {paper_authors_lookup.get(r['paper_id'], '')}".lower(),
        axis=1,
    )
    mask &= df["_searchable"].str.contains(ql, regex=False)
    df = df.drop(columns="_searchable")

filtered = df[mask].sort_values(["year", "paper_id"], ascending=[True, True]).reset_index(drop=True)

with col_q2:
    st.markdown(
        f"<div style='margin-top: 1.85rem; color:#06A77D; font-weight:600;'>"
        f"{len(filtered)} of {len(df)} papers</div>",
        unsafe_allow_html=True,
    )


# --- Table with row-selection ---------------------------------------------
st.markdown("&nbsp;", unsafe_allow_html=True)

display_df = filtered.rename(columns={
    "paper_id":           "Paper ID",
    "year":               "Year",
    "title":              "Title",
    "venue_type":         "Venue type",
    "primary_tag":        "Primary tag",
    "framework":          "Framework",
    "valuation_unit":     "Valuation unit",
    "action_types_valued": "Action types valued",
    "pool":               "Pool",
    "code_released":      "Code",
})

selection = st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Year": st.column_config.NumberColumn(width="small", format="%d"),
        "Title": st.column_config.TextColumn(width="large"),
        "Paper ID": st.column_config.TextColumn(width="medium"),
        "Action types valued": st.column_config.TextColumn(width="large"),
        "Code": st.column_config.TextColumn(width="small"),
        "Pool": st.column_config.TextColumn(width="small"),
    },
    height=460,
)

# --- Drill-down view ------------------------------------------------------
st.markdown("---")
section_eyebrow("Inspect a paper")

if selection.selection.rows:
    sel_row_idx = selection.selection.rows[0]
    sel_paper_id = filtered.iloc[sel_row_idx]["paper_id"]
else:
    sel_paper_id = st.selectbox(
        "Pick a paper",
        options=filtered["paper_id"].tolist(),
        format_func=lambda pid: (
            f"{pid} — {filtered.set_index('paper_id').loc[pid, 'title'][:80]}"
        ),
        label_visibility="collapsed",
    ) if len(filtered) else None

if not sel_paper_id:
    st.info("Filter to at least one paper to inspect.")
    footer()
    st.stop()

paper_row = next(p for p in papers if p["paper_id"] == sel_paper_id)
ext = extractions.get(sel_paper_id)


# --- Header block for the selected paper ----------------------------------
title = paper_row.get("title") or "(no title)"
authors = paper_row.get("authors") or ""
year = paper_row.get("year") or ""
venue = paper_row.get("venue") or ""
doi = paper_row.get("doi") or ""
url = paper_row.get("url") or ""
file_path = paper_row.get("file_path") or ""
pool = paper_row.get("corpus_pool") or ""

st.markdown(
    f"""
    <div class="savant-card" style="background:linear-gradient(135deg,#F5F7F6 0%,#FFFFFF 100%);">
      <div class="savant-card-title">{paper_row.get('primary_tag','')} · {pool}</div>
      <h3 style="margin-bottom:0.3rem; font-size:1.35rem;">{title}</h3>
      <div style="color:#404040; margin-bottom:0.7rem;">
        {authors.replace(';', ' · ')} &nbsp;·&nbsp; <strong>{year}</strong>
      </div>
      <div style="color:#6B7280; font-size:0.92rem;">
        {venue}{(' · DOI: <a href="https://doi.org/' + doi + '">' + doi + '</a>') if doi else ''}
        {('<br/><a href="' + url + '">' + url + '</a>') if url else ''}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Tabs --------------------------------------------------------------
if ext is None:
    badges([(pool, "muted"), ("held-out: no annotation record", "accent")])
    st.markdown(
        '<div class="savant-prose" style="margin-top:1rem;">'
        'This paper is in the corpus as <strong>held-out</strong>: only its '
        'bibliographic row exists. Stage 3\'s extraction agent will produce '
        'its structured record on its next run (and the result will be '
        'compared against the gold standard for evaluation).'
        '</div>',
        unsafe_allow_html=True,
    )
    footer()
    st.stop()


tab_overview, tab_method, tab_data, tab_eval, tab_practical, tab_code, \
    tab_claims, tab_rels, tab_json = st.tabs([
    "Overview", "Methodology", "Data", "Evaluation", "Practical",
    "Code release", "Claims & Limitations", "Relationships", "Raw JSON",
])

with tab_overview:
    scope = ext.get("scope", {})
    st.markdown(f"**Why it's in scope.** {scope.get('rationale','')}")
    if scope.get("tags"):
        st.markdown("**Tags:**")
        badges([(t, "primary") for t in scope["tags"]])

with tab_method:
    m = ext.get("model", {}) or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Name.** {m.get('name','—')}")
        st.markdown(f"**Family.** {m.get('family','—')}")
        st.markdown(f"**Framework.** {m.get('framework','—')}")
        st.markdown(f"**Valuation unit.** `{m.get('valuation_unit','—')}`")
        st.markdown(f"**Atomicity.** `{m.get('atomicity','—')}`")
        st.markdown(f"**Supervision.** {m.get('supervision','—')}")
    with c2:
        st.markdown(f"**Architecture.** {m.get('architecture','—')}")
        st.markdown(f"**Spatial resolution.** {m.get('spatial_resolution') or '—'}")
        st.markdown(f"**Temporal horizon.** {m.get('temporal_horizon') or '—'}")
        outs = m.get("outputs", {}) or {}
        st.markdown(f"**Output scale.** `{outs.get('value_scale','—')}`")
        st.markdown(
            "**Aggregation levels.** " +
            ", ".join(f"`{a}`" for a in (outs.get("aggregation_levels") or [])) or "—"
        )
    st.markdown(f"**Target.** _{m.get('target','—')}_")
    st.markdown("**Inputs.**")
    for inp in (m.get("inputs") or []):
        st.markdown(f"- {inp}")

    st.markdown("**Action coverage.**")
    ac = m.get("action_coverage", {}) or {}
    cov_badges = [
        (k.replace("_", " "), "primary" if v else "muted")
        for k, v in ac.items()
    ]
    badges(cov_badges)

with tab_data:
    d = ext.get("data", {}) or {}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Provider.** {d.get('provider','—')}")
        st.markdown("**Data type:** " + ", ".join(f"`{x}`" for x in d.get("data_type", [])))
        st.markdown("**Dataset (canonical):** " + ", ".join(
            f"`{x}`" for x in d.get("dataset_canonical", [])))
    with c2:
        st.markdown(f"**Leagues.** {', '.join(d.get('leagues', [])) or '—'}")
        st.markdown(f"**Seasons.** {', '.join(d.get('seasons', [])) or '—'}")
        st.markdown(f"**N matches.** {d.get('n_matches','—')}")
        public = d.get("is_public")
        st.markdown(
            f"**Public dataset?** " +
            ("✅ Yes" if public is True else "❌ No" if public is False else "—")
        )

with tab_eval:
    e = ext.get("evaluation", {}) or {}
    st.markdown("**Metrics.** " + ", ".join(f"`{x}`" for x in e.get("metrics", [])))
    st.markdown("**Tasks.** " + ", ".join(f"`{x}`" for x in e.get("tasks", [])))
    st.markdown("**Baselines.**")
    for b in (e.get("baselines") or []):
        st.markdown(f"- {b}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Calibration reported.** {e.get('calibration_reported','—')}")
        st.markdown(f"**Uncertainty reported.** {e.get('uncertainty_reported','—')}")
    with c2:
        st.markdown(
            f"**Validated vs expert judgment.** {e.get('validated_against_expert_judgment','—')}"
        )
        st.markdown(f"**Bootstrap/CI reported.** {e.get('bootstrap_or_ci_reported','—')}")
    st.markdown(f"**Splits.** {e.get('splits','—')}")

with tab_practical:
    p_block = ext.get("practical", {}) or {}
    st.markdown("**Use cases.**")
    badges([(uc.replace("-", " "), "accent") for uc in (p_block.get("use_cases") or [])])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Real-time capable.** {p_block.get('inference_realtime_capable','—')}")
        st.markdown(f"**Interpretability.** `{p_block.get('interpretability_level','—')}`")
    with c2:
        st.markdown(f"**Compute intensity.** `{p_block.get('compute_intensity','—')}`")

with tab_code:
    cr = ext.get("code_release", {}) or {}
    if cr.get("released"):
        st.markdown(f"**Released.** ✅")
        if cr.get("url"):
            st.markdown(f"**URL.** {cr['url']}")
        st.markdown(f"**License.** {cr.get('license') or 'unknown'}")
        st.markdown(f"**Implementer.** {cr.get('implementer') or 'unknown'}")
        st.markdown(f"**Maintenance.** {cr.get('maintenance_status') or 'unknown'}")
    else:
        st.markdown("**Released.** ❌  No public code found.")

with tab_claims:
    st.markdown("**Author-stated claims.**")
    for cl in (ext.get("claims") or []):
        st.markdown(
            f"- _{cl.get('evidence_location','?')}_ — {cl.get('statement','')}"
        )
    st.markdown("**Author-stated limitations.**")
    for lim in (ext.get("limitations") or []):
        st.markdown(f"- {lim}")
    oq = ext.get("open_questions") or []
    if oq:
        st.markdown("**Author-stated open questions.**")
        for q in oq:
            st.markdown(f"- {q}")

with tab_rels:
    rels = ext.get("relationships") or []
    if not rels:
        st.info("No paper-to-paper relationships recorded for this entry.")
    for r in rels:
        st.markdown(
            f"- **{r['type']}** → `{r['target_paper_id']}`"
        )

with tab_json:
    st.json(ext, expanded=False)


# --- PDF access -----------------------------------------------------------
if file_path:
    pdf_path = REPO / file_path
    if pdf_path.exists():
        with pdf_path.open("rb") as fh:
            st.download_button(
                "Download source PDF",
                data=fh.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )

footer()
