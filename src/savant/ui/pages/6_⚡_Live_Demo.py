"""SAVANT UI — Live demo.

Four ways to spend Anthropic tokens, in order of cost:
1. Spot check        — critic on ONE paper           (~$0.05–$0.15)
2. Gap rationale     — gap-agent on ranked.json      (~$0.20–$0.40)
3. Your own paper    — extraction + critic + KG fit  (~$0.10–$0.40)
4. Audit the corpus  — critic on all 24 records      (~$2–$4)

Safety:
- API key lives in browser session only.
- No extraction JSON is ever modified by this UI.
- Every section prints actual token cost from the call log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from savant.agents import AnthropicClient
from savant.agents.client import default_log_dir
from savant.extraction import CriticRunner, ExtractionRunner
from savant.gap_analysis import GapRunner, build_inputs_from_artifacts
from savant.knowledge_graph import coverage_diff, gap_diff, nearest_neighbours
from savant.ui.components import (
    apply_theme, badges, footer, hero, section_eyebrow,
)
from savant.ui.data import (
    REPO, load_enumeration, load_extractions, load_kg, load_ranking,
)

st.set_page_config(
    page_title="SAVANT — Live demo",
    page_icon="⚡",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Stage 2 · See it run",
    title="Run SAVANT yourself — four flavours, one Anthropic key.",
    subtitle=(
        "Bring your own ANTHROPIC_API_KEY. Spot-check one paper, audit the "
        "whole corpus, drop in your own PDF and see where it sits in the KG, "
        "or watch the gap-agent author manuscript prose live. Every section "
        "shows actual token cost — nothing is persisted, nothing modifies "
        "the gold standard."
    ),
)


# --- Session state -------------------------------------------------------
for key, default in [
    ("api_key", ""),
    ("spot_verdict", None),
    ("spot_meta", None),
    ("audit_results", None),
    ("audit_total_cost", 0.0),
    ("byop_record", None),
    ("byop_critic", None),
    ("byop_meta", None),
    ("byop_neighbours", None),
    ("byop_coverage", None),
    ("byop_gap_diff", None),
    ("gap_result", None),
    ("gap_meta", None),
    ("cost_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Pricing -------------------------------------------------------------
PRICE_PER_M = {"input": 15.00, "output": 75.00}


def _read_latest_log_entry() -> dict:
    """Return the most recently appended LLM call log entry, or empty dict."""
    log_dir = default_log_dir()
    log_files = sorted(log_dir.glob("*.jsonl"))
    if not log_files:
        return {}
    last_log = log_files[-1]
    lines = last_log.read_text().strip().splitlines()
    if not lines:
        return {}
    return json.loads(lines[-1])


def _cost(in_tok: int, out_tok: int) -> float:
    return round(
        (in_tok * PRICE_PER_M["input"] + out_tok * PRICE_PER_M["output"]) / 1_000_000,
        4,
    )


def _log_to_meta(entry: dict) -> dict:
    in_tok = entry.get("input_tokens") or 0
    out_tok = entry.get("output_tokens") or 0
    return {
        "model": entry.get("model"),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "duration_s": entry.get("duration_s"),
        "cost_usd": _cost(in_tok, out_tok),
    }


# --- API key advisory (top of page) -------------------------------------
st.markdown(
    """
    <div class="savant-card" style="
        background:linear-gradient(135deg, #FFF8EC 0%, #FFFFFF 100%);
        border-color: rgba(241,143,1,0.35);">
      <div class="savant-card-title" style="color:#B36500;">Set your API key once</div>
      <ul style="margin: 0.3rem 0 0 1rem; padding: 0; line-height: 1.6; color: #404040;">
        <li>Key lives in your browser session only — never persisted by this UI.</li>
        <li>Pricing assumed below: Opus <strong>$15 / M input tokens</strong> + <strong>$75 / M output tokens</strong>.</li>
        <li>No file is modified — verdicts and records are displayed, not saved.</li>
        <li>Get a key at <a href="https://console.anthropic.com/settings/keys">console.anthropic.com</a>.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.session_state.api_key = st.text_input(
    "ANTHROPIC_API_KEY",
    value=st.session_state.api_key,
    type="password",
    placeholder="sk-ant-…",
    label_visibility="collapsed",
)

has_key = bool(st.session_state.api_key)
if not has_key:
    st.caption("⚠ Paste your API key above to enable any of the four sections below.")


# --- Tabs ----------------------------------------------------------------
extractions = {r["paper_id"]: r for r in load_extractions()}
kg = load_kg()
enumeration = load_enumeration()
ranking = load_ranking()

tab_spot, tab_audit, tab_byop, tab_gap = st.tabs([
    "🔍  Spot check",
    "📋  Audit corpus",
    "📤  Your own paper",
    "🎯  Generate gap rationale",
])


# =========================================================================
# TAB 1 — Spot check (single-paper critic — what B8 did)
# =========================================================================
with tab_spot:
    section_eyebrow("Run the critic on one paper (~$0.05–$0.15)")
    st.markdown(
        '<div class="savant-prose" style="margin-bottom:1rem;">'
        'Pick any paper, hand the critic the record + a stand-in source '
        '(its own claims), see what the critic flags. Good for spot-checking '
        'a specific paper before David\'s formal verification or for a '
        'reviewer reproducing one of our claims.'
        '</div>',
        unsafe_allow_html=True,
    )

    paper_id = st.selectbox(
        "Paper",
        options=sorted(extractions.keys()),
        index=sorted(extractions.keys()).index("decroos_2019_vaep"),
        key="spot_paper",
    )

    if st.button("▶ Run critic on this paper", type="primary", disabled=not has_key,
                  key="spot_run"):
        record = extractions[paper_id]
        # Build minimum-source from the record's own claims
        parts = [
            f"PDF excerpt placeholder for {paper_id}.",
            f"Scope rationale: {record['scope'].get('rationale', '')}",
            "Claims:",
        ]
        for c in record.get("claims", []):
            parts.append(f"  - [{c.get('evidence_location','?')}] {c.get('statement','')}")
        source = "\n".join(parts)

        with st.spinner(f"Calling Claude on {paper_id}…"):
            try:
                client = AnthropicClient(api_key=st.session_state.api_key)
                verdict = CriticRunner(client).run(record=record, source=source)
                st.session_state.spot_verdict = verdict
                meta = _log_to_meta(_read_latest_log_entry())
                meta["paper_id"] = paper_id
                st.session_state.spot_meta = meta
                st.session_state.cost_history.append({"section": "spot", **meta})
            except Exception as exc:
                st.error(f"Call failed: {exc}")

    if st.session_state.spot_verdict:
        v = st.session_state.spot_verdict
        meta = st.session_state.spot_meta or {}
        color = {"accept": "#10B981", "revise": "#F59E0B",
                 "reject": "#EF4444"}.get(v["verdict"], "#6B7280")
        emoji = {"accept": "✓", "revise": "↻", "reject": "✗"}.get(v["verdict"], "?")
        st.markdown(
            f"""<div class="savant-card" style="border-color:{color};border-width:2px;
              background: linear-gradient(135deg, {color}15 0%, #FFFFFF 100%);">
              <div class="savant-card-title" style="color:{color};">Verdict — {meta.get('paper_id','?')}</div>
              <h2 style="color:{color};margin:0.2rem 0;font-size:1.8rem;">{emoji} {v['verdict'].upper()}</h2>
              <div style="color:#6B7280;font-size:0.9rem;margin-top:0.5rem;">
                {meta.get('model','?')} · in={meta.get('input_tokens','?')} · out={meta.get('output_tokens','?')} · cost=${meta.get('cost_usd',0):.4f}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Full verdict JSON", expanded=False):
            st.code(json.dumps(v, indent=2), language="json")


# =========================================================================
# TAB 2 — Audit corpus (critic on all 24, aggregated dashboard)
# =========================================================================
with tab_audit:
    section_eyebrow("Run the critic on every corpus paper (~$2–$4 for all 24)")

    st.markdown(
        """
        <div class="savant-prose" style="margin-bottom:1rem;">
          The critic loops over every candidate paper and returns a verdict.
          This <strong>doesn't modify any record</strong> — it produces a QA dashboard
          you can act on offline. Use it when David's verification arrives, to find
          extractions the critic and David disagree on.
        </div>

        <details style="margin-bottom: 1rem;">
          <summary style="cursor:pointer; font-weight:600;">What does each verdict mean?</summary>
          <ul style="margin-top: 0.5rem; line-height: 1.6;">
            <li><strong>accept</strong> — record is faithful, schema-conformant, complete. No action needed.</li>
            <li><strong>revise</strong> — fixable issues (wrong metric, missed limitation, hallucinated baseline, overclaim). Manual review + re-extract using critic's <code>revision_instructions</code>.</li>
            <li><strong>reject</strong> — fundamentally broken (wrong paper, fabricated content). Re-extract from scratch.</li>
          </ul>
          <p style="color:#6B7280; font-size:0.9rem;">
            Note: until PDF chunking is implemented, the critic sees the record's own
            claims as source. It catches internal contradictions + schema violations,
            not contradictions against the source PDF.
          </p>
        </details>
        """,
        unsafe_allow_html=True,
    )

    audit_col1, audit_col2 = st.columns([1, 3], gap="small")
    with audit_col1:
        max_papers = st.number_input(
            "Papers to audit (max 24)", min_value=1, max_value=24, value=24, step=1,
        )
    with audit_col2:
        est_cost = max_papers * 0.10
        st.markdown(
            f"<div style='margin-top:1.95rem;color:#404040;'>"
            f"Estimated cost: <strong>~${est_cost:.2f}</strong> "
            f"(<code>{max_papers}</code> calls × ~$0.10 each)</div>",
            unsafe_allow_html=True,
        )

    if st.button("▶ Run audit", type="primary", disabled=not has_key, key="audit_run"):
        results: list[dict] = []
        total_cost = 0.0
        paper_ids = sorted(extractions.keys())[:max_papers]
        progress = st.progress(0.0, text="Starting…")
        for i, pid in enumerate(paper_ids):
            record = extractions[pid]
            parts = [
                f"PDF excerpt placeholder for {pid}.",
                f"Scope rationale: {record['scope'].get('rationale', '')}",
                "Claims:",
            ] + [
                f"  - [{c.get('evidence_location','?')}] {c.get('statement','')}"
                for c in record.get("claims", [])
            ]
            source = "\n".join(parts)
            try:
                client = AnthropicClient(api_key=st.session_state.api_key)
                verdict = CriticRunner(client).run(record=record, source=source)
                meta = _log_to_meta(_read_latest_log_entry())
                total_cost += meta.get("cost_usd", 0) or 0
                issues_count = sum(
                    len(verdict.get(k) or []) for k in
                    ("factual_issues", "completeness_issues", "style_issues")
                )
                results.append({
                    "paper_id": pid,
                    "verdict": verdict["verdict"],
                    "schema_conformant": verdict.get("schema_conformant"),
                    "issues": issues_count,
                    "cost_usd": meta.get("cost_usd"),
                    "input_tokens": meta.get("input_tokens"),
                    "output_tokens": meta.get("output_tokens"),
                    "_full": verdict,
                })
            except Exception as exc:
                results.append({
                    "paper_id": pid, "verdict": "ERROR", "schema_conformant": None,
                    "issues": 0, "cost_usd": 0, "error": str(exc),
                })
            progress.progress((i + 1) / len(paper_ids), text=f"{i+1}/{len(paper_ids)} — {pid}")
        progress.empty()
        st.session_state.audit_results = results
        st.session_state.audit_total_cost = round(total_cost, 4)

    if st.session_state.audit_results:
        results = st.session_state.audit_results
        from collections import Counter
        counts = Counter(r["verdict"] for r in results)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.metric("✓ Accept", counts.get("accept", 0))
        with c2:
            st.metric("↻ Revise", counts.get("revise", 0))
        with c3:
            st.metric("✗ Reject", counts.get("reject", 0))
        with c4:
            st.metric("Total cost", f"${st.session_state.audit_total_cost:.3f}")

        st.markdown("&nbsp;")

        df_audit = pd.DataFrame([
            {k: v for k, v in r.items() if k != "_full"} for r in results
        ])
        st.dataframe(df_audit, hide_index=True, use_container_width=True, height=420)

        # Download audit JSON
        st.download_button(
            "Download audit report (JSON)",
            data=json.dumps([{k: v for k, v in r.items() if k != "_full"}
                              for r in results], indent=2),
            file_name="savant_audit.json",
            mime="application/json",
        )

        # Drill into a paper with issues
        flagged = [r for r in results if r["verdict"] in ("revise", "reject")]
        if flagged:
            st.markdown("---")
            section_eyebrow("Drill into flagged papers")
            pick = st.selectbox(
                "Paper", options=[r["paper_id"] for r in flagged],
                key="audit_drill",
            )
            picked = next(r for r in flagged if r["paper_id"] == pick)
            v = picked.get("_full", {})
            for k, label in (
                ("factual_issues", "Factual issues"),
                ("completeness_issues", "Completeness issues"),
                ("style_issues", "Style issues"),
            ):
                items = v.get(k) or []
                if items:
                    st.markdown(f"**{label}**")
                    for item in items:
                        if isinstance(item, dict):
                            st.markdown(
                                f"- `{item.get('field','')}` — {item.get('issue','')}"
                            )
                        else:
                            st.markdown(f"- {item}")
            if v.get("revision_instructions"):
                st.markdown("**Revision instructions**")
                st.markdown(
                    f"<div class='savant-prose' style='border-left:3px solid #F18F01;"
                    f"padding-left:1rem;'>{v['revision_instructions']}</div>",
                    unsafe_allow_html=True,
                )


# =========================================================================
# TAB 3 — Your own paper (PDF upload → extraction → critic → KG fit)
# =========================================================================
with tab_byop:
    section_eyebrow("Drop in your own paper (~$0.10–$0.40)")

    st.markdown(
        """
        <div class="savant-prose" style="margin-bottom:1rem;">
          Upload a soccer action-valuation PDF, get back the same 52-field
          structured record we produce for the corpus — plus its
          <strong>position in the SAVANT knowledge graph</strong>: nearest
          methodological neighbours, action-coverage diff, and which empty
          cells your method fills.
        </div>

        <details style="margin-bottom:1rem;">
          <summary style="cursor:pointer; font-weight:600;">
            How is this different from asking Claude directly?
          </summary>
          <ul style="margin-top: 0.5rem; line-height: 1.6;">
            <li>Output uses the <strong>same 52-field schema</strong> as the existing 24-paper corpus → directly comparable.</li>
            <li>Schema validation enforced — malformed output rejected, not silently returned.</li>
            <li>Critic re-verifies the extraction → lower hallucination rate than a naive prompt.</li>
            <li>Output is <strong>positioned in the KG</strong> — you don't just get a JSON, you get a map of where your paper sits.</li>
            <li>Free-text vocabulary canonicalised (ROC-AUC ≡ AUROC), so you can find the cell your method belongs to.</li>
          </ul>
        </details>
        """,
        unsafe_allow_html=True,
    )

    up_col1, up_col2 = st.columns([1, 1], gap="medium")
    with up_col1:
        uploaded = st.file_uploader(
            "Paper PDF", type=["pdf"],
            help="The first ~10 pages of text are extracted via PyMuPDF and sent to the agent.",
        )
    with up_col2:
        provided_title = st.text_input(
            "Paper title (optional override)",
            help="If blank, the extraction agent will infer from the PDF.",
        )

    byop_col_a, byop_col_b = st.columns(2, gap="medium")
    with byop_col_a:
        provided_authors = st.text_input(
            "Authors (semicolon-separated)",
            placeholder="A B; C D; E F",
        )
    with byop_col_b:
        provided_year = st.number_input(
            "Year", min_value=1990, max_value=2030, value=2024, step=1,
        )

    if st.button("▶ Extract + critique + position",
                  type="primary",
                  disabled=not (has_key and uploaded), key="byop_run"):
        # 1. Extract text from PDF
        try:
            import fitz  # PyMuPDF
        except ImportError:
            st.error("PyMuPDF (fitz) not installed.")
            st.stop()

        try:
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            # Take first ~10 pages to bound context
            pages_to_read = min(10, doc.page_count)
            full_text = "\n\n".join(
                doc[i].get_text() for i in range(pages_to_read)
            )
            doc.close()
        except Exception as exc:
            st.error(f"PDF read failed: {exc}")
            st.stop()

        paper_id = f"upload_{int(time.time())}"
        bibliographic = {
            "title": provided_title or "(to be filled by agent)",
            "authors": [a.strip() for a in provided_authors.split(";") if a.strip()] or ["(unknown)"],
            "year": int(provided_year),
            "venue": "user-uploaded",
            "venue_type": "other",
        }
        chunks = [{"chunk_id": "full",
                   "section": "all",
                   "page": 1,
                   "text": full_text[:50000]}]  # cap text size

        with st.spinner("Step 1 / 3: Extraction agent reading the paper…"):
            try:
                client = AnthropicClient(api_key=st.session_state.api_key)
                record = ExtractionRunner(client).run(
                    paper_id=paper_id,
                    bibliographic=bibliographic,
                    chunks=chunks,
                )
                ex_meta = _log_to_meta(_read_latest_log_entry())
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                st.stop()

        with st.spinner("Step 2 / 3: Critic re-verifying the extraction…"):
            try:
                source_for_critic = full_text[:8000]  # bound critic input size
                critic = CriticRunner(client).run(record=record, source=source_for_critic)
                cr_meta = _log_to_meta(_read_latest_log_entry())
            except Exception as exc:
                st.warning(f"Critic step skipped (failed): {exc}")
                critic = None
                cr_meta = {}

        with st.spinner("Step 3 / 3: Positioning in the SAVANT knowledge graph…"):
            try:
                st.session_state.byop_neighbours = nearest_neighbours(record, kg, top_k=5)
                st.session_state.byop_coverage   = coverage_diff(record, kg)
                st.session_state.byop_gap_diff   = gap_diff(record, enumeration)
            except Exception as exc:
                st.warning(f"KG integration step failed: {exc}")

        total_cost = round(
            (ex_meta.get("cost_usd", 0) or 0) + (cr_meta.get("cost_usd", 0) or 0), 4
        )
        st.session_state.byop_record = record
        st.session_state.byop_critic = critic
        st.session_state.byop_meta = {
            "extraction": ex_meta, "critic": cr_meta, "total_cost": total_cost,
        }
        st.session_state.cost_history.append(
            {"section": "byop", "model": ex_meta.get("model"),
             "input_tokens": (ex_meta.get("input_tokens", 0) or 0) + (cr_meta.get("input_tokens", 0) or 0),
             "output_tokens": (ex_meta.get("output_tokens", 0) or 0) + (cr_meta.get("output_tokens", 0) or 0),
             "cost_usd": total_cost, "duration_s": None,
             "paper_id": paper_id}
        )

    if st.session_state.byop_record:
        record = st.session_state.byop_record
        critic = st.session_state.byop_critic
        meta = st.session_state.byop_meta or {}

        # Cost meter
        st.markdown(
            f"<div style='color:#404040;'>"
            f"Total cost for this run: <strong>${meta.get('total_cost',0):.4f}</strong> · "
            f"extraction: ${meta.get('extraction',{}).get('cost_usd',0):.4f} · "
            f"critic: ${meta.get('critic',{}).get('cost_usd',0):.4f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;")

        # Verdict badge
        if critic:
            verdict = critic.get("verdict", "?")
            color = {"accept": "#10B981", "revise": "#F59E0B",
                     "reject": "#EF4444"}.get(verdict, "#6B7280")
            st.markdown(
                f"<div class='savant-card' style='border-color:{color};border-width:2px;"
                f"background:{color}15;'>"
                f"<div class='savant-card-title' style='color:{color};'>Critic verdict</div>"
                f"<h3 style='color:{color};margin:0.2rem 0;'>{verdict.upper()}</h3>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # KG positioning
        st.markdown("---")
        section_eyebrow("Where this paper sits in the SAVANT corpus")

        # Nearest neighbours
        if st.session_state.byop_neighbours:
            st.markdown("**Top 5 nearest methodological neighbours**")
            nb_df = pd.DataFrame([
                {
                    "paper_id": n["paper_id"],
                    "method": n["method_name"],
                    "framework": n["framework"],
                    "valuation_unit": n["valuation_unit"],
                    "similarity": n["similarity"],
                }
                for n in st.session_state.byop_neighbours
            ])
            st.dataframe(nb_df, hide_index=True, use_container_width=True)

        # Action coverage diff
        if st.session_state.byop_coverage:
            st.markdown("**Action-coverage profile vs corpus**")
            relation_emoji = {
                "fills-gap": "🟢", "rare": "🟡",
                "common": "⚪", "absent": "—",
            }
            cov_df = pd.DataFrame([
                {
                    "Action type": c["action_type"],
                    "Your paper covers": "✓" if c["new_covers"] else "✗",
                    "Corpus methods doing it": f"{c['corpus_n']} ({c['corpus_pct']}%)",
                    "Relation": f"{relation_emoji.get(c['relation'], '')} {c['relation']}",
                }
                for c in st.session_state.byop_coverage
            ])
            st.dataframe(cov_df, hide_index=True, use_container_width=True)

        # Gap diff
        if st.session_state.byop_gap_diff:
            st.markdown("**Does this paper fill any of our 62 actionable gaps?**")
            fills = [g for g in st.session_state.byop_gap_diff if g["would_fill_gap"]]
            if fills:
                for g in fills:
                    st.success(
                        f"🟢 Fills cell `{g['framework']} × {g['valuation_unit']} × {g['action_type']}` "
                        "— previously empty in the corpus!"
                    )
            else:
                st.info(
                    "This paper does not fill any 3-D coverage cell that was "
                    "previously empty — every cell it occupies already has "
                    "at least one method in our corpus."
                )

        # Raw record + critic
        with st.expander("Full extracted record (v0.3.0 schema)", expanded=False):
            st.code(json.dumps(record, indent=2), language="json")
        if critic:
            with st.expander("Full critic verdict", expanded=False):
                st.code(json.dumps(critic, indent=2), language="json")


# =========================================================================
# TAB 4 — Generate gap rationale (gap-agent live)
# =========================================================================
with tab_gap:
    section_eyebrow("Watch the gap-agent author manuscript prose (~$0.20–$0.40)")
    st.markdown(
        """
        <div class="savant-prose" style="margin-bottom:1rem;">
          The gap agent reads the algorithmic ranking + coverage matrices +
          corpus papers, then produces a 5–8 entry consolidated rationale
          with scientific motivation, practitioner motivation, and a
          public-data study suggestion per gap. Below we compare the LLM's
          output to the <strong>hand-written A7 draft</strong> so you can see what
          the model adds (or misses).
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("▶ Run gap-agent live", type="primary", disabled=not has_key,
                  key="gap_run"):
        with st.spinner("Calling Claude on the ranking + coverage…"):
            try:
                inputs = build_inputs_from_artifacts(
                    kg, enumeration, ranking,
                    extraction_records=list(extractions.values()),
                    top_k=20,
                )
                client = AnthropicClient(api_key=st.session_state.api_key)
                result = GapRunner(client).run(**inputs)
                meta = _log_to_meta(_read_latest_log_entry())
                st.session_state.gap_result = result
                st.session_state.gap_meta = meta
                st.session_state.cost_history.append({"section": "gap", **meta})
            except Exception as exc:
                st.error(f"Call failed: {exc}")

    if st.session_state.gap_result:
        result = st.session_state.gap_result
        meta = st.session_state.gap_meta or {}
        st.markdown(
            f"<div style='color:#404040;'>"
            f"<strong>{len(result['gaps'])} gaps</strong> produced — "
            f"cost ${meta.get('cost_usd',0):.4f} · "
            f"{meta.get('input_tokens')} in / {meta.get('output_tokens')} out · "
            f"{meta.get('duration_s', '?')}s"
            f"</div>",
            unsafe_allow_html=True,
        )

        for i, gap in enumerate(result["gaps"], start=1):
            with st.expander(f"Gap {i}: {gap.get('title','(no title)')}", expanded=(i == 1)):
                cells = gap.get("consolidated_auto_ranks") or []
                if cells:
                    badges([
                        (f"#{c.get('auto_rank','?')} · {c.get('framework','?')} × "
                         f"{c.get('valuation_unit','?')} × {c.get('action_type','?')}",
                         "primary")
                        for c in cells[:6]
                    ])
                st.markdown(f"**The gap.** {gap.get('the_gap','')}")
                st.markdown(f"**Scientific motivation.** {gap.get('scientific_motivation','')}")
                st.markdown(f"**Practitioner motivation.** {gap.get('practitioner_motivation','')}")
                st.markdown(f"**Closest existing work:** "
                             + ", ".join(f"`{p}`" for p in gap.get("closest_existing_work", [])))
                st.markdown(f"**Suggested first study.** {gap.get('suggested_first_study','')}")
                st.caption(
                    f"novelty={gap.get('novelty_score',0):.2f} · "
                    f"feasibility={gap.get('feasibility_score',0):.2f} · "
                    f"confidence={gap.get('confidence',0):.2f}"
                )

        with st.expander("Full LLM output (JSON)", expanded=False):
            st.code(json.dumps(result, indent=2), language="json")


# --- Session cost roll-up ------------------------------------------------
if st.session_state.cost_history:
    st.markdown("---")
    section_eyebrow("This session")
    total = sum(c.get("cost_usd", 0) or 0 for c in st.session_state.cost_history)
    n_calls = len(st.session_state.cost_history)
    sc1, sc2, sc3 = st.columns(3, gap="small")
    with sc1:
        st.metric("Total API calls", n_calls)
    with sc2:
        st.metric("Total spend (USD)", f"${total:.4f}")
    with sc3:
        if st.button("Clear session"):
            for key in (
                "spot_verdict", "spot_meta",
                "audit_results", "audit_total_cost",
                "byop_record", "byop_critic", "byop_meta",
                "byop_neighbours", "byop_coverage", "byop_gap_diff",
                "gap_result", "gap_meta", "cost_history",
            ):
                if key in st.session_state:
                    if isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    elif isinstance(st.session_state[key], float):
                        st.session_state[key] = 0.0
                    else:
                        st.session_state[key] = None
            st.rerun()

footer()
