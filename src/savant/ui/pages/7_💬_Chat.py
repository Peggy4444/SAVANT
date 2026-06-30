"""SAVANT UI — Chat with the corpus.

Citation-grounded chat over the 24-paper SAVANT corpus. The whole compact
corpus is sent as system context (only ~12K tokens for 24 papers), so no
retrieval pipeline is needed for v1. Every answer is forced to cite
`[paper_id]` slugs that exist in the corpus — citations are rendered as
chips, unknown IDs are flagged.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any  # noqa: F401  (used by chat_kwargs annotation)

import streamlit as st

from savant.agents.client import default_log_dir, prompt_hash, supports_temperature
from savant.ui.components import (
    apply_theme, footer, hero, section_eyebrow,
)
from savant.ui.data import corpus_facts, load_extractions

st.set_page_config(
    page_title="SAVANT — Chat",
    page_icon="💬",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Stage 2 · Ask the corpus",
    title="Chat with the literature, citation-grounded.",
    subtitle=(
        "Ask anything in plain English. SAVANT answers from the 24 structured "
        "records — every claim cites a `paper_id` you can click through to "
        "the corpus browser. No invented references; no answers that aren't "
        "supported by the corpus."
    ),
)


# --- Session state -------------------------------------------------------
for key, default in [
    ("api_key", ""),
    ("chat_messages", []),
    ("chat_total_cost", 0.0),
    ("chat_call_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


PRICE_PER_M = {"input": 15.00, "output": 75.00}


# --- Build compact corpus context (once per session) ---------------------
def _compact_record(record: dict[str, Any]) -> str:
    pid = record["paper_id"]
    bib = record.get("bibliographic", {}) or {}
    m = record.get("model", {}) or {}
    d = record.get("data", {}) or {}
    e = record.get("evaluation", {}) or {}
    p = record.get("practical", {}) or {}
    cr = record.get("code_release", {}) or {}
    ac = m.get("action_coverage", {}) or {}

    authors = "; ".join((bib.get("authors") or [])[:3])
    if len(bib.get("authors") or []) > 3:
        authors += " et al."
    valued = ", ".join(k for k, v in ac.items() if v) or "(none)"

    return "\n".join([
        f"[{pid}] {bib.get('title','')} — {authors} ({bib.get('year','?')}, {bib.get('venue','?')})",
        f"  framework={m.get('framework','?')} · valuation_unit={m.get('valuation_unit','?')} · "
        f"atomicity={m.get('atomicity','?')} · supervision={m.get('supervision','?')}",
        f"  action_types_valued: {valued}",
        f"  outputs: value_scale={m.get('outputs', {}).get('value_scale','?')}, "
        f"aggregation_levels={m.get('outputs', {}).get('aggregation_levels',[])}",
        f"  data: provider={d.get('provider','?')}, type={d.get('data_type','?')}, "
        f"datasets={d.get('dataset_canonical','?')}, public={d.get('is_public','?')}",
        f"  evaluation: metrics={e.get('metrics',[])}, tasks={e.get('tasks',[])}, "
        f"baselines={e.get('baselines',[])}, calibration_reported={e.get('calibration_reported','?')}, "
        f"validated_vs_experts={e.get('validated_against_expert_judgment','?')}",
        f"  practical: use_cases={p.get('use_cases',[])}, "
        f"realtime={p.get('inference_realtime_capable','?')}, "
        f"interpretability={p.get('interpretability_level','?')}, "
        f"compute={p.get('compute_intensity','?')}",
        f"  code: released={cr.get('released',False)}, "
        f"url={cr.get('url','-')}, maintenance={cr.get('maintenance_status','?')}",
    ])


@st.cache_data(show_spinner=False)
def _corpus_context() -> tuple[str, set[str]]:
    records = load_extractions()
    blocks = [_compact_record(r) for r in records]
    pids = {r["paper_id"] for r in records}
    text = (
        f"SAVANT CORPUS — {len(records)} candidate papers, schema v0.3.0.\n\n"
        + "\n\n".join(blocks)
    )
    return text, pids


CORPUS_CONTEXT, VALID_PIDS = _corpus_context()


SYSTEM_PROMPT = f"""You are SAVANT's corpus assistant.

You answer questions about soccer action-valuation methods STRICTLY from the
structured records below. Two audiences read your answers: academic
researchers writing a survey, and data scientists at football clubs deciding
which method to use. Both want specific, citation-grounded answers.

RULES:
1. Cite every claim by paper_id in square brackets, e.g. [decroos_2019_vaep].
   Only use paper_ids that appear in the CORPUS section below. NEVER invent them.
2. If you cannot answer from the corpus, say so explicitly — do not speculate.
3. Use Markdown for formatting (lists, bold, tables when comparing methods).
4. When the user asks for a recommendation (e.g. "best for X"), be specific
   and explain the trade-offs (interpretability vs accuracy, etc.).
5. When asked to compare methods, prefer a small Markdown table with the
   relevant attributes; cite each row's paper_id.
6. Be concise (≤ 250 words for most answers); a comparison table is the
   only good reason to be longer.

CORPUS:

{CORPUS_CONTEXT}
"""


# --- Citation chip renderer ---------------------------------------------
_CITATION_RE = re.compile(r"\[([a-z][a-z0-9_]+)\]")


def render_with_chips(text: str) -> str:
    """Replace [paper_id] tokens with HTML chips. Unknown IDs flagged amber."""
    def repl(m: re.Match) -> str:
        pid = m.group(1)
        if pid in VALID_PIDS:
            return (
                f'<span style="display:inline-block;background:rgba(6,167,125,0.10);'
                f'color:#048864;border:1px solid rgba(6,167,125,0.30);'
                f'border-radius:999px;padding:0 0.55rem;font-size:0.78rem;'
                f'font-weight:600;margin:0 0.15rem;">{pid}</span>'
            )
        return (
            f'<span style="display:inline-block;background:rgba(245,158,11,0.10);'
            f'color:#B45309;border:1px solid rgba(245,158,11,0.40);'
            f'border-radius:999px;padding:0 0.55rem;font-size:0.78rem;'
            f'font-weight:600;margin:0 0.15rem;" title="Not in corpus">⚠ {pid}</span>'
        )
    return _CITATION_RE.sub(repl, text)


# --- API key + advisory --------------------------------------------------
st.markdown(
    """
    <div class="savant-card" style="background:linear-gradient(135deg,#FFF8EC 0%,#FFFFFF 100%);
        border-color: rgba(241,143,1,0.35);">
      <div class="savant-card-title" style="color:#B36500;">How this works</div>
      <ul style="margin: 0.3rem 0 0 1rem; padding: 0; line-height: 1.55; color: #404040;">
        <li>The full 24-paper corpus (compact form, ~12K tokens) is sent as system context with every question — no retrieval, no recall errors.</li>
        <li>Every answer is forced to cite [<code>paper_id</code>] slugs that exist in our corpus. Unknown IDs are flagged in amber.</li>
        <li>Cost per question: roughly <strong>$0.10–$0.30</strong> at Opus pricing.</li>
        <li>Conversation history is kept in your browser session only.</li>
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
    st.caption("⚠ Paste your API key above to start chatting.")


# --- Example prompts ----------------------------------------------------
section_eyebrow("Try one of these")

EXAMPLES = [
    "Which methods value off-ball actions, and which of them use public data?",
    "Compare xT and VAEP — when would I pick which?",
    "I have only StatsBomb Open Data and need a method for defensive valuation. What are my options?",
    "Which methods are real-time capable AND fully interpretable?",
    "What gaps does this corpus reveal that an MSc student could feasibly publish on in 6 months?",
    "Show me the methodological lineage from STARSS to multi-agent RL.",
]

ex_cols = st.columns(3, gap="small")
clicked_example = None
for i, ex in enumerate(EXAMPLES):
    col = ex_cols[i % 3]
    with col:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            clicked_example = ex


# --- Render chat history ------------------------------------------------
st.markdown("&nbsp;")

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"], avatar="⚽" if msg["role"] == "assistant" else None):
        st.markdown(render_with_chips(msg["content"]), unsafe_allow_html=True)


# --- Chat input + LLM call ----------------------------------------------
user_msg = st.chat_input(
    "Ask anything about the SAVANT corpus…",
    disabled=not has_key,
)

if clicked_example and has_key and not user_msg:
    user_msg = clicked_example


def _meta_from_log_entry(entry: dict) -> dict:
    in_tok = entry.get("input_tokens") or 0
    out_tok = entry.get("output_tokens") or 0
    return {
        "model": entry.get("model"),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(
            (in_tok * PRICE_PER_M["input"] + out_tok * PRICE_PER_M["output"]) / 1_000_000, 4
        ),
    }


if user_msg:
    # Append user message + render
    st.session_state.chat_messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Call Claude directly (need multi-turn — bypass AgentRunner)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_messages
    ]
    with st.chat_message("assistant", avatar="⚽"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=st.session_state.api_key)
            chat_model = "claude-opus-4-7"
            chat_kwargs: dict[str, Any] = dict(
                model=chat_model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=history,
            )
            if supports_temperature(chat_model):
                chat_kwargs["temperature"] = 0.2
            t0 = time.perf_counter()
            resp = client.messages.create(**chat_kwargs)
            elapsed = time.perf_counter() - t0
            answer = "".join(b.text for b in resp.content if hasattr(b, "text"))
            in_tok = getattr(resp.usage, "input_tokens", 0) or 0
            out_tok = getattr(resp.usage, "output_tokens", 0) or 0
            cost = round(
                (in_tok * PRICE_PER_M["input"] + out_tok * PRICE_PER_M["output"]) / 1_000_000, 4
            )
        except Exception as exc:
            placeholder.error(f"Call failed: {exc}")
            st.session_state.chat_messages.pop()  # rollback user message
            st.stop()

        placeholder.markdown(render_with_chips(answer), unsafe_allow_html=True)

        # Caption: cost + tokens
        st.caption(
            f"`claude-opus-4-7` · "
            f"in={in_tok} · out={out_tok} · "
            f"${cost:.4f} · "
            f"{elapsed:.1f}s"
        )

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_total_cost += cost
    st.session_state.chat_call_count += 1


# --- Footer controls ---------------------------------------------------
if st.session_state.chat_messages:
    st.markdown("---")
    fc1, fc2, fc3 = st.columns([1, 1, 1], gap="small")
    with fc1:
        st.metric("Questions answered", st.session_state.chat_call_count)
    with fc2:
        st.metric("Total spend (USD)", f"${st.session_state.chat_total_cost:.4f}")
    with fc3:
        if st.button("Clear conversation"):
            st.session_state.chat_messages = []
            st.session_state.chat_total_cost = 0.0
            st.session_state.chat_call_count = 0
            st.rerun()

footer()
