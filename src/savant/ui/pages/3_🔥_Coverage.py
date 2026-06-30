"""SAVANT UI — Coverage matrices.

Six interactive Plotly heatmaps over the SAVANT enumeration:
- Framework × Action type
- Valuation unit × Action type
- Framework × Valuation unit (corpus shape)
- Framework × Dataset
- Dataset × Action type
- Framework × Valuation unit × Action type (3-D, faceted)

Hover any cell to see which methods occupy it (or "gap" if empty).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from savant.gap_analysis.enumerate import (
    ACTION_TYPES, FRAMEWORKS, VALUATION_UNITS,
)
from savant.ui.components import (
    apply_theme, footer, hero, section_eyebrow,
)
from savant.ui.data import load_enumeration, load_ranking

st.set_page_config(
    page_title="SAVANT — Coverage",
    page_icon="🔥",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Stage 2 · Coverage analysis",
    title="Where the field has worked — and where it hasn't.",
    subtitle=(
        "Six projections of the 24-method corpus across (framework × "
        "valuation_unit × action_type × dataset). Hover any cell to see "
        "which methods occupy it. The white cells are gaps."
    ),
)


enumeration = load_enumeration()
ranking = load_ranking()


# --- Heatmap helper -------------------------------------------------------
def make_heatmap(
    cells: list[dict[str, Any]],
    *,
    x_key: str, x_values: list[str], x_label: str,
    y_key: str, y_values: list[str], y_label: str,
    title: str | None = None,
    cell_height: int = 56,
) -> go.Figure:
    z = np.zeros((len(y_values), len(x_values)), dtype=int)
    hover = [["" for _ in x_values] for _ in y_values]

    for c in cells:
        if c[x_key] not in x_values or c[y_key] not in y_values:
            continue
        xi = x_values.index(c[x_key])
        yi = y_values.index(c[y_key])
        z[yi][xi] = c["count"]
        methods = c.get("methods") or []
        if methods:
            shown = methods[:8]
            extra = f"<br>(+ {len(methods) - 8} more)" if len(methods) > 8 else ""
            hover[yi][xi] = (
                f"<b>{c[y_key]}</b> × <b>{c[x_key]}</b><br>"
                f"<b>{c['count']} method(s):</b><br>"
                + "<br>".join(f"&nbsp;&nbsp;• {m}" for m in shown)
                + extra
            )
        else:
            hover[yi][xi] = (
                f"<b>{c[y_key]}</b> × <b>{c[x_key]}</b><br>"
                "<i>Empty cell — gap</i>"
            )

    # Text annotations: black on white/light, white on deep green
    annotations = []
    z_max = max(1, int(z.max()))
    for yi in range(len(y_values)):
        for xi in range(len(x_values)):
            v = int(z[yi][xi])
            if v == 0:
                continue
            text_color = "white" if v >= z_max * 0.6 else "#1A1A1A"
            annotations.append(dict(
                x=x_values[xi], y=y_values[yi],
                text=str(v), showarrow=False,
                font=dict(size=14, color=text_color, family="Inter"),
            ))

    fig = go.Figure(go.Heatmap(
        z=z, x=x_values, y=y_values,
        text=hover,
        hoverinfo="text",
        colorscale=[
            [0.0,  "#FFFFFF"],
            [0.001, "#E8F5F0"],
            [0.4,  "#65C4A4"],
            [1.0,  "#048864"],
        ],
        showscale=False,
        xgap=2, ygap=2,
        zmin=0, zmax=z_max,
    ))
    fig.update_layout(
        title=dict(text=title or "", x=0, font=dict(size=15, family="Inter", color="#1A1A1A")),
        height=max(280, cell_height * len(y_values) + 100),
        margin=dict(l=20, r=20, t=40 if title else 20, b=10),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#1A1A1A"),
        xaxis=dict(
            side="top",
            tickangle=-25,
            showgrid=False, zeroline=False,
            title=dict(text=x_label, font=dict(size=12, color="#6B7280")),
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False, zeroline=False,
            title=dict(text=y_label, font=dict(size=12, color="#6B7280")),
        ),
        annotations=annotations,
    )
    return fig


# --- Headline gap spotlights at top --------------------------------------
section_eyebrow("Headline gaps from the algorithmic ranker")

top3 = ranking["ranked_gaps"][:3]
cols = st.columns(3, gap="medium")
for col, gap in zip(cols, top3):
    fw = gap["framework"]
    vu = gap["valuation_unit"]
    at = gap["action_type"]
    rationale = gap["rationale"].split(". ")[0] + "."
    nearby = ", ".join(n["paper_id"] for n in (gap.get("nearby_methods") or [])[:3])
    with col:
        st.markdown(
            f"""<div class="savant-card" style="height:100%;">
                <div class="savant-card-title">#{gap['rank']} · score {gap['score']:.2f}</div>
                <h3 style="font-size:1.05rem; margin-bottom:0.5rem;">
                  <code>{fw}</code> × <code>{vu}</code> × <code>{at}</code>
                </h3>
                <div style="color:#404040; font-size:0.95rem; line-height:1.5;">{rationale}</div>
                {('<div style="color:#6B7280; font-size:0.85rem; margin-top:0.5rem;">'
                  'Closest existing work: <code>' + nearby + '</code></div>') if nearby else ''}
            </div>""",
            unsafe_allow_html=True,
        )


# --- Tabs for each matrix ------------------------------------------------
st.markdown("&nbsp;")
section_eyebrow("Browse the six coverage projections")

tabs = st.tabs([
    "Framework × Action type",
    "Valuation unit × Action type",
    "Framework × Valuation unit",
    "Framework × Dataset",
    "Dataset × Action type",
    "3-D faceted view",
])


with tabs[0]:
    st.caption(
        "Each cell = number of corpus methods using that framework AND valuing "
        "that action type. The empty cells are gaps — see headline cards above for the top-ranked ones."
    )
    fig = make_heatmap(
        enumeration["matrices"]["framework_x_action_type"],
        x_key="action_type",  x_values=list(ACTION_TYPES),  x_label="Action type",
        y_key="framework",     y_values=list(FRAMEWORKS),    y_label="Framework",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


with tabs[1]:
    st.caption(
        "Cross-cut by valuation_unit (action / state / both) rather than framework. "
        "Reveals which action types are valued primarily via action vs state methods."
    )
    fig = make_heatmap(
        enumeration["matrices"]["valuation_unit_x_action_type"],
        x_key="action_type",      x_values=list(ACTION_TYPES),      x_label="Action type",
        y_key="valuation_unit",   y_values=list(VALUATION_UNITS),   y_label="Valuation unit",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


with tabs[2]:
    st.caption(
        "The corpus shape. Rows = frameworks; columns = valuation_unit. "
        "Each method counts exactly once — totals sum to corpus size."
    )
    fig = make_heatmap(
        enumeration["matrices"]["framework_x_valuation_unit"],
        x_key="valuation_unit",  x_values=list(VALUATION_UNITS),  x_label="Valuation unit",
        y_key="framework",       y_values=list(FRAMEWORKS),       y_label="Framework",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


with tabs[3]:
    st.caption(
        "Which frameworks live on which datasets. Reveals data-bound research "
        "communities (e.g. tracking-only RL methods)."
    )
    datasets = enumeration["dimensions"]["datasets_in_corpus"]
    fig = make_heatmap(
        enumeration["matrices"]["framework_x_dataset"],
        x_key="dataset",    x_values=datasets,           x_label="Dataset (canonical)",
        y_key="framework",  y_values=list(FRAMEWORKS),   y_label="Framework",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


with tabs[4]:
    st.caption(
        "The reproducibility surface. Cells in public-dataset columns (statsbomb-open, "
        "wyscout-public-2017-18, other-public) are where the field could plausibly benchmark."
    )
    datasets = enumeration["dimensions"]["datasets_in_corpus"]
    fig = make_heatmap(
        enumeration["matrices"]["dataset_x_action_type"],
        x_key="action_type",  x_values=list(ACTION_TYPES),  x_label="Action type",
        y_key="dataset",      y_values=datasets,            y_label="Dataset (canonical)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


with tabs[5]:
    st.caption(
        "The primary 3-D matrix: 6 frameworks × 3 valuation units × 7 action types = 126 cells. "
        "Each panel shows the framework × action_type slice for one valuation_unit."
    )
    # Three side-by-side heatmaps, one per valuation_unit
    cells_3d = enumeration["matrices"]["framework_x_valuation_unit_x_action_type"]
    cols = st.columns(len(VALUATION_UNITS), gap="small")
    for col, vu in zip(cols, VALUATION_UNITS):
        slice_cells = [c for c in cells_3d if c["valuation_unit"] == vu]
        with col:
            fig = make_heatmap(
                slice_cells,
                x_key="action_type",  x_values=list(ACTION_TYPES),  x_label="Action",
                y_key="framework",    y_values=list(FRAMEWORKS),    y_label="",
                title=f"valuation_unit = {vu}",
                cell_height=48,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    n_occupied = sum(1 for c in cells_3d if c["count"] > 0)
    st.markdown(
        f"""<div class="savant-card" style="margin-top:1.2rem;">
              <div class="savant-card-title">3-D coverage</div>
              <h3 style="font-size:1.1rem;">{n_occupied} / {len(cells_3d)} cells occupied</h3>
              <div style="color:#404040; line-height:1.55;">
                The remaining {len(cells_3d) - n_occupied} empty cells are the
                gap-analysis input. {ranking['n_with_positive_methodological_score']} of those
                have a positive methodological-coverage score (i.e. each axis is
                represented somewhere in the corpus) — those are the
                <em>actionable</em> gaps. Browse them on the
                <a href="/Gaps">Gaps page →</a>
              </div>
           </div>""",
        unsafe_allow_html=True,
    )

footer()
