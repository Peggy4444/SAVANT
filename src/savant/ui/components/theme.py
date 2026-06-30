"""Heavy custom CSS for the SAVANT Streamlit UI.

Goals:
- Hide every Streamlit fingerprint we can (top toolbar, hamburger, footer).
- Replace the default sans with Inter + a serif for prose.
- Custom hero, stat tiles, cards, badges, sidebar styling.

Call `apply_theme()` at the top of `app.py` AND at the top of every page
under `pages/`. (Streamlit re-runs each page in a fresh context, so the
injected CSS has to be re-emitted per-page.)
"""

from __future__ import annotations

import streamlit as st

# Tuned palette — emerald primary, warm orange accent, neutral chrome.
PALETTE = {
    "primary":    "#06A77D",   # pitch emerald
    "primary_dk": "#048864",
    "accent":     "#F18F01",   # warm orange (CTAs / highlights)
    "ink":        "#1A1A1A",
    "ink_2":      "#404040",
    "ink_3":      "#6B7280",
    "surface":    "#FFFFFF",
    "surface_2":  "#F5F7F6",
    "border":     "#E5E7EB",
    "good":       "#10B981",
    "warn":       "#F59E0B",
    "bad":        "#EF4444",
    "info":       "#3B82F6",
}


def _css() -> str:
    p = PALETTE
    return f"""
    <style>

      /* ---- Typography ---- */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@400;600&display=swap');

      html, body, [class*="css"] {{
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
          color: {p['ink']};
          font-size: 17px;
      }}

      /* Streamlit's main markdown body inherits from the root .stMarkdown */
      .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown ol, .stMarkdown ul {{
          font-size: 1rem;
          line-height: 1.6;
      }}

      /* Slightly tighter line-height for headings, looser for body */
      h1, h2, h3, h4 {{
          font-family: 'Inter', sans-serif !important;
          letter-spacing: -0.015em;
          color: {p['ink']};
      }}
      h1 {{ font-weight: 700; font-size: 2.1rem; }}
      h2 {{ font-weight: 600; margin-top: 1.8rem; font-size: 1.55rem; }}
      h3 {{ font-weight: 600; font-size: 1.25rem; }}

      .savant-prose {{
          font-family: 'Source Serif 4', Georgia, serif !important;
          font-size: 1.06rem;
          line-height: 1.65;
          color: {p['ink_2']};
      }}

      /* Tables / dataframes a bit larger too */
      .stDataFrame, [data-testid="stTable"] {{
          font-size: 0.95rem !important;
      }}

      /* ---- Hide Streamlit chrome ---- */
      #MainMenu          {{ visibility: hidden; }}
      header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
      footer             {{ visibility: hidden; }}
      [data-testid="stDecoration"] {{ display: none; }}
      .stDeployButton    {{ display: none; }}

      /* Tighten the default top padding now that we hid the header.
         Use most of the viewport — no narrow max-width cap. */
      .block-container {{
          padding-top: 1.4rem !important;
          padding-bottom: 4rem !important;
          padding-left: 3rem !important;
          padding-right: 3rem !important;
          max-width: none !important;
      }}

      /* ---- Sidebar ---- */
      [data-testid="stSidebar"] {{
          background: {p['surface_2']};
          border-right: 1px solid {p['border']};
      }}
      [data-testid="stSidebar"] .sidebar-content {{
          padding-top: 1rem;
      }}
      [data-testid="stSidebarNav"] li a {{
          font-weight: 500;
          color: {p['ink_2']};
      }}
      [data-testid="stSidebarNav"] li a:hover {{
          color: {p['primary']};
      }}

      /* ---- Buttons ---- */
      button[kind="primary"], .stButton > button[kind="primary"] {{
          background: {p['primary']};
          border: 1px solid {p['primary']};
          color: white;
          font-weight: 500;
          box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      }}
      button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {{
          background: {p['primary_dk']};
          border-color: {p['primary_dk']};
      }}
      .stButton > button {{
          border-radius: 8px;
          font-weight: 500;
      }}

      /* ---- Inputs ---- */
      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
          border-radius: 8px;
          border-color: {p['border']};
      }}

      /* ---- Custom hero banner ---- */
      .savant-hero {{
          background: linear-gradient(135deg, {p['primary']} 0%, {p['primary_dk']} 55%, #024F3D 100%);
          color: white;
          padding: 2.6rem 2.4rem 2.4rem;
          border-radius: 18px;
          margin-bottom: 2rem;
          position: relative;
          overflow: hidden;
          box-shadow: 0 4px 18px rgba(6, 167, 125, 0.18);
      }}
      .savant-hero::after {{
          content: '';
          position: absolute; top: -50%; right: -10%;
          width: 60%; height: 200%;
          background: radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 60%);
          pointer-events: none;
      }}
      .savant-hero .eyebrow {{
          text-transform: uppercase;
          letter-spacing: 0.14em;
          font-size: 0.78rem;
          color: rgba(255,255,255,0.78);
          margin-bottom: 0.6rem;
          font-weight: 600;
      }}
      .savant-hero h1 {{
          font-size: 2.6rem;
          color: white !important;
          margin: 0 0 0.6rem 0;
          font-weight: 700;
          letter-spacing: -0.02em;
          line-height: 1.15;
      }}
      .savant-hero p {{
          font-size: 1.15rem;
          color: rgba(255,255,255,0.92);
          margin: 0;
          max-width: 880px;
          line-height: 1.55;
      }}

      /* ---- Stat tiles ---- */
      .savant-stat-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
          gap: 1rem;
          margin: 1.4rem 0 1.6rem 0;
      }}
      .savant-stat {{
          background: {p['surface']};
          border: 1px solid {p['border']};
          border-radius: 14px;
          padding: 1.1rem 1.2rem;
          box-shadow: 0 1px 3px rgba(0,0,0,0.03);
          transition: transform 120ms ease, box-shadow 120ms ease;
      }}
      .savant-stat:hover {{
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      }}
      .savant-stat .stat-num {{
          font-family: 'Inter', sans-serif;
          font-size: 2.5rem; font-weight: 700; color: {p['primary']};
          line-height: 1.05; letter-spacing: -0.03em;
      }}
      .savant-stat .stat-label {{
          font-size: 0.95rem; color: {p['ink_3']};
          margin-top: 0.3rem;
          line-height: 1.35;
      }}
      .savant-stat .stat-sub {{
          font-size: 0.85rem; color: {p['ink_3']};
          margin-top: 0.5rem;
      }}

      /* ---- Cards / sections ---- */
      .savant-card {{
          background: {p['surface']};
          border: 1px solid {p['border']};
          border-radius: 14px;
          padding: 1.4rem 1.5rem;
          margin-bottom: 1rem;
          box-shadow: 0 1px 3px rgba(0,0,0,0.03);
      }}
      .savant-card-title {{
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: {p['primary']};
          font-weight: 600;
          margin-bottom: 0.6rem;
      }}
      .savant-card h3 {{
          margin-top: 0 !important;
          margin-bottom: 0.5rem;
          font-size: 1.15rem;
      }}

      /* ---- Badges / chips ---- */
      .savant-badge {{
          display: inline-block;
          background: {p['surface_2']};
          border: 1px solid {p['border']};
          color: {p['ink_2']};
          font-size: 0.74rem;
          padding: 0.18rem 0.55rem;
          border-radius: 999px;
          margin-right: 0.3rem; margin-bottom: 0.3rem;
          font-weight: 500;
      }}
      .savant-badge.primary {{
          background: rgba(6, 167, 125, 0.10);
          color: {p['primary_dk']};
          border-color: rgba(6, 167, 125, 0.30);
      }}
      .savant-badge.accent {{
          background: rgba(241, 143, 1, 0.10);
          color: #B36500;
          border-color: rgba(241, 143, 1, 0.30);
      }}
      .savant-badge.muted {{ color: {p['ink_3']}; }}

      /* ---- Section header with eyebrow ---- */
      .savant-section-eyebrow {{
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.14em;
          color: {p['primary']};
          font-weight: 600;
          margin-bottom: 0.4rem;
      }}

      /* ---- Tables ---- */
      .stDataFrame, [data-testid="stTable"] {{
          border-radius: 10px !important;
          overflow: hidden;
          border: 1px solid {p['border']};
      }}

      /* ---- Footer-credit row ---- */
      .savant-footer {{
          color: {p['ink_3']};
          font-size: 0.82rem;
          padding: 2rem 0 0.5rem;
          border-top: 1px solid {p['border']};
          margin-top: 3rem;
          text-align: center;
      }}

      /* ---- Tabs ---- */
      .stTabs [data-baseweb="tab-list"] {{
          gap: 0.4rem;
          border-bottom: 1px solid {p['border']};
      }}
      .stTabs [data-baseweb="tab"] {{
          font-weight: 500;
          color: {p['ink_3']};
      }}
      .stTabs [data-baseweb="tab"][aria-selected="true"] {{
          color: {p['primary']};
      }}
      .stTabs [data-baseweb="tab-highlight"] {{
          background-color: {p['primary']};
      }}

    </style>
    """


def apply_theme() -> None:
    """Inject the SAVANT CSS into the current Streamlit page. Call once
    near the top of `app.py` AND of every file under `pages/`."""
    st.markdown(_css(), unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    """Render the gradient hero banner."""
    st.markdown(
        f"""<div class="savant-hero">
              <div class="eyebrow">{eyebrow}</div>
              <h1>{title}</h1>
              <p>{subtitle}</p>
           </div>""",
        unsafe_allow_html=True,
    )


def stat_tile(number: str | int, label: str, sub: str | None = None) -> str:
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="savant-stat">'
        f'  <div class="stat-num">{number}</div>'
        f'  <div class="stat-label">{label}</div>'
        f'  {sub_html}'
        f'</div>'
    )


def stat_grid(tiles: list[str]) -> None:
    """Render a list of stat_tile() HTML snippets in a responsive grid."""
    st.markdown(
        f'<div class="savant-stat-grid">{"".join(tiles)}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body_md: str, eyebrow: str | None = None) -> None:
    """A bordered card with optional eyebrow + bold title + markdown body."""
    eyebrow_html = (
        f'<div class="savant-card-title">{eyebrow}</div>' if eyebrow else ""
    )
    # NOTE: we render the body as markdown via st.markdown after the outer card div
    st.markdown(
        f'<div class="savant-card">{eyebrow_html}<h3>{title}</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(body_md)
    st.markdown("</div>", unsafe_allow_html=True)


def section_eyebrow(text: str) -> None:
    st.markdown(
        f'<div class="savant-section-eyebrow">{text}</div>',
        unsafe_allow_html=True,
    )


def badges(items: list[tuple[str, str]]) -> None:
    """items: list of (text, variant) where variant ∈ {primary, accent, muted, ''}."""
    html = "".join(
        f'<span class="savant-badge {variant}">{text}</span>'
        for text, variant in items
    )
    st.markdown(html, unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="savant-footer">'
        'SAVANT — Soccer Action Valuation, an agentic framework for knowledge synthesis, gap discovery, and benchmarking.'
        '</div>',
        unsafe_allow_html=True,
    )
