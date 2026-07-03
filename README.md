---
title: SAVANT
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.50.0
python_version: "3.11"
app_file: streamlit_app.py
pinned: true
license: mit
short_description: Soccer action valuation — KG, gap discovery, agentic chat
tags:
  - soccer
  - action-valuation
  - knowledge-graph
  - sports-analytics
  - llm-agent
  - survey
---

# SAVANT — Soccer Action Valuation

This Space hosts the SAVANT interactive UI. SAVANT is an agentic framework
that:

1. Reads the soccer action-valuation literature into a structured 52-field
   schema (24 candidate papers + 4 held-out).
2. Builds a knowledge graph of methods, datasets, action types, and
   relationships (165 nodes, 476 typed edges).
3. Identifies and ranks 62 actionable research gaps.
4. Lets researchers chat with the corpus, audit it live, drop in their own
   paper, or watch the gap agent write manuscript prose — all
   citation-grounded.

## Nine pages

| Page | What it does |
|---|---|
| ⚽ Home | Overview + headline stats |
| 📚 Corpus | Filter / search / drill-down across all 28 papers |
| 🕸 Knowledge Graph | Interactive `pyvis` HTML + 3 manuscript figures |
| 🔥 Coverage | 6 Plotly heatmaps with hover-to-see-methods |
| 🎯 Gaps | 62 algorithmic gaps + the manuscript top-5 rationale |
| 🤖 Agentic Pipeline | Mermaid architecture + per-agent prompt + dry-run |
| ⚡ Live Demo | Bring your own API key — 4 modes (spot / audit / your-paper / gap-agent) |
| 💬 Chat | Citation-grounded chat over the 24-paper corpus |
| ℹ️ About | Schema, methods, BibTeX, acknowledgements |

## Bring your own ANTHROPIC_API_KEY

The Live Demo and Chat pages call Claude (Anthropic) on demand. They take
your API key via password input — **the key lives in your browser session
only and is never written to disk** by this Space.

If you don't have one, get a key at
[console.anthropic.com](https://console.anthropic.com/settings/keys).

## Run it locally instead

```bash
git clone https://github.com/<your-username>/savant-action-valuation.git
cd savant-action-valuation
pip install -e .
streamlit run streamlit_app.py
```

## Acknowledgements

Developed by Pegah Rahimian (lead), with Axum (corpus annotation) and
David Sumpter (methodological verification). Built on the open
soccer-analytics community — `socceraction`, StatsBomb Open Data,
SkillCorner, Metrica Sports.

See the **About** page for the BibTeX entry and full methods description.

## License

MIT.
