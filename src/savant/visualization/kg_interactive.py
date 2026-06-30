"""Interactive HTML viewer for the SAVANT knowledge graph.

Builds a single self-contained HTML file (vis.js under the hood) that you
can open in any browser. Click a node to see its full attribute dictionary;
drag to rearrange; scroll to zoom; filter by node type via the legend.

Use case: shareable with collaborators who don't have Python installed —
just attach `savant_kg.html` to an email.

Public API:
    render_interactive(g, out_path) -> Path
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from pyvis.network import Network

from ..knowledge_graph import build_kg, load_all_extractions
from ..utils.config import repo_root

DEFAULT_HTML_PATH = repo_root() / "data" / "outputs" / "kg" / "savant_kg.html"

# Same palette family as the static figures so the three views feel coherent.
NODE_COLORS = {
    "paper":            "#1f77b4",  # blue
    "method":           "#ff7f0e",  # orange
    "author":           "#bcbd22",  # olive
    "venue":            "#17becf",  # cyan
    "framework":        "#2ca02c",  # green
    "valuation_unit":   "#9467bd",  # purple
    "action_type":      "#d62728",  # red
    "dataset":          "#8c564b",  # brown
    "use_case":         "#e377c2",  # pink
    "evaluation_task":  "#7f7f7f",  # grey
}

NODE_SIZES = {
    "paper":            22,
    "method":           20,
    "framework":        18,
    "valuation_unit":   18,
    "action_type":      16,
    "dataset":          16,
    "use_case":         14,
    "evaluation_task":  14,
    "author":           10,
    "venue":            12,
}

EDGE_COLORS = {
    # bibliographic
    "AUTHORED_BY":       "#cccccc",
    "PUBLISHED_IN":      "#cccccc",
    # method-structural
    "PROPOSES":          "#ff7f0e",
    "HAS_FRAMEWORK":     "#2ca02c",
    "HAS_VALUATION_UNIT": "#9467bd",
    "VALUES":            "#d62728",
    "USES_DATASET":      "#8c564b",
    "EVALUATED_ON_TASK": "#7f7f7f",
    "SUITABLE_FOR":      "#e377c2",
    # paper-to-paper
    "EXTENDS":           "#2a7fb8",
    "IMPROVES_UPON":     "#1a9850",
    "COMPARES_AGAINST":  "#a0a0a0",
    "USES_DATASET_OF":   "#8c564b",
    "CRITIQUES":         "#d73027",
}


def _node_title(node_id: str, attrs: dict) -> str:
    """HTML shown when you hover over a node — pretty-printed attribute dump."""
    lines = [f"<b>{node_id}</b>"]
    for k in sorted(attrs):
        v = attrs[k]
        if v is None or v == "":
            continue
        lines.append(f"<b>{k}:</b> {v}")
    return "<br>".join(lines)


def _node_label(node_id: str, attrs: dict) -> str:
    """Short label shown on the node itself."""
    nt = attrs.get("node_type")
    if nt in ("framework", "valuation_unit", "action_type", "dataset",
              "use_case", "evaluation_task"):
        return attrs.get("name") or node_id.split(":", 1)[-1]
    if nt == "paper":
        pid = attrs.get("paper_id") or node_id.split(":", 1)[-1]
        parts = pid.split("_", 2)
        return f"{parts[0].capitalize()} {parts[1]}" if len(parts) >= 2 else pid
    if nt == "method":
        return attrs.get("name") or node_id.split("::", 1)[-1]
    if nt == "author":
        return attrs.get("name") or node_id.split(":", 1)[-1]
    if nt == "venue":
        name = attrs.get("name") or ""
        return name if len(name) < 40 else name[:37] + "…"
    return node_id


def render_interactive(g: nx.MultiDiGraph | None = None,
                       out_path: Path | None = None,
                       height: str = "900px",
                       hide_by_default: tuple[str, ...] = ("author", "venue"),
                       ) -> Path:
    """Render the KG as a single interactive HTML file with a built-in
    legend and per-node-type visibility toggles.

    Parameters
    ----------
    g : optional MultiDiGraph. Built from `load_all_extractions()` if None.
    out_path : where to write the HTML (default: data/outputs/kg/savant_kg.html).
    height : CSS height for the canvas.
    hide_by_default : node types whose legend checkbox is UNCHECKED on load,
        so the corresponding nodes are hidden until the viewer toggles them
        back on. Default: ("author", "venue") since those 70+ peripheral
        nodes crowd the view.
    """
    if g is None:
        g = build_kg(load_all_extractions())
    if out_path is None:
        out_path = DEFAULT_HTML_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height=height, width="100%",
        bgcolor="#fafafa", font_color="#222",
        directed=True, notebook=False, cdn_resources="in_line",
    )

    net.barnes_hut(
        gravity=-12000,
        central_gravity=0.2,
        spring_length=140,
        spring_strength=0.02,
        damping=0.5,
    )
    net.show_buttons(filter_=["physics"])

    node_counts: dict[str, int] = {}
    for n, attrs in g.nodes(data=True):
        nt = attrs.get("node_type", "other")
        node_counts[nt] = node_counts.get(nt, 0) + 1
        net.add_node(
            n,
            label=_node_label(n, attrs),
            title=_node_title(n, attrs),
            color=NODE_COLORS.get(nt, "#cccccc"),
            size=NODE_SIZES.get(nt, 12),
            group=nt,
        )

    for u, v, _, attrs in g.edges(keys=True, data=True):
        et = attrs.get("edge_type", "")
        net.add_edge(
            u, v,
            color=EDGE_COLORS.get(et, "#cccccc"),
            title=et.replace("_", " ").title(),
            arrows="to",
            width=1.2,
            smooth={"type": "continuous"},
        )

    net.write_html(str(out_path), notebook=False, open_browser=False)
    _inject_legend_and_filters(out_path, node_counts, hide_by_default)
    return out_path


def _inject_legend_and_filters(
    html_path: Path,
    node_counts: dict[str, int],
    hide_by_default: tuple[str, ...],
) -> None:
    """Post-process the pyvis-generated HTML: prepend a fixed-position
    legend with one checkbox per node type that toggles visibility, plus
    one for each edge type group."""
    html = html_path.read_text()

    # Build the legend HTML
    node_items = []
    for nt, color in NODE_COLORS.items():
        count = node_counts.get(nt, 0)
        checked = "" if nt in hide_by_default else "checked"
        node_items.append(
            f'<label class="kg-row">'
            f'<input type="checkbox" data-group="{nt}" {checked}>'
            f'<span class="kg-swatch" style="background:{color}"></span>'
            f'<span class="kg-label">{nt}</span>'
            f'<span class="kg-count">{count}</span>'
            f"</label>"
        )

    edge_groups = {
        "Bibliographic":     ["AUTHORED_BY", "PUBLISHED_IN"],
        "Method structural": ["PROPOSES", "HAS_FRAMEWORK", "HAS_VALUATION_UNIT",
                              "VALUES", "USES_DATASET",
                              "EVALUATED_ON_TASK", "SUITABLE_FOR"],
        "Paper to paper":    ["EXTENDS", "IMPROVES_UPON", "COMPARES_AGAINST",
                              "USES_DATASET_OF", "CRITIQUES"],
    }
    edge_items = []
    for group_name, edge_types in edge_groups.items():
        edge_items.append(f'<div class="kg-edge-group-title">{group_name}</div>')
        for et in edge_types:
            color = EDGE_COLORS.get(et, "#999")
            edge_items.append(
                f'<div class="kg-edge-row">'
                f'<span class="kg-edge-swatch" style="background:{color}"></span>'
                f'<span class="kg-label">{et.replace("_", " ").title()}</span>'
                f"</div>"
            )

    legend_html = f"""
<div id="savant-legend">
  <div class="kg-panel-header">SAVANT KG <span class="kg-hint">click to toggle</span></div>
  <div class="kg-section-title">Nodes</div>
  {''.join(node_items)}
  <div class="kg-section-title">Edges</div>
  {''.join(edge_items)}
</div>
<style>
  #savant-legend {{
    position: fixed; top: 12px; left: 12px;
    background: rgba(255,255,255,0.96); border: 1px solid #ccc;
    border-radius: 6px; padding: 10px 12px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px; color: #222; max-height: 88vh; overflow-y: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12); z-index: 1000; min-width: 220px;
  }}
  #savant-legend .kg-panel-header {{
    font-weight: 600; font-size: 13px; margin-bottom: 6px;
    border-bottom: 1px solid #eaeaea; padding-bottom: 5px;
  }}
  #savant-legend .kg-hint {{ font-weight: normal; color: #888; font-size: 10.5px; }}
  #savant-legend .kg-section-title {{
    font-weight: 600; margin: 8px 0 4px 0; color: #444; font-size: 11.5px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  #savant-legend .kg-row {{
    display: flex; align-items: center; gap: 6px;
    padding: 2px 0; cursor: pointer; user-select: none;
  }}
  #savant-legend .kg-row input {{ cursor: pointer; margin: 0; }}
  #savant-legend .kg-swatch {{
    display: inline-block; width: 11px; height: 11px;
    border-radius: 50%; border: 1px solid rgba(0,0,0,0.18);
  }}
  #savant-legend .kg-label {{ flex: 1; }}
  #savant-legend .kg-count {{ color: #888; font-variant-numeric: tabular-nums; }}
  #savant-legend .kg-edge-group-title {{
    font-size: 11px; color: #666; margin: 6px 0 2px 0; font-weight: 500;
  }}
  #savant-legend .kg-edge-row {{
    display: flex; align-items: center; gap: 6px; padding: 1px 0; margin-left: 4px;
  }}
  #savant-legend .kg-edge-swatch {{
    display: inline-block; width: 18px; height: 3px; border-radius: 1px;
  }}
</style>
<script>
  (function () {{
    function applyVisibility(group, checked) {{
      var updates = nodes.get().filter(function (n) {{ return n.group === group; }})
                         .map(function (n) {{ return {{ id: n.id, hidden: !checked }}; }});
      nodes.update(updates);
    }}
    var boxes = document.querySelectorAll('#savant-legend input[type="checkbox"]');
    boxes.forEach(function (cb) {{
      cb.addEventListener('change', function (e) {{
        applyVisibility(e.target.dataset.group, e.target.checked);
      }});
      // Apply initial state (handles default-hidden groups)
      applyVisibility(cb.dataset.group, cb.checked);
    }});
  }})();
</script>
"""

    html = html.replace("</body>", legend_html + "\n</body>")
    html_path.write_text(html)
