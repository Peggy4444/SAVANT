"""Manuscript-quality figures derived from the SAVANT knowledge graph.

Three figures, one per concern:
1. `kg_lineage.png`  — method-to-method paper lineage diagram (year on Y axis,
   edges typed by EXTENDS / IMPROVES_UPON / COMPARES_AGAINST / CRITIQUES).
2. `kg_coverage_heatmap.png` — 24 methods × 7 action types, binary coverage
   matrix grouped by valuation_unit + framework.
3. `kg_dataset_overlay.png` — 24 methods × dataset_canonical, coloured by
   public-vs-private.

All three are PNG @ 300 dpi sized for two-column journal layouts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..knowledge_graph import build_kg, load_all_extractions
from ..utils.config import repo_root

DEFAULT_FIGURES_DIR = repo_root() / "manuscript" / "figures"

# Consistent palette across figures so the manuscript reads coherently.
EDGE_COLORS = {
    "EXTENDS":          "#2a7fb8",  # blue
    "IMPROVES_UPON":    "#1a9850",  # green
    "COMPARES_AGAINST": "#a0a0a0",  # grey
    "CRITIQUES":        "#d73027",  # red
}

FRAMEWORK_COLORS = {
    "expectation-based":       "#1f77b4",
    "markov-decision-process": "#ff7f0e",
    "reinforcement-learning":  "#2ca02c",
    "physics-based":           "#9467bd",
    "rule-based":              "#8c564b",
    "hybrid":                  "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Figure 1 — Lineage diagram
# ---------------------------------------------------------------------------

def _short_label(g: nx.MultiDiGraph, paper_node: str) -> str:
    """Compact label like 'Decroos 2019 (VAEP)' for the lineage figure."""
    d = g.nodes[paper_node]
    pid = d.get("paper_id") or paper_node.split(":", 1)[-1]
    parts = pid.split("_", 2)  # firstauthor _ year _ shortname
    author = parts[0].capitalize() if parts else pid
    year = parts[1] if len(parts) > 1 else (d.get("year") or "")
    short = parts[2].upper().replace("_", " ") if len(parts) > 2 else ""
    return f"{author} {year}\n({short})" if short else f"{author} {year}"


def plot_lineage(g: nx.MultiDiGraph, out_path: Path | None = None) -> Path:
    """Method-to-method lineage diagram. Y axis = publication year. Only
    papers that participate in at least one paper-to-paper edge are shown."""
    if out_path is None:
        out_path = DEFAULT_FIGURES_DIR / "kg_lineage.png"

    p2p_types = set(EDGE_COLORS.keys())

    edges_to_keep = [(u, v, k, d) for u, v, k, d in g.edges(keys=True, data=True)
                     if d.get("edge_type") in p2p_types]
    papers_in_play = set()
    for u, v, _, _ in edges_to_keep:
        papers_in_play.add(u); papers_in_play.add(v)
    sub = g.subgraph(papers_in_play).copy()

    # Layout: Y = year (integer ticks), X = wide spacing so multi-node years don't crowd
    nodes_by_year: dict[int, list[str]] = {}
    for n in sub.nodes():
        year = g.nodes[n].get("year") or 2020
        nodes_by_year.setdefault(year, []).append(n)

    max_per_year = max(len(v) for v in nodes_by_year.values())
    x_span = max(max_per_year + 1, 5)  # at least 5 slots wide
    pos: dict[str, tuple[float, float]] = {}
    for year, nlist in nodes_by_year.items():
        nlist.sort()
        # Centre nodes on the canvas
        offset = (x_span - len(nlist)) / 2.0
        for i, n in enumerate(nlist):
            pos[n] = (offset + i + 0.5, year)

    # Figure size scales with corpus density
    fig_w = max(14, x_span * 1.6)
    fig_h = max(8, len(nodes_by_year) * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    # Draw edges, each type with a distinct curvature so multi-edges between
    # the same pair don't overlap.
    curvature = {"EXTENDS": 0.0, "IMPROVES_UPON": 0.15,
                 "COMPARES_AGAINST": -0.15, "CRITIQUES": 0.3}
    for etype, color in EDGE_COLORS.items():
        elist = [(u, v) for u, v, _, d in edges_to_keep if d.get("edge_type") == etype]
        if not elist:
            continue
        nx.draw_networkx_edges(
            sub, pos, edgelist=elist, ax=ax,
            edge_color=color, width=1.8, alpha=0.75,
            arrows=True, arrowsize=15, arrowstyle="-|>",
            connectionstyle=f"arc3,rad={curvature[etype]}",
            node_size=2400,
        )

    nx.draw_networkx_nodes(
        sub, pos, ax=ax,
        node_size=2400, node_color="white",
        edgecolors="#333333", linewidths=1.2,
    )

    labels = {n: _short_label(g, n) for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=7.5)

    ymin = min(nodes_by_year.keys()) - 0.6
    ymax = max(nodes_by_year.keys()) + 0.6
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(sorted(nodes_by_year.keys()))
    ax.set_yticklabels([str(y) for y in sorted(nodes_by_year.keys())], fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, x_span)
    ax.set_ylabel("Year", fontsize=11)
    ax.set_title(
        "SAVANT — Lineage of Soccer Action Valuation Methods\n"
        "(only papers participating in at least one in-corpus citation edge are shown)",
        fontsize=12, pad=12,
    )

    legend_handles = [
        mpatches.Patch(color=c, label=t.replace("_", " ").title())
        for t, c in EDGE_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        frameon=True, fontsize=9, title="Relationship",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — Coverage heatmap
# ---------------------------------------------------------------------------

ACTION_ORDER = [
    "passes", "shots", "dribbles_carries",
    "defensive_actions", "off_ball_actions",
    "set_pieces", "goalkeeper_actions",
]
ACTION_DISPLAY = {
    "passes":             "Passes",
    "shots":              "Shots",
    "dribbles_carries":   "Dribbles / Carries",
    "defensive_actions":  "Defensive",
    "off_ball_actions":   "Off-ball",
    "set_pieces":         "Set-pieces",
    "goalkeeper_actions": "Goalkeeper",
}


def _method_rows(g: nx.MultiDiGraph) -> list[dict]:
    """Return a stable, paper-ordered list of method dicts for the matrix
    figures. Sorted by (valuation_unit, framework, year, name)."""
    rows = []
    for n, d in g.nodes(data=True):
        if d.get("node_type") != "method":
            continue
        # The paper that proposes this method
        proposer = next(
            (u for u, _, ed in g.in_edges(n, data=True)
             if ed.get("edge_type") == "PROPOSES"), None
        )
        year = g.nodes[proposer].get("year") if proposer else None
        rows.append({
            "node": n,
            "name": d.get("name") or n,
            "valuation_unit": d.get("valuation_unit") or "",
            "framework": d.get("framework") or "",
            "year": year or 0,
        })
    rows.sort(key=lambda r: (r["valuation_unit"], r["framework"], r["year"], r["name"]))
    return rows


def plot_coverage_heatmap(g: nx.MultiDiGraph, out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = DEFAULT_FIGURES_DIR / "kg_coverage_heatmap.png"

    methods = _method_rows(g)
    matrix = np.zeros((len(methods), len(ACTION_ORDER)), dtype=int)
    for i, m in enumerate(methods):
        for _, target, ed in g.out_edges(m["node"], data=True):
            if ed.get("edge_type") != "VALUES":
                continue
            atype = g.nodes[target].get("name")
            if atype in ACTION_ORDER:
                matrix[i, ACTION_ORDER.index(atype)] = 1

    # Color each filled cell by the method's framework
    rgb = np.ones((len(methods), len(ACTION_ORDER), 3))  # white background
    for i, m in enumerate(methods):
        c = FRAMEWORK_COLORS.get(m["framework"], "#444444")
        # convert hex to rgb
        rgb_color = np.array([int(c[k:k+2], 16) / 255 for k in (1, 3, 5)])
        for j in range(len(ACTION_ORDER)):
            if matrix[i, j]:
                rgb[i, j] = rgb_color

    fig, ax = plt.subplots(figsize=(8, max(7, 0.32 * len(methods))), dpi=300)
    ax.imshow(rgb, aspect="auto", interpolation="nearest")

    # Y labels: method name + valuation unit badge
    yticks = []
    for m in methods:
        badge = {"action": "[A]", "state": "[S]", "both": "[B]"}.get(m["valuation_unit"], "")
        yticks.append(f"{badge}  {m['name']}")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(yticks, fontsize=8)
    ax.set_xticks(range(len(ACTION_ORDER)))
    ax.set_xticklabels([ACTION_DISPLAY[a] for a in ACTION_ORDER],
                       rotation=35, ha="right", fontsize=9)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(ACTION_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="#dcdcdc", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    ax.set_title(
        "Action Coverage by Method\n"
        "Rows grouped by valuation unit [A]=action [S]=state [B]=both;  "
        "cell colour = methodological framework",
        fontsize=11, pad=12,
    )

    # Legend for framework colours
    fw_handles = [mpatches.Patch(color=c, label=f) for f, c in FRAMEWORK_COLORS.items()]
    ax.legend(handles=fw_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8,
              title="Framework", title_fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 — Dataset overlay
# ---------------------------------------------------------------------------

def plot_dataset_overlay(g: nx.MultiDiGraph, out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = DEFAULT_FIGURES_DIR / "kg_dataset_overlay.png"

    methods = _method_rows(g)
    dataset_nodes = sorted(
        (n for n, d in g.nodes(data=True) if d.get("node_type") == "dataset"),
        key=lambda n: (
            0 if g.nodes[n].get("is_public") else 1,  # public datasets first
            g.nodes[n].get("name", n),
        ),
    )
    dataset_names = [g.nodes[n]["name"] for n in dataset_nodes]
    is_public = [g.nodes[n].get("is_public", False) for n in dataset_nodes]

    matrix = np.zeros((len(methods), len(dataset_nodes)), dtype=int)
    for i, m in enumerate(methods):
        for _, target, ed in g.out_edges(m["node"], data=True):
            if ed.get("edge_type") != "USES_DATASET":
                continue
            if target in dataset_nodes:
                matrix[i, dataset_nodes.index(target)] = 1

    # Color: green for public-dataset usage, dark grey for private
    rgb = np.ones((len(methods), len(dataset_nodes), 3))
    for j, public in enumerate(is_public):
        color = np.array([26, 158, 70]) / 255 if public else np.array([90, 90, 90]) / 255
        for i in range(len(methods)):
            if matrix[i, j]:
                rgb[i, j] = color

    fig, ax = plt.subplots(figsize=(8, max(7, 0.32 * len(methods))), dpi=300)
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m["name"] for m in methods], fontsize=8)
    ax.set_xticks(range(len(dataset_nodes)))
    ax.set_xticklabels(dataset_names, rotation=35, ha="right", fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(dataset_nodes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="#dcdcdc", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    ax.set_title(
        "Datasets Used by Method\n"
        "Green = public dataset (reproducible);  Grey = private / proprietary",
        fontsize=11, pad=12,
    )

    handles = [
        mpatches.Patch(color=(26/255, 158/255, 70/255), label="Public dataset"),
        mpatches.Patch(color=(90/255, 90/255, 90/255), label="Private dataset"),
    ]
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def render_all(g: nx.MultiDiGraph | None = None,
               out_dir: Path | None = None) -> dict[str, Path]:
    if g is None:
        g = build_kg(load_all_extractions())
    out_dir = out_dir or DEFAULT_FIGURES_DIR
    return {
        "lineage":  plot_lineage(g, out_dir / "kg_lineage.png"),
        "coverage": plot_coverage_heatmap(g, out_dir / "kg_coverage_heatmap.png"),
        "dataset":  plot_dataset_overlay(g, out_dir / "kg_dataset_overlay.png"),
    }
