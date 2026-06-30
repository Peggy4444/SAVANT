"""Figures for the manuscript and exploratory notebooks."""

from .kg_figures import (
    plot_coverage_heatmap,
    plot_dataset_overlay,
    plot_lineage,
    render_all,
)
from .kg_interactive import render_interactive

__all__ = [
    "plot_coverage_heatmap",
    "plot_dataset_overlay",
    "plot_lineage",
    "render_all",
    "render_interactive",
]
