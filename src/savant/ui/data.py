"""Cached data loaders shared by every UI page.

Streamlit re-runs each page on every interaction; without caching every
page would rebuild the KG (~50ms) and the coverage matrices (~20ms) on
each rerun. Wrapping the loaders in `st.cache_data` / `st.cache_resource`
makes the UI feel instant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import streamlit as st

from ..gap_analysis import enumerate_cells, rank_gaps
from ..knowledge_graph import build_kg, load_all_extractions
from ..utils.config import repo_root


REPO = repo_root()


@st.cache_data(show_spinner=False)
def load_extractions() -> list[dict[str, Any]]:
    return load_all_extractions()


@st.cache_resource(show_spinner=False)
def load_kg() -> nx.MultiDiGraph:
    return build_kg(load_all_extractions())


@st.cache_data(show_spinner=False)
def load_enumeration() -> dict[str, Any]:
    return enumerate_cells(load_kg())


@st.cache_data(show_spinner=False)
def load_ranking() -> dict[str, Any]:
    return rank_gaps(load_enumeration())


@st.cache_data(show_spinner=False)
def load_papers_csv() -> list[dict[str, str]]:
    """The papers.csv (28 rows: 24 candidate + 4 held-out)."""
    import csv
    path = REPO / "data" / "papers" / "papers.csv"
    if not path.exists():
        # Fall back to the committed template if the working copy doesn't
        # exist (fresh clone).
        path = REPO / "data" / "papers" / "papers.template.csv"
    with path.open() as fh:
        return list(csv.DictReader(fh))


@st.cache_data(show_spinner=False)
def load_top5_rationale_md() -> str:
    path = REPO / "data" / "outputs" / "gaps" / "top5_rationale.md"
    return path.read_text() if path.exists() else ""


@st.cache_data(show_spinner=False)
def figure_paths() -> dict[str, Path]:
    fdir = REPO / "manuscript" / "figures"
    return {
        "lineage":   fdir / "kg_lineage.png",
        "coverage":  fdir / "kg_coverage_heatmap.png",
        "dataset":   fdir / "kg_dataset_overlay.png",
    }


@st.cache_data(show_spinner=False)
def kg_html_path() -> Path:
    return REPO / "data" / "outputs" / "kg" / "savant_kg.html"


# --- Track-B benchmark artefacts -----------------------------------


@st.cache_data(show_spinner=False)
def benchmark_dir() -> Path:
    return REPO / "data" / "outputs" / "benchmark"


@st.cache_data(show_spinner=False)
def load_leaderboard() -> "pd.DataFrame | None":
    """Return the T3 leaderboard produced by `scripts/evaluate_benchmark.py`,
    or None if no benchmark run has been executed yet."""
    import pandas as pd
    path = benchmark_dir() / "leaderboard.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_t3_bootstrap() -> "pd.DataFrame | None":
    """Point + bootstrap-CI table for T3 metrics."""
    import pandas as pd
    path = benchmark_dir() / "per_task" / "t3_pearson_bootstrap.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_t1_bootstrap() -> "pd.DataFrame | None":
    """T1 per-action metrics: Brier + log-loss + ECE + AUC with bootstrap CIs."""
    import pandas as pd
    path = benchmark_dir() / "per_task" / "t1_brier_bootstrap.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_t2_bootstrap() -> "pd.DataFrame | None":
    """T2 multiclass possession-outcome metrics with bootstrap CIs."""
    import pandas as pd
    path = benchmark_dir() / "per_task" / "t2_multiclass_bootstrap.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_t4_delta() -> "pd.DataFrame | None":
    """T4 cross-competition Δ metrics.
    Produced by `scripts/evaluate_benchmark.py --holdout-bench-dir ...`."""
    import pandas as pd
    path = benchmark_dir() / "per_task" / "t4_delta.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_t3_wilcoxon() -> "pd.DataFrame | None":
    """Pairwise Wilcoxon signed-rank + BH-adjusted p-values on T3 errors."""
    import pandas as pd
    path = benchmark_dir() / "per_task" / "t3_wilcoxon_pairwise.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_run_manifest() -> dict[str, Any] | None:
    """The `run_manifest.json` from the benchmark run — carries the split,
    match counts, and per-method timings displayed alongside the leaderboard."""
    path = benchmark_dir() / "run_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_evaluation_manifest() -> dict[str, Any] | None:
    path = benchmark_dir() / "evaluation_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def corpus_facts() -> dict[str, Any]:
    """One stop for every headline-cite-able number used across the UI."""
    g = load_kg()
    enum = load_enumeration()
    ranking = load_ranking()
    papers = load_papers_csv()

    n_candidates = sum(1 for r in papers if r.get("corpus_pool") == "candidate")
    n_heldout = sum(1 for r in papers if r.get("corpus_pool") == "held-out")
    n_with_code = 0
    n_realtime = 0
    n_public_data_methods = 0

    public_dataset_nodes = {
        n for n, d in g.nodes(data=True)
        if d.get("node_type") == "dataset" and d.get("is_public")
    }
    for n, d in g.nodes(data=True):
        if d.get("node_type") != "method":
            continue
        if d.get("code_released"):
            n_with_code += 1
        if d.get("inference_realtime_capable") is True:
            n_realtime += 1
        for _, target, ed in g.out_edges(n, data=True):
            if ed.get("edge_type") == "USES_DATASET" and target in public_dataset_nodes:
                n_public_data_methods += 1
                break

    cells_3d = enum["matrices"]["framework_x_valuation_unit_x_action_type"]
    nonempty_3d = sum(1 for c in cells_3d if c["count"] > 0)

    return {
        "n_papers_total": len(papers),
        "n_candidates": n_candidates,
        "n_heldout": n_heldout,
        "n_kg_nodes": g.number_of_nodes(),
        "n_kg_edges": g.number_of_edges(),
        "n_methods": sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "method"),
        "n_authors": sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "author"),
        "n_3d_cells_occupied": nonempty_3d,
        "n_3d_cells_total": len(cells_3d),
        "n_ranked_gaps": ranking["n_candidate_cells"],
        "n_actionable_gaps": ranking["n_with_positive_methodological_score"],
        "n_methods_with_code": n_with_code,
        "n_methods_realtime": n_realtime,
        "n_methods_on_public_data": n_public_data_methods,
        "years_min": min((int(p["year"]) for p in papers if p.get("year")), default=0),
        "years_max": max((int(p["year"]) for p in papers if p.get("year")), default=0),
    }
