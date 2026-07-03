"""Empirical-agentic feedback loop: write Track-B benchmark results back
into the knowledge graph as Method-node attributes.

CLAUDE.md §5 Stage 6 calls this out as Track B's contribution to the
SAVANT architecture beyond the leaderboard itself. Concretely:

    After each benchmark run, every method's `empirical_*` metrics are
    written back into the KG as node attributes on the Method node. The
    gap-agent and critic-agent are re-runnable with the enriched KG so
    their outputs are grounded in measured performance, not only in
    extracted claims.

Multi-variant handling
----------------------
Some corpus papers ship multiple benchmark wrappers (e.g. `singh_2018_xt`
has both `xt_state_singh` and `xt_action_sumpter` under it, testing the
schema's `model.valuation_unit` axis). Rather than picking one, the
enricher stores each variant as an entry in a list on the Method node:

    node["empirical_variants"] = [
        {"variant_id": "xt_state_singh",   "t3_pearson_rho": 0.521, ...},
        {"variant_id": "xt_action_sumpter", "t3_pearson_rho": 0.412, ...},
    ]

with corpus-level provenance:

    node["empirical_source"]  = "data/outputs/benchmark/"
    node["empirical_run_at"]  = "2026-07-03T..."
    node["empirical_n_matches_train"]   = 660
    node["empirical_n_matches_holdout"] = 183
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# --- Public API ----------------------------------------------------


def attach_benchmark_results(
    g: nx.MultiDiGraph,
    bench_dir: Path,
) -> nx.MultiDiGraph:
    """Enrich `g` in-place with T1/T2/T3/T4 metrics from `bench_dir`.

    Returns the same graph (mutated). Method nodes without a benchmark row
    are left untouched.
    """
    variant_to_paper = _wrapper_to_corpus_paper_id()
    per_variant = _load_benchmark_rows(bench_dir)
    if not per_variant:
        logger.warning("attach_benchmark_results: no rows loaded from %s", bench_dir)
        return g

    manifest = _load_evaluation_manifest(bench_dir)
    run_manifest = _load_run_manifest(bench_dir)

    corpus_run_at   = run_manifest.get("run_at") or run_manifest.get("partials", [{}])[0].get("run_at")
    n_train         = int(run_manifest.get("n_matches_total", 0))
    n_matches_eval  = int(manifest.get("n_matches", 0)) if manifest else 0

    # Group variants by the corpus paper_id they map onto.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for variant_id, row in per_variant.items():
        pid = variant_to_paper.get(variant_id)
        if pid is None:
            # Baselines aren't corpus methods — skip.
            continue
        grouped.setdefault(pid, []).append({"variant_id": variant_id, **row})

    n_attached = 0
    for pid, variants in grouped.items():
        method_nodes = [
            n for n, d in g.nodes(data=True)
            if d.get("node_type") == "method"
            and _paper_id_of_method_node(g, n) == pid
        ]
        if not method_nodes:
            logger.warning(
                "attach_benchmark_results: no Method node for corpus_paper_id "
                "%s (variants: %s) — the KG may not include this paper.",
                pid, [v["variant_id"] for v in variants],
            )
            continue
        # If a paper has multiple Method nodes (unusual), attach to all of
        # them — the extraction schema doesn't currently make that split
        # explicit, so this is the safe default.
        for mnode in method_nodes:
            g.nodes[mnode]["empirical_variants"]         = variants
            g.nodes[mnode]["empirical_source"]           = str(bench_dir)
            if corpus_run_at:
                g.nodes[mnode]["empirical_run_at"]       = corpus_run_at
            if n_train:
                g.nodes[mnode]["empirical_n_matches_train"] = n_train
            if n_matches_eval and n_matches_eval != n_train:
                g.nodes[mnode]["empirical_n_matches_eval"]  = n_matches_eval
            n_attached += 1

    logger.info(
        "attach_benchmark_results: enriched %d Method node(s) from %d corpus paper(s)",
        n_attached, len(grouped),
    )
    return g


def summarise_enrichment(g: nx.MultiDiGraph) -> dict[str, Any]:
    """Post-enrichment stats for a manifest or CLI report."""
    n_methods = sum(1 for _, d in g.nodes(data=True)
                    if d.get("node_type") == "method")
    n_with_empirical = sum(
        1 for _, d in g.nodes(data=True)
        if d.get("node_type") == "method" and "empirical_variants" in d
    )
    variants_by_paper: dict[str, list[str]] = {}
    for n, d in g.nodes(data=True):
        if d.get("node_type") != "method" or "empirical_variants" not in d:
            continue
        pid = _paper_id_of_method_node(g, n)
        if pid is None:
            continue
        variants_by_paper[pid] = [v["variant_id"] for v in d["empirical_variants"]]
    return {
        "n_methods_total":     n_methods,
        "n_methods_enriched":  n_with_empirical,
        "coverage_pct":        (100 * n_with_empirical / max(n_methods, 1)),
        "variants_by_paper":   variants_by_paper,
    }


# --- Internals -----------------------------------------------------


def _wrapper_to_corpus_paper_id() -> dict[str, str]:
    """Build {wrapper_id → corpus_paper_id} from METHODS_BY_ID.

    Imported lazily so the KG module doesn't hard-depend on the benchmarking
    stack (keeps the extraction / gap-analysis pipelines lean).
    """
    from savant.benchmarking import METHODS_BY_ID
    return {wid: cls.corpus_paper_id for wid, cls in METHODS_BY_ID.items()}


def _paper_id_of_method_node(g: nx.MultiDiGraph, method_node: str) -> str | None:
    """The Method node has no `paper_id` attribute directly — the link is
    the incoming PROPOSES edge from the Paper node."""
    for u, _, ed in g.in_edges(method_node, data=True):
        if ed.get("edge_type") == "PROPOSES":
            return g.nodes[u].get("paper_id")
    return None


def _load_benchmark_rows(bench_dir: Path) -> dict[str, dict[str, float | int]]:
    """Flatten leaderboard + T1 + T2 + T4 into {variant_id → metrics}.

    Missing tasks are silently skipped (partial runs still enrich).
    """
    per_variant: dict[str, dict[str, float | int]] = {}

    # T3 (Pearson ρ, Spearman ρ, MAE + 95% CI bands).
    t3_path = bench_dir / "per_task" / "t3_pearson_bootstrap.csv"
    if t3_path.exists():
        for _, r in pd.read_csv(t3_path).iterrows():
            per_variant.setdefault(r["model_id"], {}).update({
                "t3_pearson_rho":       float(r["pearson_rho_point"]),
                "t3_pearson_rho_lo":    float(r["pearson_rho_lo"]),
                "t3_pearson_rho_hi":    float(r["pearson_rho_hi"]),
                "t3_spearman_rho":      float(r["spearman_rho_point"]),
                "t3_mae":               float(r["mae_point"]),
            })

    # T1 (Brier + log-loss + ECE + AUC).
    t1_path = bench_dir / "per_task" / "t1_brier_bootstrap.csv"
    if t1_path.exists():
        for _, r in pd.read_csv(t1_path).iterrows():
            per_variant.setdefault(r["model_id"], {}).update({
                "t1_brier":             float(r["brier_point"]),
                "t1_log_loss":          float(r["log_loss_binary_point"]),
                "t1_ece":               float(r["ece_point"]),
                "t1_roc_auc":           float(r["roc_auc_point"]),
            })

    # T2 (multiclass log-loss + top-1 acc + multiclass ECE).
    t2_path = bench_dir / "per_task" / "t2_multiclass_bootstrap.csv"
    if t2_path.exists():
        for _, r in pd.read_csv(t2_path).iterrows():
            per_variant.setdefault(r["model_id"], {}).update({
                "t2_log_loss_multiclass": float(r["log_loss_multiclass_point"]),
                "t2_top1_accuracy":       float(r["top1_accuracy_point"]),
                "t2_ece_multiclass":      float(r["ece_multiclass_point"]),
            })

    # T4 (Δ Pearson ρ across competitions).
    t4_path = bench_dir / "per_task" / "t4_delta.csv"
    if t4_path.exists():
        for _, r in pd.read_csv(t4_path).iterrows():
            per_variant.setdefault(r["model_id"], {}).update({
                "t4_delta_pearson_rho": float(r["pearson_rho_delta"]),
            })

    return per_variant


def _load_evaluation_manifest(bench_dir: Path) -> dict[str, Any]:
    p = bench_dir / "evaluation_manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_run_manifest(bench_dir: Path) -> dict[str, Any]:
    p = bench_dir / "run_manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}
