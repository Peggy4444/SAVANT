"""Attach Track-B benchmark results as node attributes on the KG.

Reads `data/outputs/benchmark/` (T3 leaderboard + T1/T2/T4 per-task CSVs)
and merges them into `data/outputs/kg/savant_kg.{json,graphml,html}` so
the gap-agent and critic-agent can reason over measured performance
instead of only extracted claims.

Usage
-----
    PYTHONPATH=src python scripts/enrich_kg_with_benchmark.py
    PYTHONPATH=src python scripts/enrich_kg_with_benchmark.py \\
        --bench-dir data/outputs/benchmark \\
        --kg-out    data/outputs/kg \\
        --dry-run

Rebuilds the pyvis HTML too so the UI reflects the enriched graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from savant.knowledge_graph import (
    attach_benchmark_results, build_kg, load_all_extractions, save_kg,
    summarise_enrichment,
)


def _step(msg: str) -> None:
    print(f"  {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", type=Path,
                        default=REPO / "data" / "outputs" / "benchmark",
                        help="Track-B benchmark output dir.")
    parser.add_argument("--kg-out", type=Path,
                        default=REPO / "data" / "outputs" / "kg",
                        help="Where to write the enriched KG artefacts.")
    parser.add_argument("--rebuild-html", action="store_true", default=True,
                        help="Rebuild the pyvis HTML for the UI (default on).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print enrichment summary but don't write files.")
    args = parser.parse_args()

    print("\n=== SAVANT KG empirical enrichment ===\n")

    _step("loading extractions...")
    records = load_all_extractions()
    _step(f"  {len(records)} extraction records")

    _step("building KG...")
    g = build_kg(records)
    _step(f"  {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    _step(f"attaching benchmark rows from {args.bench_dir.relative_to(REPO) if args.bench_dir.is_relative_to(REPO) else args.bench_dir}...")
    g = attach_benchmark_results(g, args.bench_dir)
    summary = summarise_enrichment(g)
    _step(f"  {summary['n_methods_enriched']}/{summary['n_methods_total']} "
          f"Method nodes enriched ({summary['coverage_pct']:.0f}%)")
    for pid, variants in sorted(summary["variants_by_paper"].items()):
        _step(f"    {pid}: {', '.join(variants)}")

    if args.dry_run:
        _step("dry-run: not writing files.")
        return

    _step(f"saving KG → {args.kg_out.relative_to(REPO) if args.kg_out.is_relative_to(REPO) else args.kg_out}")
    paths = save_kg(g, args.kg_out)
    _step(f"  wrote {paths['json'].name}")
    _step(f"  wrote {paths['graphml'].name}")

    # HTML rebuild for the UI (this used to be under savant.visualization).
    if args.rebuild_html:
        try:
            from savant.visualization.kg_interactive import render_interactive
            html_path = args.kg_out / "savant_kg.html"
            render_interactive(g, out_path=html_path)
            _step(f"  wrote {html_path.name}")
        except Exception as exc:
            _step(f"  HTML rebuild skipped: {exc}")

    # Manifest so the enrichment is diff-able across runs.
    manifest = {
        "bench_dir": str(args.bench_dir),
        "kg_out":    str(args.kg_out),
        "summary":   summary,
        "artefacts": {k: str(v) for k, v in paths.items()},
    }
    manifest_path = args.kg_out / "enrichment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    _step(f"  wrote {manifest_path.name}")

    print(f"\n=== done — KG now carries empirical_variants on "
          f"{summary['n_methods_enriched']} Method nodes ===\n")


if __name__ == "__main__":
    main()
