"""SAVANT Track-B evaluation runner.

Consumes a benchmark output directory (from `scripts/run_benchmark.py`)
plus the cached raw events and produces:

    - `leaderboard.csv`: ranked by T3 Pearson ρ against team match-xG diff
    - `per_task/t3_pearson_bootstrap.csv`: point + 95% bootstrap CI per method
    - `per_task/t3_wilcoxon_pairwise.csv`: paired-match Wilcoxon + BH-adjusted p
    - `per_task/t4_delta.csv`: Δ(train − holdout) per method  (T4, optional)

Usage
-----
    PYTHONPATH=src python scripts/evaluate_benchmark.py \
        --bench-dir data/outputs/benchmark

Add `--holdout-bench-dir <dir>` to also compute T4 (needs a separate
benchmark run on the holdout competitions from `configs/benchmark_tasks.yaml`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from savant.benchmarking import (
    SPADLLoader, StatsBombOpenLoader, holdout_competitions, load_config,
    train_competitions,
)
from savant.benchmarking.evaluation import run as evaluate_run


def _load_events_for_bench(competitions_fn, max_matches: int | None = None) -> pd.DataFrame:
    """Re-load raw StatsBomb events for the competitions listed in the given
    config accessor. Uses the same disk cache the benchmark run populates,
    so this is fast when the benchmark has already been executed."""
    cfg = load_config()
    loader = StatsBombOpenLoader()
    parts: list[pd.DataFrame] = []
    remaining = max_matches
    for spec in competitions_fn(cfg):
        comp = loader.resolve(spec.name, spec.season)
        matches = loader.list_matches(comp)
        mids = matches["match_id"].tolist()
        if remaining is not None:
            mids = mids[:remaining]
            remaining = max(0, remaining - len(mids))
        if not mids:
            continue
        parts.append(loader.load_events_batch(comp, mids))
        if remaining is not None and remaining == 0:
            break
    if not parts:
        raise RuntimeError("No events loaded — was the benchmark run first?")
    return pd.concat(parts, ignore_index=True)


def _load_actions_for_bench(competitions_fn, max_matches: int | None = None) -> pd.DataFrame:
    """Re-load SPADL actions in the same competition order as the benchmark
    run so T1 label construction can key on the same game_ids."""
    cfg = load_config()
    loader = SPADLLoader()
    parts: list[pd.DataFrame] = []
    remaining = max_matches
    for spec in competitions_fn(cfg):
        comp = loader.resolve(spec.name, spec.season)
        matches = loader.list_matches(comp)
        gids = matches["game_id"].tolist()
        if remaining is not None:
            gids = gids[:remaining]
            remaining = max(0, remaining - len(gids))
        if not gids:
            continue
        parts.append(loader.load_actions_batch(comp, gids))
        if remaining is not None and remaining == 0:
            break
    if not parts:
        raise RuntimeError("No SPADL actions loaded — was the benchmark run first?")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", type=Path,
                        default=REPO / "data" / "outputs" / "benchmark",
                        help="Output dir from run_benchmark.py.")
    parser.add_argument("--holdout-bench-dir", type=Path, default=None,
                        help="Optional: output dir from a holdout benchmark run "
                             "(competitions_holdout_for_t4). Enables T4 Δ.")
    parser.add_argument("--train-partial-root", type=Path,
                        default=REPO / "data" / "outputs" / "benchmark_partial",
                        help="Per-competition partial dir from "
                             "run_benchmark_by_competition.py — enables T3 "
                             "per-competition breakdown + robustness column.")
    parser.add_argument("--holdout-partial-root", type=Path,
                        default=REPO / "data" / "outputs" / "benchmark_holdout_partial",
                        help="Per-competition partial dir for the holdout run.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write leaderboard.csv. Defaults to --bench-dir.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Cap matches when loading events (smoke runs).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    print(f"\n=== SAVANT Track-B evaluation ===\n")
    print(f"  bench_dir: {args.bench_dir}")
    print(f"  loading train events...")
    events = _load_events_for_bench(train_competitions, args.max_matches)
    print(f"  train events: {len(events):,} rows over "
          f"{events['match_id'].nunique()} matches")
    print(f"  loading SPADL actions for T1 labels...")
    actions = _load_actions_for_bench(train_competitions, args.max_matches)
    print(f"  actions: {len(actions):,} rows over "
          f"{actions['game_id'].nunique()} games")

    holdout_events = None
    if args.holdout_bench_dir is not None:
        print(f"  loading holdout events...")
        holdout_events = _load_events_for_bench(holdout_competitions, args.max_matches)
        print(f"  holdout events: {len(holdout_events):,} rows over "
              f"{holdout_events['match_id'].nunique()} matches")

    # Only pass partial-root paths if they actually exist — the runner
    # gracefully skips per-competition analysis when either dir is missing.
    train_partial   = args.train_partial_root   if args.train_partial_root.exists()   else None
    holdout_partial = args.holdout_partial_root if args.holdout_partial_root.exists() else None

    manifest = evaluate_run(
        bench_dir=args.bench_dir,
        events=events,
        actions=actions,
        out_dir=args.out_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        holdout_bench_dir=args.holdout_bench_dir,
        holdout_events=holdout_events,
        train_partial_root=train_partial,
        holdout_partial_root=holdout_partial,
    )
    print(f"\n  wrote leaderboard → {manifest['artefacts']['leaderboard']}")
    print(f"  {manifest['n_methods']} methods evaluated on {manifest['n_matches']} matches")


if __name__ == "__main__":
    main()
