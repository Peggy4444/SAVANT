"""Per-competition T3 breakdown.

Isolates each competition's Pearson ρ on T3 (team match-xG differential).
Answers: is the pooled train-corpus ρ being dragged down by a specific
competition, or is the pattern uniform?

Reads:
    data/outputs/benchmark_partial/comp_*/methods/*.parquet
    data/outputs/benchmark_holdout_partial/comp_*/methods/*.parquet

Writes:
    data/outputs/benchmark/per_task/t3_per_competition.csv

Prints a matrix (methods × competitions) with ρ per cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from savant.benchmarking import (
    StatsBombOpenLoader, holdout_competitions, load_config, train_competitions,
)
from savant.benchmarking.evaluation import pearson_rho, team_match_xg_diff


def _competition_label(spec) -> str:
    """Compact label for the report — 'La Liga 15/16' etc."""
    season = spec.season.replace("/", "-")
    if "/" in spec.season:
        # "2020/2021" → "20/21"; "2015/2016" → "15/16"
        y0, y1 = spec.season.split("/")
        season = f"{y0[-2:]}/{y1[-2:]}"
    return f"{spec.name} {season}"


def _load_competition_events(loader: StatsBombOpenLoader, spec) -> pd.DataFrame:
    comp = loader.resolve(spec.name, spec.season)
    game_ids = loader.list_matches(comp)["match_id"].tolist()
    return loader.load_events_batch(comp, game_ids)


def _per_method_rho_for_dir(
    partial_dir: Path, events: pd.DataFrame,
) -> dict[str, tuple[float, int]]:
    """For every method / baseline parquet in `partial_dir`, return
    {model_id: (rho, n_pairs)} against team_match_xg_diff derived from `events`."""
    target = team_match_xg_diff(events)
    out: dict[str, tuple[float, int]] = {}
    for sub in ("baselines", "methods"):
        for p in sorted((partial_dir / sub).glob("*__t3_team_match_values.parquet")):
            mid = p.name.replace("__t3_team_match_values.parquet", "")
            preds = pd.read_parquet(p)
            joined = preds.merge(target[["game_id", "team_id", "xg_diff"]],
                                 on=["game_id", "team_id"], how="inner")
            if len(joined) < 4:
                out[mid] = (float("nan"), len(joined))
                continue
            out[mid] = (
                pearson_rho(joined["value"].values, joined["xg_diff"].values),
                len(joined),
            )
    return out


def _matches_in_partial(partial_dir: Path) -> int:
    for sub in ("methods", "baselines"):
        for p in (partial_dir / sub).glob("*__t3_team_match_values.parquet"):
            return int(pd.read_parquet(p)["game_id"].nunique())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-partial-root", type=Path,
                        default=REPO / "data" / "outputs" / "benchmark_partial")
    parser.add_argument("--holdout-partial-root", type=Path,
                        default=REPO / "data" / "outputs" / "benchmark_holdout_partial")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "data" / "outputs" / "benchmark" / "per_task")
    args = parser.parse_args()

    cfg = load_config()
    train_comps = train_competitions(cfg)
    holdout_comps = holdout_competitions(cfg)
    loader = StatsBombOpenLoader()

    print("\n=== per-competition T3 ρ breakdown ===\n")

    rows: list[dict] = []
    all_specs = [("train", i, spec) for i, spec in enumerate(train_comps)] + \
                [("holdout", i, spec) for i, spec in enumerate(holdout_comps)]

    for corpus, index, spec in all_specs:
        root = args.train_partial_root if corpus == "train" else args.holdout_partial_root
        pdir = root / f"comp_{index:02d}"
        if not pdir.exists():
            print(f"  skip {corpus} comp {index}: no partial dir")
            continue
        label = _competition_label(spec)
        n_matches = _matches_in_partial(pdir)
        print(f"  {corpus:<7s} comp {index}  {label}  ({n_matches} matches)")

        # Load events for THIS competition only.
        events = _load_competition_events(loader, spec)
        per_model = _per_method_rho_for_dir(pdir, events)
        for mid, (rho, n) in per_model.items():
            rows.append({
                "corpus":       corpus,
                "index":        index,
                "competition":  label,
                "n_matches":    n_matches,
                "model_id":     mid,
                "pearson_rho":  rho,
                "n_pairs":      n,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  no data — did you run the benchmark first?")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "t3_per_competition.csv"
    df.to_csv(out_path, index=False)

    # --- Report ------------------------------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 32)

    pivot = df.pivot_table(
        index="model_id", columns="competition", values="pearson_rho",
        aggfunc="first",
    )
    # Sort columns to put train competitions first (in config order), then holdout.
    train_labels = [_competition_label(s) for s in train_comps]
    holdout_labels = [_competition_label(s) for s in holdout_comps]
    ordered_cols = [c for c in train_labels + holdout_labels if c in pivot.columns]
    pivot = pivot[ordered_cols]

    # Sort methods by mean train-competition ρ (best first).
    train_cols = [c for c in train_labels if c in pivot.columns]
    pivot["_mean_train"] = pivot[train_cols].mean(axis=1)
    pivot = pivot.sort_values("_mean_train", ascending=False)
    pivot = pivot.drop(columns="_mean_train")

    print()
    print("=" * 100)
    print("Pearson ρ per model × competition  (T3, higher is better)")
    print("=" * 100)
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}" if pd.notna(x) else "  —  "))

    # --- Callouts ---------------------------------------------------
    print()
    print("=" * 100)
    print("Per-competition summary  (all methods, all baselines pooled)")
    print("=" * 100)
    summary = (df.groupby(["corpus", "competition", "n_matches"], sort=False)
                 ["pearson_rho"]
                 .agg(["mean", "median", "min", "max"])
                 .reset_index())
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print(f"\n=== wrote {out_path.relative_to(REPO)} ===\n")


if __name__ == "__main__":
    main()
