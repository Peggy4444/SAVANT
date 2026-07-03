"""SAVANT Track-B benchmark orchestrator.

Wires:  config → data load → match-level split → leakage audit → baseline
(and later method) dispatch → per-task predictions → leaderboard.

For BM-4 the CLI runs the 3 baselines end-to-end on a configurable subset
of training competitions and writes their T3 team-match-value predictions
to `data/outputs/benchmark/`. Metric evaluation itself (Brier, Pearson ρ,
etc.) lands in BM-10. Model wrappers (xT, VAEP, DAxT, ...) land in BM-6+.

Usage
-----
    PYTHONPATH=src python scripts/run_benchmark.py                       # all baselines, all train comps
    PYTHONPATH=src python scripts/run_benchmark.py --max-matches 30      # smoke run
    PYTHONPATH=src python scripts/run_benchmark.py --models xg_per_shot  # single baseline

Every run writes:
    data/outputs/benchmark/
      ├── baselines/{baseline_id}__t3_team_match_values.parquet
      ├── run_manifest.json      # timings, config snapshot, git sha
      └── (later) leaderboard.csv, per_task/*.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from savant.benchmarking import (
    BASELINES_BY_ID, METHODS_BY_ID, SPADLLoader, StatsBombOpenLoader,
    holdout_competitions, leakage_audit, load_config, split_matches,
    split_spec, train_competitions,
)

logger = logging.getLogger("run_benchmark")


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO, capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _step(msg: str, elapsed: float | None = None) -> None:
    prefix = f"[{elapsed:6.2f}s] " if elapsed is not None else " " * 10
    print(f"{prefix}{msg}")


def run(
    baseline_ids: list[str] | None = None,
    method_ids: list[str] | None = None,
    max_matches: int | None = None,
    out_dir: Path | None = None,
    competition_index: int | None = None,
    holdout: bool = False,
) -> dict:
    """Programmatic entrypoint. Returns a manifest dict.

    Set `competition_index` to N (0-indexed) to restrict the run to a
    single training competition — used by the by-competition runner so
    that 360-heavy competitions don't OOM alongside the others.
    """
    t_total = time.perf_counter()
    print("\n=== SAVANT Track-B benchmark ===\n")

    baseline_ids = baseline_ids if baseline_ids is not None else list(BASELINES_BY_ID.keys())
    method_ids = method_ids if method_ids is not None else list(METHODS_BY_ID.keys())
    out_dir = out_dir or (REPO / "data" / "outputs" / "benchmark")
    (out_dir / "baselines").mkdir(parents=True, exist_ok=True)
    (out_dir / "methods").mkdir(parents=True, exist_ok=True)
    (out_dir / "methods_per_action").mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    spec = split_spec(cfg)
    # `holdout=True` reroutes the pool from the training corpus to the
    # T4 evaluation set (WSL 2020/21 + AFCON 2023 per configs/benchmark_tasks.yaml).
    train_comps = (holdout_competitions(cfg) if holdout else train_competitions(cfg))
    corpus_label = "holdout" if holdout else "train"
    if competition_index is not None:
        if not (0 <= competition_index < len(train_comps)):
            raise IndexError(
                f"--competition-index {competition_index} out of range "
                f"[0, {len(train_comps)})",
            )
        train_comps = [train_comps[competition_index]]
        _step(f"config loaded: 1 {corpus_label} competition (index {competition_index}: "
              f"{train_comps[0].name} {train_comps[0].season}), "
              f"{len(baseline_ids)} baselines + {len(method_ids)} methods to run")
    else:
        _step(f"config loaded: {len(train_comps)} {corpus_label} competitions, "
              f"{len(baseline_ids)} baselines + {len(method_ids)} methods to run")

    # --- Data load -----------------------------------------------------
    spadl_loader = SPADLLoader()
    sb_loader = StatsBombOpenLoader()   # for raw events (xG baseline uses these)

    all_spadl: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    total_matches = 0
    remaining = max_matches
    need_360 = any(
        getattr(METHODS_BY_ID.get(mid, None), "requires_360", False)
        for mid in method_ids
    )

    for comp_spec in train_comps:
        t0 = time.perf_counter()
        sc = spadl_loader.resolve(comp_spec.name, comp_spec.season)
        sb_comp = sb_loader.resolve(comp_spec.name, comp_spec.season)
        matches = spadl_loader.list_matches(sc)
        game_ids = matches["game_id"].tolist()
        if remaining is not None:
            game_ids = game_ids[:remaining]
            remaining = max(0, remaining - len(game_ids))
        if not game_ids:
            continue

        spadl_df = spadl_loader.load_actions_batch(sc, game_ids)
        events_df = sb_loader.load_events_batch(sb_comp, game_ids)
        # Normalise the match-id column across the two loaders for the
        # baselines to share a common `game_id`.
        if "match_id" in events_df.columns and "game_id" not in events_df.columns:
            events_df = events_df.rename(columns={"match_id": "game_id"})

        all_spadl.append(spadl_df)
        all_events.append(events_df)

        # Only fetch 360 if a selected method actually needs it (avoids the
        # extra round-trip on runs that don't include OBSO_FF).
        if need_360 and sb_comp.has_360:
            frames_df = sb_loader.load_360_batch(sb_comp, game_ids)
            if frames_df is not None and len(frames_df):
                if "match_id" in frames_df.columns and "game_id" not in frames_df.columns:
                    frames_df = frames_df.rename(columns={"match_id": "game_id"})
                all_frames.append(frames_df)

        total_matches += len(game_ids)
        _step(f"loaded {comp_spec.name} {comp_spec.season}: "
              f"{len(game_ids)} matches → {len(spadl_df):,} SPADL actions",
              time.perf_counter() - t0)
        if remaining is not None and remaining == 0:
            break

    if not all_spadl:
        raise RuntimeError("No matches loaded — check config or --max-matches")

    actions = pd.concat(all_spadl, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    frames = pd.concat(all_frames, ignore_index=True) if all_frames else None
    matches_meta = pd.DataFrame({"game_id": sorted(actions["game_id"].unique())})
    combined_msg = (f"combined: {total_matches} matches, {len(actions):,} SPADL actions, "
                    f"{len(events):,} raw events")
    if frames is not None:
        combined_msg += f", {len(frames):,} 360 frame rows"
    _step(combined_msg)

    # --- Split + leakage audit ----------------------------------------
    t0 = time.perf_counter()
    split = split_matches(matches_meta["game_id"].tolist(), spec=spec)
    audit = leakage_audit(actions, split, match_id_column="game_id")
    _step(f"split (seed={spec.seed}): "
          f"{audit['n_matches_train']}/{audit['n_matches_val']}/{audit['n_matches_test']} matches, "
          f"{audit['n_events_train']:,}/{audit['n_events_val']:,}/{audit['n_events_test']:,} actions "
          "(leakage audit PASSED)",
          time.perf_counter() - t0)

    # --- Baseline dispatch --------------------------------------------
    baseline_results: dict[str, dict] = {}
    for bid in baseline_ids:
        if bid not in BASELINES_BY_ID:
            print(f"  WARNING: unknown baseline {bid!r}, skipping")
            continue
        t0 = time.perf_counter()
        model_cls = BASELINES_BY_ID[bid]
        model = model_cls()
        model.fit(actions, matches_meta)

        # T3: team-match values. xG baseline needs raw events, the other
        # two use SPADL — the model's method signature is the same either way.
        input_frame = events if bid == "xg_per_shot" else actions
        preds = model.predict_team_match_values(input_frame, matches_meta)

        out_path = out_dir / "baselines" / f"{bid}__t3_team_match_values.parquet"
        preds.to_parquet(out_path, index=False)
        try:
            out_path_rel = str(out_path.relative_to(REPO))
        except ValueError:
            out_path_rel = str(out_path)   # out_dir outside repo (e.g. tests)
        baseline_results[bid] = {
            "n_predictions": int(len(preds)),
            "n_distinct_games": int(preds["game_id"].nunique()),
            "n_distinct_teams": int(preds["team_id"].nunique()),
            "value_summary": {
                "mean": float(preds["value"].mean()),
                "std": float(preds["value"].std()),
                "min": float(preds["value"].min()),
                "max": float(preds["value"].max()),
            },
            "output_path": out_path_rel,
            "duration_s": round(time.perf_counter() - t0, 3),
        }
        _step(f"baseline {bid}: {len(preds)} team-match predictions "
              f"(mean={preds['value'].mean():.2f}, max={preds['value'].max():.2f})",
              time.perf_counter() - t0)

    # --- Method dispatch (fit on train split; predict on all matches) ---
    train_actions = actions[actions["game_id"].isin(split.train)]
    train_events = events[events["game_id"].isin(split.train)]
    method_results: dict[str, dict] = {}
    for mid in method_ids:
        if mid not in METHODS_BY_ID:
            print(f"  WARNING: unknown method {mid!r}, skipping")
            continue
        t0 = time.perf_counter()
        model_cls = METHODS_BY_ID[mid]
        model = model_cls()
        needs_frames = getattr(model_cls, "requires_360", False)
        # xT-Action's fit accepts optional raw_events for real xG labelling.
        if mid == "xt_action_sumpter":
            model.fit(train_actions, matches_meta, raw_events=train_events)
        elif needs_frames:
            train_frames = (frames[frames["game_id"].isin(split.train)]
                            if frames is not None else None)
            model.fit(train_actions, matches_meta, frames=train_frames)
        else:
            model.fit(train_actions, matches_meta)
        if needs_frames:
            preds = model.predict_team_match_values(actions, matches_meta, frames=frames)
            per_action_df = model.predict_per_action(actions, frames=frames)
        else:
            preds = model.predict_team_match_values(actions, matches_meta)
            per_action_df = model.predict_per_action(actions)

        out_path = out_dir / "methods" / f"{mid}__t3_team_match_values.parquet"
        preds.to_parquet(out_path, index=False)

        # BM-12: per-action values for T1 evaluation. Include NaNs so
        # DAxT / OBSO_FF partial-coverage methods stay diagnosable downstream.
        pa_path = out_dir / "methods_per_action" / f"{mid}__per_action_values.parquet"
        per_action_df.to_parquet(pa_path, index=False)
        try:
            out_path_rel = str(out_path.relative_to(REPO))
        except ValueError:
            out_path_rel = str(out_path)
        method_results[mid] = {
            "n_predictions": int(len(preds)),
            "n_distinct_games": int(preds["game_id"].nunique()),
            "n_distinct_teams": int(preds["team_id"].nunique()),
            "value_summary": {
                "mean": float(preds["value"].mean()),
                "std":  float(preds["value"].std()),
                "min":  float(preds["value"].min()),
                "max":  float(preds["value"].max()),
            },
            "output_path": out_path_rel,
            "duration_s": round(time.perf_counter() - t0, 3),
        }
        _step(f"method {mid}: {len(preds)} team-match predictions "
              f"(mean={preds['value'].mean():.3f}, max={preds['value'].max():.3f})",
              time.perf_counter() - t0)

    # --- Manifest -----------------------------------------------------
    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - t_total, 3),
        "git_sha": _git_sha(),
        "config_path": "configs/benchmark_tasks.yaml",
        "n_train_competitions": len(train_comps),
        "n_matches_total": int(total_matches),
        "split_seed": spec.seed,
        "audit": audit,
        "baselines": baseline_results,
        "methods": method_results,
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    try:
        manifest_pretty = manifest_path.relative_to(REPO)
    except ValueError:
        manifest_pretty = manifest_path
    _step(f"manifest saved → {manifest_pretty}")

    print(f"\n=== done in {manifest['duration_s']:.1f}s ===\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--baselines", nargs="*", default=None,
        help="baseline ids to run. Default: all baselines. Empty string = none.",
    )
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="method ids to run. Default: all methods. Empty string = none.",
    )
    parser.add_argument(
        "--max-matches", type=int, default=None,
        help="Cap total matches across all competitions (for smoke runs).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--competition-index", type=int, default=None,
        help="If set, only process the competition at this 0-indexed position "
             "in the selected corpus (train or holdout). Used by "
             "scripts/run_benchmark_by_competition.py to spread memory "
             "pressure over sequential runs.",
    )
    parser.add_argument(
        "--holdout", action="store_true",
        help="Run on the T4 holdout competitions (WSL 2020/21 + AFCON 2023) "
             "instead of the training corpus.",
    )
    args = parser.parse_args()
    run(
        baseline_ids=args.baselines,
        method_ids=args.methods,
        max_matches=args.max_matches,
        out_dir=args.out_dir,
        competition_index=args.competition_index,
        holdout=args.holdout,
    )


if __name__ == "__main__":
    main()
