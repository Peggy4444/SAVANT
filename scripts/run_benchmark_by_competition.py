"""Run Track-B benchmark one training competition at a time, then merge.

Rationale
---------
The single-shot orchestrator loads every competition's SPADL + events + 360
frames into RAM up-front. At the full 660-match training corpus that risks
OOM (OBSO_FF alone concatenates ~32M freeze-frame rows). By running each
competition sequentially and merging the parquet outputs, we bound peak
RAM to whichever competition happens to be the largest — La Liga 2015/16
at 380 matches.

Each competition's fit still happens INSIDE that competition (the same
train/test match split is derived from that competition's game_ids only,
using the same seed). This is a fold-per-competition regime rather than
a corpus-wide split — arguably a more honest test of "how does this
method work in this league" — but different from what a single-shot
run_benchmark.py would produce. The T3 leaderboard is computed on the
UNION of all per-competition test predictions.

Usage
-----
    PYTHONPATH=src python scripts/run_benchmark_by_competition.py

Skip competitions that have already produced parquets:
    PYTHONPATH=src python scripts/run_benchmark_by_competition.py --skip-existing

Restrict to a subset of competition indices (useful for retries):
    PYTHONPATH=src python scripts/run_benchmark_by_competition.py --indices 0 3 6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from savant.benchmarking import (
    holdout_competitions, load_config, train_competitions,
)


def _step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_one(index: int, partial_dir: Path, holdout: bool = False) -> None:
    """Kick off scripts/run_benchmark.py for one competition."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_benchmark.py"),
        "--competition-index", str(index),
        "--out-dir", str(partial_dir),
    ]
    if holdout:
        cmd.append("--holdout")
    env_pypath = f"{REPO / 'src'}"
    result = subprocess.run(
        cmd,
        cwd=REPO,
        env={"PYTHONPATH": env_pypath, "PATH": _env_path()},
        capture_output=False,     # stream to our stdout
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"run_benchmark.py failed for competition index {index} "
            f"(exit {result.returncode})",
        )


def _env_path() -> str:
    import os
    return os.environ.get("PATH", "")


def _merge_parquets(partial_dirs: list[Path], merged_dir: Path) -> None:
    """For each `{baselines,methods,methods_per_action}/*.parquet`, concat
    across all partial dirs and write to `merged_dir` under the same
    subdirectory."""
    for sub in ("baselines", "methods", "methods_per_action"):
        # Discover model-id filenames across partials.
        files_by_name: dict[str, list[Path]] = {}
        for pdir in partial_dirs:
            for f in sorted((pdir / sub).glob("*.parquet")):
                files_by_name.setdefault(f.name, []).append(f)
        if not files_by_name:
            continue
        (merged_dir / sub).mkdir(parents=True, exist_ok=True)
        for fname, paths in files_by_name.items():
            _step(f"  merging {sub}/{fname} across {len(paths)} partials")
            df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
            df.to_parquet(merged_dir / sub / fname, index=False)


def _merge_run_manifests(partial_dirs: list[Path], merged_dir: Path) -> None:
    """Concat per-competition run manifests into one summary JSON."""
    import json
    summaries = []
    for pdir in partial_dirs:
        mpath = pdir / "run_manifest.json"
        if mpath.exists():
            try:
                pdir_rel = str(pdir.relative_to(REPO))
            except ValueError:
                pdir_rel = str(pdir)   # outside the repo (e.g. /tmp for tests)
            summaries.append({
                "partial_dir": pdir_rel,
                **json.loads(mpath.read_text()),
            })
    if summaries:
        (merged_dir / "run_manifest.json").write_text(
            json.dumps({
                "partials":            summaries,
                "n_matches_total":     sum(s.get("n_matches_total", 0) for s in summaries),
                "n_train_competitions": len(summaries),
                "split_seed":          summaries[0].get("split_seed"),
                "audit": {
                    "n_matches_train": sum(s.get("audit", {}).get("n_matches_train", 0) for s in summaries),
                    "n_matches_val":   sum(s.get("audit", {}).get("n_matches_val",   0) for s in summaries),
                    "n_matches_test":  sum(s.get("audit", {}).get("n_matches_test",  0) for s in summaries),
                },
            }, indent=2, default=str) + "\n",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--partial-root", type=Path,
        default=REPO / "data" / "outputs" / "benchmark_partial",
        help="Where per-competition partial outputs land.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO / "data" / "outputs" / "benchmark",
        help="Where merged corpus outputs land.",
    )
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Competition indices to run. Default: all.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip competitions whose partial dir already has a run_manifest.json.",
    )
    parser.add_argument(
        "--wipe-out-dir", action="store_true",
        help="Delete --out-dir before merging (fresh corpus). Off by default.",
    )
    parser.add_argument(
        "--holdout", action="store_true",
        help="Run on T4 holdout competitions instead of the training corpus.",
    )
    args = parser.parse_args()

    train_comps = (holdout_competitions(load_config()) if args.holdout
                   else train_competitions(load_config()))
    indices = args.indices if args.indices is not None else list(range(len(train_comps)))
    for i in indices:
        if not (0 <= i < len(train_comps)):
            raise IndexError(f"--indices {i} out of range [0, {len(train_comps)})")

    t_total = time.perf_counter()
    _step(f"=== Track-B by-competition runner: {len(indices)} competitions ===")
    for i in indices:
        comp = train_comps[i]
        pdir = args.partial_root / f"comp_{i:02d}"
        manifest = pdir / "run_manifest.json"
        if args.skip_existing and manifest.exists():
            _step(f"→ competition {i}  ({comp.name} {comp.season}): SKIP (manifest exists)")
            continue
        _step(f"→ competition {i}  ({comp.name} {comp.season}): running ...")
        t0 = time.perf_counter()
        _run_one(i, pdir, holdout=args.holdout)
        _step(f"   done in {time.perf_counter() - t0:.1f}s")

    # --- Merge partials into the corpus-wide output dir --------------
    _step(f"=== Merging {len(indices)} partials → {args.out_dir} ===")
    if args.wipe_out_dir and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial_dirs = [args.partial_root / f"comp_{i:02d}" for i in indices]
    _merge_parquets(partial_dirs, args.out_dir)
    _merge_run_manifests(partial_dirs, args.out_dir)

    _step(f"=== Total: {time.perf_counter() - t_total:.1f}s ===")
    _step(f"Next: run `scripts/evaluate_benchmark.py --bench-dir {args.out_dir}`")


if __name__ == "__main__":
    main()
