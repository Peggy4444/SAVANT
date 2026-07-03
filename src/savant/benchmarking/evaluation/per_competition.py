"""Per-competition T3 breakdown + robustness score.

Track-B's pooled T4 metric conflates method robustness with the difficulty
of the individual holdout competition. Decomposing ρ_T3 per competition
gives a fairer robustness signal: methods with LOW std across competitions
generalise, regardless of which holdout you chose.

This module computes per-competition ρ from the by-competition partial
outputs (`data/outputs/benchmark_partial/comp_*/`) and returns a long-form
DataFrame + a wide pivot + per-method robustness stats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    CompetitionSpec, holdout_competitions, load_config, train_competitions,
)
from ..loaders import StatsBombOpenLoader
from .metrics import pearson_rho
from .targets import team_match_xg_diff

logger = logging.getLogger(__name__)


# --- Public API ----------------------------------------------------


def competition_label(spec: CompetitionSpec) -> str:
    """Compact per-report label — 'La Liga 15/16' etc."""
    season = spec.season
    if "/" in season:
        y0, y1 = season.split("/", 1)
        season = f"{y0[-2:]}/{y1[-2:]}"
    return f"{spec.name} {season}"


def evaluate_per_competition(
    train_partial_root: Path | None = None,
    holdout_partial_root: Path | None = None,
) -> pd.DataFrame:
    """Return long-form DataFrame with per-competition ρ per method.

    Columns: corpus, index, competition, n_matches, model_id,
             pearson_rho, n_pairs.

    Returns an empty DataFrame if neither partial dir exists.
    """
    cfg = load_config()
    train_comps = train_competitions(cfg)
    holdout_comps = holdout_competitions(cfg)
    loader = StatsBombOpenLoader()

    rows: list[dict[str, Any]] = []
    all_specs = []
    if train_partial_root is not None:
        all_specs += [("train", i, spec) for i, spec in enumerate(train_comps)]
    if holdout_partial_root is not None:
        all_specs += [("holdout", i, spec) for i, spec in enumerate(holdout_comps)]

    for corpus, index, spec in all_specs:
        root = train_partial_root if corpus == "train" else holdout_partial_root
        pdir = root / f"comp_{index:02d}"
        if not pdir.exists():
            logger.warning("per_competition: %s comp %d partial missing (%s)",
                           corpus, index, pdir)
            continue

        # Load events once per competition (cached by SB loader).
        comp = loader.resolve(spec.name, spec.season)
        gids = loader.list_matches(comp)["match_id"].tolist()
        events = loader.load_events_batch(comp, gids)
        target = team_match_xg_diff(events)

        n_matches = _matches_in_partial(pdir)
        label = competition_label(spec)
        for sub in ("baselines", "methods"):
            for p in sorted((pdir / sub).glob("*__t3_team_match_values.parquet")):
                mid = p.name.replace("__t3_team_match_values.parquet", "")
                preds = pd.read_parquet(p)
                joined = preds.merge(
                    target[["game_id", "team_id", "xg_diff"]],
                    on=["game_id", "team_id"], how="inner",
                )
                if len(joined) < 4:
                    rho = float("nan")
                else:
                    rho = pearson_rho(
                        joined["value"].values, joined["xg_diff"].values,
                    )
                rows.append({
                    "corpus":       corpus,
                    "index":        index,
                    "competition":  label,
                    "n_matches":    n_matches,
                    "model_id":     mid,
                    "pearson_rho":  rho,
                    "n_pairs":      len(joined),
                })

    return pd.DataFrame(rows)


def robustness_stats(per_competition_df: pd.DataFrame) -> pd.DataFrame:
    """One row per method with:  robustness_mean_rho  robustness_std_rho
    robustness_min_rho  robustness_max_rho  n_competitions.

    Robustness = **low std across competitions**. A method with high mean
    and low std generalises better than a method with the same mean and
    high std.
    """
    if per_competition_df.empty:
        return pd.DataFrame(columns=[
            "model_id", "robustness_mean_rho", "robustness_std_rho",
            "robustness_min_rho", "robustness_max_rho", "n_competitions",
        ])
    g = per_competition_df.groupby("model_id")["pearson_rho"]
    return pd.DataFrame({
        "model_id":            g.count().index,
        "robustness_mean_rho": g.mean().values,
        "robustness_std_rho":  g.std(ddof=1).values,
        "robustness_min_rho":  g.min().values,
        "robustness_max_rho":  g.max().values,
        "n_competitions":      g.count().values,
    })


# --- Internals -----------------------------------------------------


def _matches_in_partial(pdir: Path) -> int:
    for sub in ("methods", "baselines"):
        for p in (pdir / sub).glob("*__t3_team_match_values.parquet"):
            return int(pd.read_parquet(p)["game_id"].nunique())
    return 0
