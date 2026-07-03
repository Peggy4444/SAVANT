"""Base contract for corpus-derived benchmark methods.

Structurally identical to `savant.benchmarking.baselines.Baseline`; the
distinction is documentary and lets the CLI report "baselines" and "methods"
separately in the leaderboard. Every method (xT-State, xT-Action, VAEP, DAxT,
Van Roy MDP, Van Roy Leaving Goals, PlayeRank, OBSO_FF, Liu 2020) satisfies
this Protocol.

Task projection roadmap:
- BM-5 wires `predict_team_match_values` (T3). Every method here supports T3.
- BM-12 adds `predict_per_action(actions)` (T1). Every method returns a
  DataFrame [game_id, team_id, action_id, value] aligned to `actions`.
  Methods that only fire on a subset (DAxT → defensive; OBSO_FF → 360
  freeze-frame events) emit NaN for the rest so downstream calibration /
  bootstrap can pairwise-drop them.
- T2 (multiclass possession outcome) remains a separate deliverable —
  needs possession segmentation + a per-method multiclass head.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Method(Protocol):
    """Contract for a corpus-derived benchmark method."""

    id: str
    name: str
    corpus_paper_id: str
    reports_on_tasks: list[str]

    def fit(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> None:
        """Fit model parameters from a training subset of SPADL actions."""
        ...

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> pd.DataFrame:
        """T3 projection.
        Returns: DataFrame with columns [game_id, team_id, value]."""
        ...

    def predict_per_action(
        self,
        actions: pd.DataFrame,
    ) -> pd.DataFrame:
        """T1 projection.
        Returns: DataFrame with columns [game_id, team_id, action_id, value]
        aligned row-wise to `actions`.  Actions the method doesn't score
        (e.g. non-defensive actions for DAxT) emit NaN in `value`."""
        ...
