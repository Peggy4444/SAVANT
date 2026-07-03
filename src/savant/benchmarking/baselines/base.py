"""Base contract for Track-B benchmark baselines (and, later, methods).

Every baseline and every full method runner satisfies this Protocol so the
`scripts/run_benchmark.py` dispatch loop stays uniform. The Protocol is
minimal by design — task-specific projections are added incrementally as
each method wrapper (BM-6 onwards) needs them.

Task-projection method mapping:
- `predict_team_match_values(actions, matches)` → shape for T3 (Pearson ρ vs match xG diff)
- `predict_per_action_values(actions)`         → shape for T1 (Brier vs P(∑ xG in next k))
- `predict_state_values(actions)`              → shape for T2 (state → outcome multi-class)

Baselines only need to implement the projections they actually report on
(as specified in `configs/benchmark_tasks.yaml`'s `reports_on_tasks`).
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Baseline(Protocol):
    """Minimum interface satisfied by every baseline (and every method)."""

    id: str
    name: str
    reports_on_tasks: list[str]

    def fit(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> None:
        """Optional training. Trivial baselines are no-ops."""
        ...

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> pd.DataFrame:
        """T3 projection.
        Returns: DataFrame with columns [game_id, team_id, value]."""
        ...
