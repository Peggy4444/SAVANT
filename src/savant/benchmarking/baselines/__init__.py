"""Trivial baselines for Track-B benchmark leaderboard.

Every method in the benchmark must beat these on the tasks they cover
(otherwise the method is not adding signal). See CLAUDE.md §5 Stage 6.
"""

from .assists import AssistsPer90Baseline
from .base import Baseline
from .goals import GoalsPer90Baseline
from .xg import XGPerShotBaseline

BASELINES_BY_ID = {
    GoalsPer90Baseline.id:   GoalsPer90Baseline,
    AssistsPer90Baseline.id: AssistsPer90Baseline,
    XGPerShotBaseline.id:    XGPerShotBaseline,
}

__all__ = [
    "AssistsPer90Baseline",
    "BASELINES_BY_ID",
    "Baseline",
    "GoalsPer90Baseline",
    "XGPerShotBaseline",
]
