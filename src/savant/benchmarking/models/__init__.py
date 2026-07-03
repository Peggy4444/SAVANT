"""Corpus-derived benchmark methods for Track B.

Each method wrapper:
- Reimplements a specification from the corpus paper (or its canonical
  reference implementation, e.g. Sumpter's Soccermatics tutorial for xT).
- Satisfies the `Method` Protocol so `scripts/run_benchmark.py` can
  dispatch it identically to a baseline.
- Cites the corpus paper via `corpus_paper_id` matching `data/outputs/
  extractions/{paper_id}.json`.

Registry follows the same pattern as `savant.benchmarking.baselines`.
"""

from .base import Method
from .daxt import DAxT
from .liu_2020 import Liu2020
from .obso_ff import OBSO_FF
from .playerank import PlayeRank
from .vaep import VAEP
from .vanroy_leaving_goals import VanRoyLeavingGoals
from .vanroy_mdp import VanRoyMDP
from .xt_action import XTAction
from .xt_state import XTState

METHODS_BY_ID = {
    XTState.id:            XTState,
    XTAction.id:           XTAction,
    VAEP.id:               VAEP,
    DAxT.id:               DAxT,
    VanRoyMDP.id:          VanRoyMDP,
    VanRoyLeavingGoals.id: VanRoyLeavingGoals,
    PlayeRank.id:          PlayeRank,
    OBSO_FF.id:            OBSO_FF,
    Liu2020.id:            Liu2020,
}

__all__ = [
    "DAxT",
    "Liu2020",
    "METHODS_BY_ID",
    "Method",
    "OBSO_FF",
    "PlayeRank",
    "VAEP",
    "VanRoyLeavingGoals",
    "VanRoyMDP",
    "XTAction",
    "XTState",
]
