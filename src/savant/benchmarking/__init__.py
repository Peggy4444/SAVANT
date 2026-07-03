"""SAVANT Track-B — StatsBomb Open + 360 benchmark for soccer AV methods.

Shipped so far:
- `config`  — parses configs/benchmark_tasks.yaml (single seam).
- `loaders.statsbomb_open` — cached wrapper around statsbombpy (events + 360).
- `spadl`  — cached SPADL loader (socceraction).
- `splits` — deterministic match-level splitter + fail-fast leakage audit.
- `baselines` — goals-per-90, assists-per-90, xG-per-shot.
- `models` — xt_state_singh, xt_action_sumpter, vaep_socceraction,
             daxt_merhej, vanroy_mdp, vanroy_leaving_goals,
             playerank_pappalardo, obso_ff_spearman, liu_2020_ttdp.

Coming:
- `evaluation` — Brier / ECE / Spearman / Jaccard / bootstrap CIs.
- `leaderboard` — writes to data/outputs/benchmark/ + UI write-back.
"""

from .baselines import (
    BASELINES_BY_ID, AssistsPer90Baseline, Baseline, GoalsPer90Baseline,
    XGPerShotBaseline,
)
from .config import (
    CompetitionSpec, SplitSpec, cache_dir, holdout_competitions, load_config,
    methods, split_spec, three_sixty_enabled, train_competitions,
)
from .models import (
    DAxT, Liu2020, METHODS_BY_ID, Method, OBSO_FF, PlayeRank, VAEP,
    VanRoyLeavingGoals, VanRoyMDP, XTAction, XTState,
)
from .loaders import (
    StatsBombOpenLoader, load_events_for_competitions, resolve_competition,
)
from .spadl import (
    SPADLCompetition, SPADLLoader, spadl_action_types, spadl_bodyparts,
    spadl_field_dimensions, spadl_results,
)
from .splits import LeakageError, MatchSplit, leakage_audit, split_matches

__all__ = [
    "AssistsPer90Baseline",
    "BASELINES_BY_ID",
    "Baseline",
    "CompetitionSpec",
    "DAxT",
    "GoalsPer90Baseline",
    "LeakageError",
    "Liu2020",
    "MatchSplit",
    "METHODS_BY_ID",
    "Method",
    "OBSO_FF",
    "PlayeRank",
    "SPADLCompetition",
    "SPADLLoader",
    "SplitSpec",
    "StatsBombOpenLoader",
    "VAEP",
    "VanRoyLeavingGoals",
    "VanRoyMDP",
    "XGPerShotBaseline",
    "XTAction",
    "XTState",
    "cache_dir",
    "holdout_competitions",
    "leakage_audit",
    "load_config",
    "load_events_for_competitions",
    "methods",
    "resolve_competition",
    "spadl_action_types",
    "spadl_bodyparts",
    "spadl_field_dimensions",
    "spadl_results",
    "split_matches",
    "split_spec",
    "three_sixty_enabled",
    "train_competitions",
]
