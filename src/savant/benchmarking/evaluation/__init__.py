"""Track-B evaluation module.

Consumes per-method / per-baseline parquets produced by
`scripts/run_benchmark.py`, joins them against StatsBomb-native ground
truth, and emits a leaderboard + statistical significance table.
"""

from .bootstrap import bootstrap_ci_by_match
from .metrics import (
    METRIC_FUNCS, brier, ece, ece_multiclass, log_loss_binary,
    log_loss_multiclass, mae, pearson_rho, roc_auc, score, spearman_rho,
    top1_accuracy,
)
from .runner import (
    T1_METRICS, T2_METRICS, T3_METRICS, evaluate_t1, evaluate_t2,
    evaluate_t3, evaluate_t4, load_per_action_predictions, load_predictions,
    run,
)
from .stats import wilcoxon_pairwise
from .targets import (
    T2_OUTCOME_CLASSES, T2_OUTCOME_INDEX, t1_goal_next_k_actions,
    t2_possession_outcome, team_match_xg, team_match_xg_diff,
)

__all__ = [
    "METRIC_FUNCS",
    "T1_METRICS",
    "T2_METRICS",
    "T2_OUTCOME_CLASSES",
    "T2_OUTCOME_INDEX",
    "T3_METRICS",
    "bootstrap_ci_by_match",
    "brier",
    "ece",
    "ece_multiclass",
    "evaluate_t1",
    "evaluate_t2",
    "evaluate_t3",
    "evaluate_t4",
    "load_per_action_predictions",
    "load_predictions",
    "log_loss_binary",
    "log_loss_multiclass",
    "mae",
    "pearson_rho",
    "roc_auc",
    "run",
    "score",
    "spearman_rho",
    "t1_goal_next_k_actions",
    "t2_possession_outcome",
    "team_match_xg",
    "team_match_xg_diff",
    "top1_accuracy",
    "wilcoxon_pairwise",
]
