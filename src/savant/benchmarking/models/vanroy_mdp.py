"""Van Roy MDP — Van Roy et al. (2021), "Learning a Markov reward function
for evaluating soccer decision making" (RL4RealLife workshop @ ICML 2021).

Reimplemented from the paper spec — no code release accompanies the paper
(see extraction `vanroy_2021_markov_reward`).

Simplified variant
------------------
The paper fits a hierarchical Bayesian MDP with per-team pooling (PyMC3
ADVI) plus XGBoost end-location predictors and PRISM policy checking. That
stack takes hours to fit and roughly a thousand lines of PyMC3 code — well
beyond what belongs inside a benchmark wrapper.

We keep the *load-bearing* pieces of the paper's method: a discrete-state
MDP with pass / cross / dribble / take-on / shot as on-ball actions,
per-zone empirical transition estimates, and a closed-form value function
derived from the fundamental-matrix formulation.

Concretely:

    States       : 192 pitch zones (12 rows × 16 cols) + 3 absorbing states
                   {GOAL, NO_GOAL, LOST_POSSESSION}
    Transitions  : P̂(s' | s) estimated from empirical counts on the training
                   split (Laplace-smoothed so no row has zero mass)
    Reward       : 1 for GOAL, 0 elsewhere
    Value        : V(s) = P(reach GOAL from s) under the empirical policy,
                   solved as V = (I − P_NN)⁻¹ P_NG where P_NN is the
                   nonterminal-to-nonterminal sub-matrix and P_NG the
                   nonterminal-to-goal column.

This is *not* the hierarchical Bayesian model — it is the empirical MDP
that the Bayesian model shrinks toward, i.e. the paper's own reference
point. For our benchmark this gives us a state-value V(s) ∈ [0, 1] per
zone, which is what T2 and T3 both consume.

Per-action valuation
--------------------
    V(action) = V(s_end) − V(s_start)          (successful in-play move)
    V(action) = 1 − V(s_start)                 (successful shot)
    V(action) = 0 − V(s_start)                 (failed action or missed shot)
    V(action) = 0                              (defensive / non-on-ball action)

T3 (team-match): Σ V(action) per (game_id, team_id).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SPADL pitch dimensions.
FIELD_LENGTH: float = 105.0
FIELD_WIDTH:  float = 68.0


class VanRoyMDP:
    """Reference specification: Van Roy et al. (2021).
    Corpus paper: vanroy_2021_markov_reward."""

    id: str = "vanroy_mdp"
    name: str = "Van Roy MDP (Van Roy 2021, simplified MDP variant)"
    corpus_paper_id: str = "vanroy_2021_markov_reward"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
        "t2_state_value_predicts_possession_outcome",
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    def __init__(
        self,
        n_cols: int = 16,
        n_rows: int = 12,
        smoothing: float = 1.0,
    ) -> None:
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.smoothing = smoothing
        self.V: np.ndarray | None = None            # shape (n_zones,)
        self._on_ball_ids: set[int] = set()
        self._shot_ids: set[int] = set()
        self._success_id: int = -1

    @property
    def n_zones(self) -> int:
        return self.n_cols * self.n_rows

    # ---------------------------------------------------------------

    def _zone_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = np.clip(np.floor(x * self.n_cols / FIELD_LENGTH).astype(int),
                      0, self.n_cols - 1)
        row = np.clip(np.floor(y * self.n_rows / FIELD_WIDTH).astype(int),
                      0, self.n_rows - 1)
        return row * self.n_cols + col

    def _load_spadl_labels(self) -> None:
        from socceraction.spadl import config as spadlcfg
        self._on_ball_ids = {
            spadlcfg.actiontypes.index(n)
            for n in ("pass", "cross", "dribble", "take_on",
                      "shot", "shot_penalty", "shot_freekick")
        }
        self._shot_ids = {
            spadlcfg.actiontypes.index(n)
            for n in ("shot", "shot_penalty", "shot_freekick")
        }
        self._success_id = spadlcfg.results.index("success")

    # ---------------------------------------------------------------

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame | None = None) -> None:
        self._load_spadl_labels()

        GOAL, NO_GOAL, LOST = self.n_zones, self.n_zones + 1, self.n_zones + 2
        counts = np.zeros((self.n_zones, self.n_zones + 3))

        types = actions["type_id"].values
        results = actions["result_id"].values
        on_ball = np.isin(types, list(self._on_ball_ids))
        is_shot = np.isin(types, list(self._shot_ids))
        is_success = results == self._success_id

        sub = actions[on_ball]
        if len(sub) == 0:
            # degenerate — no on-ball actions to estimate from
            self.V = np.zeros(self.n_zones)
            return

        z_start = self._zone_of(
            sub["start_x"].values.astype(float),
            sub["start_y"].values.astype(float),
        )
        z_end = self._zone_of(
            sub["end_x"].values.astype(float),
            sub["end_y"].values.astype(float),
        )
        sub_types = types[on_ball]
        sub_results = results[on_ball]
        sub_is_shot = np.isin(sub_types, list(self._shot_ids))
        sub_is_success = sub_results == self._success_id

        # Successful shots → GOAL, missed shots → NO_GOAL.
        shot_goal_mask = sub_is_shot & sub_is_success
        shot_miss_mask = sub_is_shot & ~sub_is_success
        np.add.at(counts, (z_start[shot_goal_mask], GOAL),    1)
        np.add.at(counts, (z_start[shot_miss_mask], NO_GOAL), 1)

        # Successful non-shot on-ball actions → next zone.
        move_ok = ~sub_is_shot & sub_is_success
        np.add.at(counts, (z_start[move_ok], z_end[move_ok]), 1)

        # Failed non-shot on-ball actions → LOST_POSSESSION.
        move_fail = ~sub_is_shot & ~sub_is_success
        np.add.at(counts, (z_start[move_fail], LOST), 1)

        # Laplace smoothing so every row sums to > 0.
        counts += self.smoothing / (self.n_zones + 3)
        P = counts / counts.sum(axis=1, keepdims=True)

        # Nonterminal-nonterminal sub-block and nonterminal→GOAL column.
        P_NN = P[:, :self.n_zones]           # (n_zones, n_zones)
        P_NG = P[:, GOAL]                    # (n_zones,)

        # V = (I - P_NN)^{-1} P_NG. This is the probability of eventually
        # reaching GOAL from each nonterminal state under the empirical
        # policy — i.e. the paper's state-value V_π.
        I = np.eye(self.n_zones)
        self.V = np.linalg.solve(I - P_NN, P_NG)

    # ---------------------------------------------------------------

    def _per_action_values(self, actions: pd.DataFrame) -> np.ndarray:
        """Δ-V per action.  Non-on-ball actions get 0."""
        assert self.V is not None, "call fit() first"
        n = len(actions)
        if n == 0:
            return np.zeros(0)

        types = actions["type_id"].values
        results = actions["result_id"].values
        on_ball_mask = np.isin(types, list(self._on_ball_ids))
        is_shot = np.isin(types, list(self._shot_ids))
        is_success = results == self._success_id

        z_start = self._zone_of(
            actions["start_x"].values.astype(float),
            actions["start_y"].values.astype(float),
        )
        z_end = self._zone_of(
            actions["end_x"].values.astype(float),
            actions["end_y"].values.astype(float),
        )

        v_start = self.V[z_start]
        v_end   = self.V[z_end]

        vals = np.zeros(n)
        # successful in-play moves
        move_ok = on_ball_mask & ~is_shot & is_success
        vals[move_ok] = v_end[move_ok] - v_start[move_ok]
        # goals
        shot_goal = on_ball_mask & is_shot & is_success
        vals[shot_goal] = 1.0 - v_start[shot_goal]
        # missed shots / failed moves — arrive at absorbing 0 state
        shot_miss = on_ball_mask & is_shot & ~is_success
        move_fail = on_ball_mask & ~is_shot & ~is_success
        vals[shot_miss | move_fail] = 0.0 - v_start[shot_miss | move_fail]
        return vals

    def predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """T1 projection. Δ-V per action; 0 for non-on-ball actions."""
        return pd.DataFrame({
            "game_id":   actions["game_id"].values,
            "team_id":   actions["team_id"].values,
            "action_id": actions["action_id"].values,
            "value":     self._per_action_values(actions).astype(float),
        })

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        vals = self._per_action_values(actions)
        df = actions[["game_id", "team_id"]].copy()
        df["value"] = vals
        agg = df.groupby(["game_id", "team_id"], as_index=False)["value"].sum()
        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
