"""Van Roy et al. (2021) — Leaving Goals on the Pitch (MIT Sloan 2021).

Reimplemented from the paper spec — no code release (see extraction
`vanroy_2021_leaving_goals`).

Simplified variant
------------------
The paper's headline contribution is a *per-team* MDP whose transitions
and policy are learned from that team's own event data, plus a
counterfactual policy-alteration analysis (PRISM model checker, GBDT
end-location predictor, expected number of state visits from the
fundamental matrix N = (I − Q)⁻¹). The counterfactual is a policy
comparison, not a per-action value function, so it doesn't map onto
SAVANT's four task metrics directly.

What we keep — the load-bearing pieces the paper actually publishes
predictions from — is the per-team MDP itself:

    States       : 192 pitch zones (12 rows × 16 cols) + {GOAL, NO_GOAL,
                   LOST_POSSESSION}                                       [same as VanRoyMDP]
    Transitions  : learned SEPARATELY per team from that team's SPADL
                   actions in the training split                          [core LG contribution]
    Value        : V_team(s) = P(reach GOAL from s | that team's empirical
                   policy), closed-form from (I − P_NN)⁻¹ P_NG            [same math as VanRoyMDP]

At predict time, each action is valued with V_team of the *acting team*
— so team A's pass from zone z gets team A's V-delta, not the pooled
one. Teams never seen in training back off to a global-pool fallback.

Δ-per-action semantics are identical to VanRoyMDP; only the source of
V differs.

Deviations from paper spec
--------------------------
- Paper uses 22×17 offensive grid + 1 defensive-half state (375 states);
  we use the regular 12×16 = 192-zone grid to stay aligned with
  VanRoyMDP and xT-State so the four Track-B methods share a comparable
  spatial resolution.
- No GBDT end-location predictor for failed actions — we lump them into
  LOST_POSSESSION as the empirical transition suggests.
- No counterfactual policy alteration (PRISM). That analysis is a
  separate deliverable, not a per-action valuation. See §4.3–4.4 of the
  paper if you want to reconstruct it downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .vanroy_mdp import FIELD_LENGTH, FIELD_WIDTH, VanRoyMDP


class VanRoyLeavingGoals:
    """Reference specification: Van Roy et al. (2021), MIT Sloan.
    Corpus paper: vanroy_2021_leaving_goals."""

    id: str = "vanroy_leaving_goals"
    name: str = "Van Roy Leaving Goals (per-team MDP variant)"
    corpus_paper_id: str = "vanroy_2021_leaving_goals"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
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
        # V per team_id; fallback pooled V for teams unseen in training.
        self.V_by_team: dict[int, np.ndarray] = {}
        self.V_pool: np.ndarray | None = None
        self._on_ball_ids: set[int] = set()
        self._shot_ids: set[int] = set()
        self._success_id: int = -1

    @property
    def n_zones(self) -> int:
        return self.n_cols * self.n_rows

    # ---------------------------------------------------------------
    # Fit: one MDP per team + a pooled fallback.
    # ---------------------------------------------------------------

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame | None = None) -> None:
        # Reuse VanRoyMDP for the actual per-team fit — one wrapper, same math.
        pooled = VanRoyMDP(
            n_cols=self.n_cols, n_rows=self.n_rows, smoothing=self.smoothing,
        )
        pooled.fit(actions)
        self.V_pool = pooled.V
        # Cache spadl label ids from the pooled instance so predict() doesn't
        # need to reload them.
        self._on_ball_ids = pooled._on_ball_ids
        self._shot_ids    = pooled._shot_ids
        self._success_id  = pooled._success_id

        # Per-team fits.
        self.V_by_team = {}
        for team_id, team_actions in actions.groupby("team_id"):
            model = VanRoyMDP(
                n_cols=self.n_cols, n_rows=self.n_rows, smoothing=self.smoothing,
            )
            model.fit(team_actions)
            if model.V is not None:
                self.V_by_team[int(team_id)] = model.V

    # ---------------------------------------------------------------

    def _zone_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = np.clip(np.floor(x * self.n_cols / FIELD_LENGTH).astype(int),
                      0, self.n_cols - 1)
        row = np.clip(np.floor(y * self.n_rows / FIELD_WIDTH).astype(int),
                      0, self.n_rows - 1)
        return row * self.n_cols + col

    def _per_action_values(self, actions: pd.DataFrame) -> np.ndarray:
        """Δ-V per action using the actor team's V; pooled V for unseen teams."""
        assert self.V_pool is not None, "call fit() first"
        n = len(actions)
        if n == 0:
            return np.zeros(0)

        types = actions["type_id"].values
        results = actions["result_id"].values
        team_ids = actions["team_id"].values
        on_ball_mask = np.isin(types, list(self._on_ball_ids))
        is_shot      = np.isin(types, list(self._shot_ids))
        is_success   = results == self._success_id

        z_start = self._zone_of(
            actions["start_x"].values.astype(float),
            actions["start_y"].values.astype(float),
        )
        z_end = self._zone_of(
            actions["end_x"].values.astype(float),
            actions["end_y"].values.astype(float),
        )

        # V vector per row: pick this team's V, fall back to pooled.
        V_rows = np.empty((n, self.n_zones))
        for i, tid in enumerate(team_ids):
            V_rows[i] = self.V_by_team.get(int(tid), self.V_pool)
        v_start = V_rows[np.arange(n), z_start]
        v_end   = V_rows[np.arange(n), z_end]

        vals = np.zeros(n)
        move_ok = on_ball_mask & ~is_shot & is_success
        vals[move_ok] = v_end[move_ok] - v_start[move_ok]
        shot_goal = on_ball_mask & is_shot & is_success
        vals[shot_goal] = 1.0 - v_start[shot_goal]
        shot_miss = on_ball_mask & is_shot & ~is_success
        move_fail = on_ball_mask & ~is_shot & ~is_success
        vals[shot_miss | move_fail] = 0.0 - v_start[shot_miss | move_fail]
        return vals

    def predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """T1 projection. Same Δ-V as VanRoyMDP but using per-team V."""
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
