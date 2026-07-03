"""xT-State — Karun Singh (2018) zone-based Markov Expected Threat.

Reimplemented from the Sumpter Soccermatics tutorial spec at
https://soccermatics.readthedocs.io/en/latest/gallery/lesson4/plot_ExpectedThreat.html
(source under CC-BY-NC-SA; our reimplementation is MIT).

Algorithm
---------
Divide the pitch (SPADL 105m × 68m, attacking left→right) into an M×N grid
(defaults 16×12, matching Van Roy et al. 2020's replication of Singh).
Compute per-zone empirical rates:

    s(z) = P(shot | possession is in zone z)
    m(z) = P(move | possession is in zone z)          — where a move is a
                                                        successful pass, dribble,
                                                        cross, or take-on
    g(z) = P(goal | shot from zone z)                  — the "shot xG-like" term
    T(z, z') = P(next zone = z' | move from z)         — transition matrix

Then iterate to convergence (~5 iterations):

    xT(z) = s(z) * g(z)  +  m(z) * Σ_{z'} T(z, z') * xT(z')

Per-action value
----------------
For every SPADL action, V(action) = xT(z_end) − xT(z_start). This is
positive for actions that move the ball to a more threatening zone,
negative otherwise. Failed actions get value 0 (they don't complete the
move, so no threat is delivered).

Team-match value (T3): Σ V(action) over the team's actions in that match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SPADL pitch dimensions (metres). Match `spadl.config.field_length/width`.
FIELD_LENGTH: float = 105.0
FIELD_WIDTH: float = 68.0


class XTState:
    """Reference specification: Sumpter Soccermatics `plot_ExpectedThreat`.
    Corpus paper: singh_2018_xt."""

    id: str = "xt_state_singh"
    name: str = "xT-State (Singh 2018 zone-based Markov)"
    corpus_paper_id: str = "singh_2018_xt"
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
        n_iterations: int = 5,
    ) -> None:
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_iterations = n_iterations
        self.xt_map: np.ndarray | None = None       # (n_rows, n_cols)
        self._shot_type_ids: set[int] = set()
        self._move_type_ids: set[int] = set()
        self._success_result_id: int = -1

    # ---------------------------------------------------------------

    def _load_spadl_labels(self) -> None:
        """Populate SPADL type/result ids from the installed socceraction."""
        from socceraction.spadl import config as spadlcfg
        self._shot_type_ids = {
            i for i, name in enumerate(spadlcfg.actiontypes)
            if name in ("shot", "shot_freekick", "shot_penalty")
        }
        self._move_type_ids = {
            i for i, name in enumerate(spadlcfg.actiontypes)
            if name in ("pass", "cross", "dribble", "take_on",
                        "freekick_short", "freekick_crossed",
                        "corner_short", "corner_crossed", "throw_in")
        }
        self._success_result_id = spadlcfg.results.index("success")

    def _zone_of(self, x: float | np.ndarray, y: float | np.ndarray):
        """Map SPADL (x, y) → (row, col) grid indices. Clipped to bounds."""
        col = np.clip(
            np.floor(x * self.n_cols / FIELD_LENGTH).astype(int),
            0, self.n_cols - 1,
        )
        row = np.clip(
            np.floor(y * self.n_rows / FIELD_WIDTH).astype(int),
            0, self.n_rows - 1,
        )
        return row, col

    # ---------------------------------------------------------------

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame | None = None) -> None:
        """Estimate the xT map from SPADL actions."""
        self._load_spadl_labels()

        if len(actions) == 0:
            self.xt_map = np.zeros((self.n_rows, self.n_cols))
            return

        # Per-zone counts.
        start_r, start_c = self._zone_of(
            actions["start_x"].values.astype(float),
            actions["start_y"].values.astype(float),
        )
        end_r, end_c = self._zone_of(
            actions["end_x"].values.astype(float),
            actions["end_y"].values.astype(float),
        )
        types = actions["type_id"].values
        results = actions["result_id"].values

        is_shot = np.isin(types, list(self._shot_type_ids))
        is_move = np.isin(types, list(self._move_type_ids))
        is_success = results == self._success_result_id
        is_goal = is_shot & is_success                       # goal = successful shot

        # Aggregate to per-zone rates.
        shot_count = np.zeros((self.n_rows, self.n_cols))
        goal_count = np.zeros((self.n_rows, self.n_cols))
        move_count = np.zeros((self.n_rows, self.n_cols))
        total_count = np.zeros((self.n_rows, self.n_cols))
        # transition matrix: (from_zone, to_zone) — flattened for indexing
        n_zones = self.n_rows * self.n_cols
        T = np.zeros((n_zones, n_zones))
        move_success_count = np.zeros(n_zones)

        # numpy accumulate
        np.add.at(shot_count,  (start_r[is_shot],  start_c[is_shot]),  1)
        np.add.at(goal_count,  (start_r[is_goal],  start_c[is_goal]),  1)
        np.add.at(move_count,  (start_r[is_move],  start_c[is_move]),  1)
        np.add.at(total_count, (start_r,           start_c),           1)

        # Successful moves — transitions.
        move_success = is_move & is_success
        from_flat = (start_r[move_success] * self.n_cols
                     + start_c[move_success])
        to_flat = (end_r[move_success] * self.n_cols
                   + end_c[move_success])
        np.add.at(T, (from_flat, to_flat), 1)
        np.add.at(move_success_count, from_flat, 1)

        # Normalise
        with np.errstate(invalid="ignore", divide="ignore"):
            s_z = np.where(total_count > 0, shot_count / total_count, 0.0)
            m_z = np.where(total_count > 0, move_count / total_count, 0.0)
            g_z = np.where(shot_count > 0, goal_count / shot_count, 0.0)
            T = np.where(move_success_count[:, None] > 0,
                          T / np.clip(move_success_count[:, None], 1e-9, None),
                          0.0)

        # Bellman iteration
        xt = np.zeros((self.n_rows, self.n_cols))
        for _ in range(self.n_iterations):
            xt_flat = xt.flatten()
            expected_move_value = (T @ xt_flat).reshape(self.n_rows, self.n_cols)
            xt = s_z * g_z + m_z * expected_move_value

        self.xt_map = xt

    # ---------------------------------------------------------------

    def _per_action_values(self, actions: pd.DataFrame) -> np.ndarray:
        """xT delta per action (0 for failed and non-move actions)."""
        assert self.xt_map is not None, "call fit() first"
        start_r, start_c = self._zone_of(
            actions["start_x"].values.astype(float),
            actions["start_y"].values.astype(float),
        )
        end_r, end_c = self._zone_of(
            actions["end_x"].values.astype(float),
            actions["end_y"].values.astype(float),
        )
        delta = self.xt_map[end_r, end_c] - self.xt_map[start_r, start_c]

        # Only credit successful moves; shots and failures are 0.
        types = actions["type_id"].values
        results = actions["result_id"].values
        is_move = np.isin(types, list(self._move_type_ids))
        is_success = results == self._success_result_id
        delta = np.where(is_move & is_success, delta, 0.0)
        return delta

    def predict_per_action(
        self,
        actions: pd.DataFrame,
    ) -> pd.DataFrame:
        """T1 projection. xT delta per SPADL action; 0 for failed/non-move actions."""
        assert self.xt_map is not None, "call fit() first"
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
        """T3 projection: Σ per-action xT delta per (game, team)."""
        assert self.xt_map is not None, "call fit() first"
        actions = actions.copy()
        actions["xt_value"] = self._per_action_values(actions)
        agg = (actions.groupby(["game_id", "team_id"], as_index=False)["xt_value"]
                      .sum()
                      .rename(columns={"xt_value": "value"}))

        # Include zero-value (game, team) pairs so downstream Pearson ρ is
        # not biased by silent drops.
        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
