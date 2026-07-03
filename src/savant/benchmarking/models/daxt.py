"""DAxT — Defensive Action Expected Threat, Merhej et al. (2021).

Reimplemented from the paper spec — no code was released alongside the
publication (see extraction `merhej_2021_daxt`).

Idea
----
Value a defensive action (interception / tackle) by predicting the xT of
the *next* on-ball action that would have happened had the defence not
intervened. In other words: how threatening a state did the defender
prevent the opposing team from reaching?

Training targets
----------------
Let xT(e) ≜ xT_map[end_zone(e)] be the state-value the action leads to.
Given a sliding window of a = 2 preceding actions, learn

    f: (feats_of_actions_{i-2, i-1}) → xT(action_i)

on ALL positions in the event stream (not just defensive actions). Then
at inference on a defensive action at position j, feed features of
actions_{j-2, j-1} into f to get the predicted "xT of what would have
happened next".

Deviations from paper spec
--------------------------
- Paper uses Keras MLP with ELU + MAE loss + Adadelta. We use scikit-learn
  MLPRegressor (ReLU + MSE + Adam) so the wrapper stays inside our
  benchmarking deps (no TF/Keras). Same three hidden layers of 10 neurons.
  Squared vs absolute error changes the loss surface slightly but not the
  qualitative rankings.
- Paper's "xT scalar per event" is not fully specified; we use xT of the
  action's end zone, which is Singh's own state-value interpretation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .xt_state import XTState


class DAxT:
    """Reference specification: Merhej et al. (2021) KDD paper.
    Corpus paper: merhej_2021_daxt."""

    id: str = "daxt_merhej"
    name: str = "DAxT (Merhej 2021, event-data defensive-action xT)"
    corpus_paper_id: str = "merhej_2021_daxt"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    def __init__(
        self,
        window_size: int = 2,               # `a` in the paper (Experiment 1 optimum)
        xt_n_cols: int = 16,
        xt_n_rows: int = 12,
        xt_n_iterations: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (10, 10, 10),
        max_iter: int = 50,
        random_state: int = 0,
    ) -> None:
        self.window_size = window_size
        self.xt_n_cols = xt_n_cols
        self.xt_n_rows = xt_n_rows
        self.xt_n_iterations = xt_n_iterations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.xt_state: XTState | None = None
        self.scaler: Any = None
        self.mlp: Any = None

    # ---------------------------------------------------------------
    # xT scalar per action = xt_map at the action's end zone (bounded in [0,1]).
    # ---------------------------------------------------------------

    def _xt_scalar(self, actions: pd.DataFrame) -> np.ndarray:
        assert self.xt_state is not None and self.xt_state.xt_map is not None
        end_r, end_c = self.xt_state._zone_of(
            actions["end_x"].values.astype(float),
            actions["end_y"].values.astype(float),
        )
        return self.xt_state.xt_map[end_r, end_c]

    def _build_windows(
        self,
        sorted_actions: pd.DataFrame,
        xt_scalar: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, target_positions). Windows crossing game_id boundaries
        are dropped, so labels never leak from a different match."""
        game_ids = sorted_actions["game_id"].values
        start_x = sorted_actions["start_x"].values.astype(float)
        start_y = sorted_actions["start_y"].values.astype(float)
        end_x = sorted_actions["end_x"].values.astype(float)
        end_y = sorted_actions["end_y"].values.astype(float)

        a = self.window_size
        X_rows: list[list[float]] = []
        y_rows: list[float] = []
        pos_rows: list[int] = []
        for i in range(a, len(sorted_actions)):
            # window must stay inside one match
            if game_ids[i] != game_ids[i - a]:
                continue
            feats: list[float] = []
            for k in range(1, a + 1):
                j = i - k
                feats.extend([xt_scalar[j], start_x[j], start_y[j], end_x[j], end_y[j]])
            X_rows.append(feats)
            y_rows.append(xt_scalar[i])
            pos_rows.append(i)

        return (
            np.asarray(X_rows, dtype=float),
            np.asarray(y_rows, dtype=float),
            np.asarray(pos_rows, dtype=int),
        )

    # ---------------------------------------------------------------

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame | None = None) -> None:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import MinMaxScaler

        # 1) Fit the composed xT-State on the same training slice.
        self.xt_state = XTState(
            n_cols=self.xt_n_cols,
            n_rows=self.xt_n_rows,
            n_iterations=self.xt_n_iterations,
        )
        self.xt_state.fit(actions)

        # 2) Sort actions so windows respect within-match temporal order.
        sorted_actions = actions.sort_values(
            ["game_id", "period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)
        xt_scalar = self._xt_scalar(sorted_actions)

        X, y, _ = self._build_windows(sorted_actions, xt_scalar)
        if len(X) == 0:
            self.mlp = None
            return

        self.scaler = MinMaxScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",       # paper uses ELU — see docstring
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(Xs, y)

    # ---------------------------------------------------------------

    def _predict_defensive(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Return per-action DAxT value for defensive actions only.
        Non-defensive actions are excluded (DAxT is a defensive metric)."""
        from socceraction.spadl import config as spadlcfg

        if self.mlp is None or self.xt_state is None:
            return pd.DataFrame(columns=["game_id", "team_id", "value"])

        defensive_ids = {
            spadlcfg.actiontypes.index("tackle"),
            spadlcfg.actiontypes.index("interception"),
        }

        sorted_actions = actions.sort_values(
            ["game_id", "period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)
        xt_scalar = self._xt_scalar(sorted_actions)
        X, _, positions = self._build_windows(sorted_actions, xt_scalar)
        if len(X) == 0:
            return pd.DataFrame(columns=["game_id", "team_id", "value"])

        preds = self.mlp.predict(self.scaler.transform(X))

        type_ids = sorted_actions["type_id"].values[positions]
        mask = np.isin(type_ids, list(defensive_ids))
        return pd.DataFrame({
            "game_id": sorted_actions["game_id"].values[positions][mask],
            "team_id": sorted_actions["team_id"].values[positions][mask],
            "value":   preds[mask].astype(float),
        })

    def predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """T1 projection. NaN for non-defensive actions (DAxT is
        defensive-only); numeric DAxT value for tackles/interceptions."""
        # Start with NaN for every input action, then merge in defensive scores.
        out = pd.DataFrame({
            "game_id":   actions["game_id"].values,
            "team_id":   actions["team_id"].values,
            "action_id": actions["action_id"].values,
            "value":     np.nan,
        })
        defensive = self._predict_defensive(actions)
        if len(defensive) == 0:
            return out
        # Defensive frame lacks action_id; re-derive by joining on the
        # sorted position → action_id we already have from _build_windows.
        # Easier: recompute defensive with action_id included, inline.
        from socceraction.spadl import config as spadlcfg
        if self.mlp is None or self.xt_state is None:
            return out
        defensive_ids = {
            spadlcfg.actiontypes.index("tackle"),
            spadlcfg.actiontypes.index("interception"),
        }
        sorted_actions = actions.sort_values(
            ["game_id", "period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)
        xt_scalar = self._xt_scalar(sorted_actions)
        X, _, positions = self._build_windows(sorted_actions, xt_scalar)
        if len(X) == 0:
            return out
        preds = self.mlp.predict(self.scaler.transform(X))
        type_ids = sorted_actions["type_id"].values[positions]
        mask = np.isin(type_ids, list(defensive_ids))
        defensive_full = pd.DataFrame({
            "game_id":   sorted_actions["game_id"].values[positions][mask],
            "action_id": sorted_actions["action_id"].values[positions][mask],
            "value":     preds[mask].astype(float),
        })
        # Merge on (game_id, action_id) — action_id is only unique WITHIN a
        # game, so a single-key merge produces cross-game duplicates.
        out = out.drop(columns=["value"]).merge(
            defensive_full, on=["game_id", "action_id"], how="left",
        )
        return out[["game_id", "team_id", "action_id", "value"]]

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        per_action = self._predict_defensive(actions)
        agg = (per_action.groupby(["game_id", "team_id"], as_index=False)["value"]
                           .sum())
        # Include zero-value (game, team) pairs for teams with no defensive
        # actions (rare, but keeps T3 downstream unbiased).
        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
