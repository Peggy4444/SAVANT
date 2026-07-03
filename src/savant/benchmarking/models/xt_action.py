"""xT-Action — Sumpter's action-based P(shot | a) × xG(shot | a) variant.

Reimplemented from the Sumpter Soccermatics tutorial spec at
https://soccermatics.readthedocs.io/en/latest/lesson4/xTAction.html
(source under CC-BY-NC-SA; our reimplementation is MIT).

Methodological delta vs xt_state_singh
--------------------------------------
- Direct per-action target: V(a) = P(shot | a) × xG(shot | a).
- Two learned models (logistic classifier + linear regressor) rather than
  a zone-value map with Bellman recursion.
- No zone map; features are (start_x, start_y, end_x, end_y) + one-hot of
  a small action-type family (pass, dribble, cross, take_on).

Label construction
------------------
For each action a in the training set:
- Look at the NEXT k=10 SPADL actions in the same game by the SAME team,
  in temporal order (matches the T1 spec's k=10 horizon).
- If any of those is a SHOT with a recorded StatsBomb xG, then
      y_shot = 1
      y_xg   = the shot's xG
  otherwise
      y_shot = 0
      y_xg   = 0

Because SPADL drops StatsBomb's per-shot xG, this model can only be trained
when we have access to raw StatsBomb events for xG labelling. The fit()
method therefore accepts an optional `raw_events` DataFrame; if omitted, the
model falls back to using an "empirical xG" = P(goal | shot) computed from
SPADL alone. Both paths are documented so reviewers can see the choice.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# Same shot-xG regex used by the xG baseline (kept local to avoid coupling).
_XG_KEY_RE = re.compile(r"['\"]statsbomb_xg['\"]\s*:\s*([0-9.]+)")


def _xg_from_shot_str(shot_str: str) -> float:
    if not isinstance(shot_str, str):
        return 0.0
    m = _XG_KEY_RE.search(shot_str)
    return float(m.group(1)) if m else 0.0


class XTAction:
    """Reference specification: Sumpter Soccermatics `xTAction`.
    Corpus paper: singh_2018_xt (Sumpter's action variant of Singh's idea)."""

    id: str = "xt_action_sumpter"
    name: str = "xT-Action (Sumpter action-based P(shot)·xG)"
    corpus_paper_id: str = "singh_2018_xt"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
        "t2_state_value_predicts_possession_outcome",
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    def __init__(self, look_ahead: int = 10) -> None:
        self.look_ahead = look_ahead
        self._shot_type_ids: set[int] = set()
        self._move_type_ids: set[int] = set()
        self._success_result_id: int = -1
        self.shot_clf = None       # logistic P(shot|action features)
        self.xg_reg = None         # linear regressor xG conditional on shot

    # ---------------------------------------------------------------

    def _load_spadl_labels(self) -> None:
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

    # ---------------------------------------------------------------

    def _feature_matrix(self, actions: pd.DataFrame) -> np.ndarray:
        """Compact feature vector per action."""
        # Coordinates.
        cols = np.column_stack([
            actions["start_x"].astype(float).values,
            actions["start_y"].astype(float).values,
            actions["end_x"].astype(float).values,
            actions["end_y"].astype(float).values,
        ])
        # One-hot of a small action-type family: pass / dribble / cross / take_on.
        # Anything else contributes zero to the one-hot; the coord features
        # still carry the signal.
        types = actions["type_id"].values
        from socceraction.spadl import config as spadlcfg
        pass_id     = spadlcfg.actiontypes.index("pass")
        dribble_id  = spadlcfg.actiontypes.index("dribble")
        cross_id    = spadlcfg.actiontypes.index("cross")
        take_on_id  = spadlcfg.actiontypes.index("take_on")
        onehot = np.column_stack([
            (types == pass_id).astype(int),
            (types == dribble_id).astype(int),
            (types == cross_id).astype(int),
            (types == take_on_id).astype(int),
        ])
        return np.hstack([cols, onehot])

    def _shot_xg_lookup(
        self, raw_events: pd.DataFrame | None
    ) -> dict[tuple[int, str], float] | None:
        """Optional (game_id, event_uuid) → StatsBomb xG mapping.
        Returns None if raw_events not supplied."""
        if raw_events is None or len(raw_events) == 0:
            return None
        # statsbombpy uses 'id' for the event UUID; column may be lowercased.
        id_col = "id" if "id" in raw_events.columns else "event_id"
        game_col = "match_id" if "match_id" in raw_events.columns else "game_id"
        shots = raw_events[raw_events["type"].astype(str) == "Shot"].copy()
        if "shot_statsbomb_xg" in shots.columns:
            shots["xg"] = pd.to_numeric(shots["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
        elif "shot" in shots.columns:
            shots["xg"] = shots["shot"].astype(str).map(_xg_from_shot_str)
        else:
            shots["xg"] = 0.0
        return dict(zip(
            zip(shots[game_col].astype(int).values, shots[id_col].astype(str).values),
            shots["xg"].astype(float).values,
        ))

    def _build_labels(
        self,
        actions: pd.DataFrame,
        raw_events: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_shot, y_xg) per row of `actions`."""
        xg_lookup = self._shot_xg_lookup(raw_events)
        # Sort within each game by (period, time, action_id) so we can walk forwards.
        actions = actions.sort_values(
            ["game_id", "period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)

        y_shot = np.zeros(len(actions), dtype=int)
        y_xg = np.zeros(len(actions), dtype=float)

        types = actions["type_id"].values
        results = actions["result_id"].values
        teams = actions["team_id"].values
        games = actions["game_id"].astype(int).values
        uuids = actions.get("original_event_id",
                             pd.Series([""] * len(actions))).astype(str).values

        is_shot = np.isin(types, list(self._shot_type_ids))
        is_success = results == self._success_result_id

        for i in range(len(actions)):
            end = min(i + self.look_ahead + 1, len(actions))
            for j in range(i + 1, end):
                if games[j] != games[i]:
                    break                          # different match
                if not is_shot[j] or teams[j] != teams[i]:
                    continue
                # Same team took a shot within k actions.
                y_shot[i] = 1
                if xg_lookup is not None:
                    y_xg[i] = xg_lookup.get(
                        (int(games[j]), str(uuids[j])), 0.0
                    )
                else:
                    # Fallback: empirical goal-per-shot rate (approximation
                    # of xG when raw StatsBomb xG is unavailable).
                    y_xg[i] = 1.0 if is_success[j] else 0.0
                break
        return y_shot, y_xg

    # ---------------------------------------------------------------

    def fit(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
        raw_events: pd.DataFrame | None = None,
    ) -> None:
        from sklearn.linear_model import LinearRegression, LogisticRegression

        self._load_spadl_labels()

        if len(actions) == 0:
            self.shot_clf = None
            self.xg_reg = None
            return

        y_shot, y_xg = self._build_labels(actions, raw_events)
        X = self._feature_matrix(actions)

        # Classifier: P(shot | features). Balance is often very skewed, so
        # class_weight="balanced" helps.
        self.shot_clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear",
        ).fit(X, y_shot)

        # Regressor: xG | shot. Only train on rows that led to a shot,
        # since xG on non-shot rows is meaningless.
        shot_mask = y_shot.astype(bool)
        if shot_mask.sum() >= 5:
            self.xg_reg = LinearRegression().fit(X[shot_mask], y_xg[shot_mask])
        else:
            self.xg_reg = None                     # too few shots → fall back to mean

    # ---------------------------------------------------------------

    def _per_action_values(self, actions: pd.DataFrame) -> np.ndarray:
        X = self._feature_matrix(actions)
        if self.shot_clf is None:
            return np.zeros(len(actions))
        p_shot = self.shot_clf.predict_proba(X)[:, 1]
        if self.xg_reg is not None:
            xg_hat = np.clip(self.xg_reg.predict(X), 0.0, 1.0)
        else:
            xg_hat = np.full_like(p_shot, 0.1)     # bland fallback
        return p_shot * xg_hat

    def predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """T1 projection. Per-action P(shot) × predicted xG."""
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
        actions = actions.copy()
        actions["xt_action_value"] = self._per_action_values(actions)
        agg = (actions.groupby(["game_id", "team_id"], as_index=False)["xt_action_value"]
                      .sum()
                      .rename(columns={"xt_action_value": "value"}))

        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
