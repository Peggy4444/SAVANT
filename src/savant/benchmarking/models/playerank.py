"""PlayeRank — Pappalardo et al. (2019), TIST 10(5):59.

Reimplemented from the paper spec plus the authors' `mesosbrodleto/playerank`
GitHub reference (see extraction `pappalardo_2019_playerank`).

Framework recap
---------------
PlayeRank is a role-aware ranking framework built on top of a per-action
weighting model. Its load-bearing piece for a benchmark is Phase 1
("Learning phase"): train a Linear SVC to predict team match outcome
(win = 1 vs non-win = 0) from team-aggregated action-count features. The
learned feature weights `w` then define the per-action value: an action
of type / result-code `i` contributes `w_i` to its team-match rating.

The paper's Phases 2 and 3 (rating normalisation, EWMA aggregation, and
role-based ranking via k-means on centres of performance) are player-level
personalisation on top of the team-level LSVC. For SAVANT's T3 metric —
"per-team-per-match value" — we score directly with the LSVC decision
function on the team's feature vector for that match. This is equivalent
to Σ_i w_i · x_i, the paper's own definition of the per-match rating
before normalisation.

Deviations from the paper spec
------------------------------
- Paper uses ~76 hand-defined Wyscout event / subtype / tag combinations
  (Table 7). SPADL doesn't carry Wyscout tags, so we use
  (type_id, result_id) pairs from SPADL as the feature key — 23 action
  types × 6 result codes = up to 138 columns. This is a coarser
  vocabulary but captures the same distinctions PlayeRank cares about
  (e.g. accurate vs inaccurate pass ↔ pass × success/fail).
- Paper's supervision is win vs non-win; we do the same, computed from
  successful-shot counts per team in the training split.
- Roles + EWMA are out of scope for a team-match benchmark score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PlayeRank:
    """Reference specification: Pappalardo et al. (2019).
    Corpus paper: pappalardo_2019_playerank."""

    id: str = "playerank_pappalardo"
    name: str = "PlayeRank (Pappalardo 2019, LSVC-weighted event counts)"
    corpus_paper_id: str = "pappalardo_2019_playerank"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    def __init__(self, C: float = 1.0, random_state: int = 0) -> None:
        self.C = C
        self.random_state = random_state
        self.svc: Any = None
        self._feature_columns: list[int] | None = None
        self._shot_ids: set[int] = set()
        self._success_id: int = -1

    # ---------------------------------------------------------------

    def _load_spadl_labels(self) -> None:
        from socceraction.spadl import config as spadlcfg
        self._shot_ids = {
            spadlcfg.actiontypes.index(n)
            for n in ("shot", "shot_penalty", "shot_freekick")
        }
        self._success_id = spadlcfg.results.index("success")

    @staticmethod
    def _feature_key(type_id: pd.Series, result_id: pd.Series) -> pd.Series:
        # Compact int encoding: leaves room for up to 100 result codes,
        # which is far more than SPADL will ever have.
        return type_id.astype(int) * 100 + result_id.astype(int)

    def _feature_vectors(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Return (game_id, team_id) × feature-count matrix."""
        df = actions[["game_id", "team_id", "type_id", "result_id"]].copy()
        df["feat"] = self._feature_key(df["type_id"], df["result_id"])
        pivot = df.pivot_table(
            index=["game_id", "team_id"],
            columns="feat",
            values="type_id",
            aggfunc="count",
            fill_value=0,
        )
        return pivot

    def _match_outcomes(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Return (game_id, team_id, win) with win ∈ {0, 1}. Draws are
        non-wins (matches paper's binary supervision)."""
        goal_mask = (actions["type_id"].isin(list(self._shot_ids))
                     & (actions["result_id"] == self._success_id))
        goals = (actions[goal_mask].groupby(["game_id", "team_id"])
                     .size().reset_index(name="goals_for"))
        pairs = actions[["game_id", "team_id"]].drop_duplicates()
        goals = pairs.merge(goals, on=["game_id", "team_id"], how="left").fillna({"goals_for": 0})

        # goals_against = total goals in match − goals_for
        totals = goals.groupby("game_id")["goals_for"].transform("sum")
        goals["goals_against"] = totals - goals["goals_for"]
        goals["win"] = (goals["goals_for"] > goals["goals_against"]).astype(int)
        return goals[["game_id", "team_id", "win"]]

    # ---------------------------------------------------------------

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame | None = None) -> None:
        from sklearn.svm import LinearSVC

        self._load_spadl_labels()
        X = self._feature_vectors(actions)
        y = self._match_outcomes(actions).set_index(["game_id", "team_id"])["win"]

        # Align on the multi-index (game_id, team_id) so rows correspond.
        X_aligned, y_aligned = X.align(y, join="inner", axis=0)
        self._feature_columns = X_aligned.columns.tolist()

        # Degenerate training splits (a single class) — the paper's LSVC
        # would refuse to fit. Store None so predict() falls back to zeros.
        if len(np.unique(y_aligned.values)) < 2:
            self.svc = None
            return

        self.svc = LinearSVC(
            C=self.C,
            random_state=self.random_state,
            dual="auto",
            max_iter=5000,
        ).fit(X_aligned.values, y_aligned.values)

    def predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """T1 projection. Per-action value = the LSVC weight of the
        (type_id, result_id) feature that action belongs to.
        Rationale: per the paper, r(u,m) = (1/R) Σ w_i·x_i — the per-action
        contribution is exactly the weight of that action's feature key."""
        # No trained classifier → NaN everywhere.
        if self.svc is None or self._feature_columns is None:
            return pd.DataFrame({
                "game_id":   actions["game_id"].values,
                "team_id":   actions["team_id"].values,
                "action_id": actions["action_id"].values,
                "value":     np.nan,
            })

        # Build feat → weight map from the LSVC's coefs.
        w = self.svc.coef_.ravel()
        weight_by_feat = dict(zip(self._feature_columns, w))
        feat_key = (actions["type_id"].astype(int) * 100
                    + actions["result_id"].astype(int)).values
        values = np.array([weight_by_feat.get(int(f), np.nan) for f in feat_key])
        return pd.DataFrame({
            "game_id":   actions["game_id"].values,
            "team_id":   actions["team_id"].values,
            "action_id": actions["action_id"].values,
            "value":     values.astype(float),
        })

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.svc is None or self._feature_columns is None:
            all_pairs = actions[["game_id", "team_id"]].drop_duplicates().copy()
            all_pairs["value"] = 0.0
            return all_pairs.sort_values(["game_id", "team_id"]).reset_index(drop=True)

        # Rebuild the same key encoding used at training and reindex so
        # unseen feature columns don't propagate NaN into decision_function.
        self._load_spadl_labels()  # in case predict is called without fit again
        X = (self._feature_vectors(actions)
                 .reindex(columns=self._feature_columns, fill_value=0))
        raw = self.svc.decision_function(X.values)
        out = X.reset_index()[["game_id", "team_id"]].copy()
        out["value"] = raw.astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
