"""Liu et al. (2020) — Deep RL action-value function for soccer players.

Reimplemented from the paper spec (see extraction `liu_2020_deep_soccer`).

Simplified variant
------------------
The paper's TTDP-LSTM is a two-tower stacked LSTM (256-256 per tower,
embedding + 2 hidden 256-node layers on each side), trained with on-
policy TD(0) / Sarsa on 4.5M events with Adam. Reproducing that stack
inside a benchmark wrapper — with all the LSTM state-management, TD
target sampling, and Sarsa look-back — is prohibitive.

We keep the *load-bearing signal* of the paper: the labelling scheme.

    Liu 2020 target:  Q_team(s, a) = P(team scores the NEXT goal
                                      after state s, without a bounded
                                      k-action horizon)

vs.

    VAEP target:      Q(s, a) = P(team scores within the next 10 actions)

The unbounded next-goal horizon is what distinguishes Liu 2020's
signal from VAEP's, and is where the two rankings diverge in the
paper's Table 10.

So we wrap this by re-using socceraction's canonical VAEP feature
pipeline (145-column feature block from `vaep.features`) — same features
Decroos et al. give as sufficient for action valuation — but swap the
labels for our unbounded next-goal targets, and train two XGBoost
classifiers (matching VAEP's default). This gives a Q_scores / Q_concedes
pair whose ΔQ is directly the paper's action-value.

Deviations from paper spec
--------------------------
- No LSTM history — features already include the 3-action look-back
  window (nb_prev_actions=3) that VAEP uses. Full recurrence over
  variable-length episodes is out of scope.
- No fine-tuning phase (FT-GIM). We train one global model.
- No CART mimic tree for interpretability. Not needed for the T-tasks.

Value definition
----------------
For each SPADL action i in a game:

    V(a_i) = (Q_scores(s_i) − Q_scores(s_{i-1}))
             − (Q_concedes(s_i) − Q_concedes(s_{i-1}))

where s_i is state after action i. Sums per (game_id, team_id) → T3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Liu2020:
    """Reference specification: Liu et al. (2020), DMKD 34.
    Corpus paper: liu_2020_deep_soccer."""

    id: str = "liu_2020_ttdp"
    name: str = "Liu 2020 (next-goal Q, simplified GBM variant)"
    corpus_paper_id: str = "liu_2020_deep_soccer"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
        "t2_state_value_predicts_possession_outcome",
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    def __init__(
        self,
        nb_prev_actions: int = 3,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.nb_prev_actions = nb_prev_actions
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.random_state = random_state
        self.scores_clf: Any = None
        self.concedes_clf: Any = None
        self._feature_columns: list[str] | None = None
        self._shot_type_ids: set[int] = set()
        self._success_id: int = -1

    # ---------------------------------------------------------------

    def _load_spadl_labels(self) -> None:
        from socceraction.spadl import config as spadlcfg
        self._shot_type_ids = {
            spadlcfg.actiontypes.index(n)
            for n in ("shot", "shot_penalty", "shot_freekick")
        }
        self._success_id = spadlcfg.results.index("success")

    def _features_for_game(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Same 11-block concat as VAEP; `actions` must already have SPADL
        name columns (add via socceraction.spadl.add_names)."""
        from socceraction.vaep import features as fs
        game_states = fs.gamestates(actions, nb_prev_actions=self.nb_prev_actions)
        return pd.concat([
            fs.actiontype_onehot(game_states),
            fs.result_onehot(game_states),
            fs.bodypart_onehot(game_states),
            fs.startlocation(game_states),
            fs.endlocation(game_states),
            fs.movement(game_states),
            fs.startpolar(game_states),
            fs.endpolar(game_states),
            fs.team(game_states),
            fs.time(game_states),
            fs.time_delta(game_states),
        ], axis=1)

    def _next_goal_labels(self, actions: pd.DataFrame) -> pd.DataFrame:
        """For each action, does the actor's team (or opponent) score the
        NEXT goal in the game? Unbounded look-ahead — the Liu 2020 target."""
        n = len(actions)
        types    = actions["type_id"].values
        results  = actions["result_id"].values
        team_ids = actions["team_id"].values

        is_goal = np.isin(types, list(self._shot_type_ids)) & (results == self._success_id)

        # Walk backwards from the end; for each row, "next goal" = the
        # earliest goal at position ≥ i. O(n) instead of O(n²).
        next_scoring_team = np.full(n, -1, dtype=np.int64)
        latest_scorer = -1
        for i in range(n - 1, -1, -1):
            if is_goal[i]:
                latest_scorer = int(team_ids[i])
            next_scoring_team[i] = latest_scorer

        scores   = (next_scoring_team == team_ids) & (next_scoring_team != -1)
        concedes = (next_scoring_team != team_ids) & (next_scoring_team != -1)
        return pd.DataFrame({
            "scores":   scores.astype(int),
            "concedes": concedes.astype(int),
        })

    def _features_and_labels(
        self, actions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        import socceraction.spadl as spadl

        X_parts: list[pd.DataFrame] = []
        Y_parts: list[pd.DataFrame] = []
        for _, ga in actions.groupby("game_id", sort=False):
            ga = ga.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True)
            named = spadl.add_names(ga)
            X_parts.append(self._features_for_game(named).reset_index(drop=True))
            Y_parts.append(self._next_goal_labels(named).reset_index(drop=True))
        X = pd.concat(X_parts, ignore_index=True) if X_parts else pd.DataFrame()
        Y = pd.concat(Y_parts, ignore_index=True) if Y_parts else pd.DataFrame()
        return X, Y

    # ---------------------------------------------------------------

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame | None = None) -> None:
        from xgboost import XGBClassifier

        self._load_spadl_labels()
        X, Y = self._features_and_labels(actions)
        if len(X) == 0:
            return

        self._feature_columns = X.columns.tolist()
        common = dict(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            base_score=0.5,          # guard against single-class training slices
        )
        self.scores_clf   = XGBClassifier(**common).fit(X.values, Y["scores"].values)
        self.concedes_clf = XGBClassifier(**common).fit(X.values, Y["concedes"].values)

    # ---------------------------------------------------------------

    def _per_action_values(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Return [game_id, team_id, value] aligned to actions."""
        import socceraction.spadl as spadl

        assert self.scores_clf is not None and self.concedes_clf is not None, "call fit()"

        out_parts: list[pd.DataFrame] = []
        for _, ga in actions.groupby("game_id", sort=False):
            ga = ga.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True)
            named = spadl.add_names(ga)
            X = self._features_for_game(named)
            if self._feature_columns is not None:
                X = X.reindex(columns=self._feature_columns, fill_value=0)

            p_scores   = self.scores_clf.predict_proba(X.values)[:, 1]
            p_concedes = self.concedes_clf.predict_proba(X.values)[:, 1]

            # ΔQ over consecutive actions (first action of each game has no
            # predecessor → value 0). Same-team continuation is assumed within
            # the LSTM's own episode boundary; for a simplified GBM this is a
            # reasonable proxy for the paper's TD target.
            dq_scores   = np.diff(p_scores,   prepend=p_scores[0])
            dq_concedes = np.diff(p_concedes, prepend=p_concedes[0])
            values = dq_scores - dq_concedes

            df = pd.DataFrame({
                "game_id":   named["game_id"].values,
                "team_id":   named["team_id"].values,
                "action_id": named["action_id"].values,
                "value":     values.astype(float),
            })
            out_parts.append(df)

        return pd.concat(out_parts, ignore_index=True) if out_parts else \
            pd.DataFrame(columns=["game_id", "team_id", "action_id", "value"])

    def predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """T1 projection. Δ-Q per action under the paper's next-goal target."""
        return self._per_action_values(actions)

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        per_action = self._per_action_values(actions)
        agg = (per_action.groupby(["game_id", "team_id"], as_index=False)["value"]
                          .sum())
        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
