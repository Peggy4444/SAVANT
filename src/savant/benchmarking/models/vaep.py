"""VAEP — Decroos et al. (2019) via socceraction's canonical implementation.

VAEP frames per-action valuation as two binary classifiers:

    V(a) = ΔP(team_scores in next k) − ΔP(team_concedes in next k)

with `k` = 10 SPADL actions by default. We use socceraction's canonical
reference (ML-KULeuven maintains it) so our numbers are directly comparable
to the community's understanding of VAEP.

Pipeline
--------
1. `socceraction.spadl.add_names` — enrich SPADL with type_name / result_name
   / bodypart_name columns (labels + formula need them).
2. `socceraction.vaep.features.gamestates(actions, nb_prev_actions=3)` —
   the temporal context window Decroos et al. specify.
3. Concatenate 11 built-in feature functions → ~145-dim feature matrix.
4. `socceraction.vaep.labels.scores` / `concedes` per action.
5. Train two XGBoost binary classifiers (Decroos et al.'s original used
   CatBoost; XGBoost is socceraction's default and gives comparable Brier).
6. Predict P(scores) and P(concedes) per action → apply
   `socceraction.vaep.formula.value` → per-action VAEP values.
7. Aggregate to per-team-per-match for T3.
"""

from __future__ import annotations

import pandas as pd


class VAEP:
    """Reference specification: `socceraction` (ML-KULeuven).
    Corpus paper: decroos_2019_vaep."""

    id: str = "vaep_socceraction"
    name: str = "VAEP (Decroos 2019, socceraction reference)"
    corpus_paper_id: str = "decroos_2019_vaep"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
        "t2_state_value_predicts_possession_outcome",
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    # Default hyperparameters — match socceraction's own tutorial closely.
    def __init__(
        self,
        nb_prev_actions: int = 3,
        label_nr_actions: int = 10,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.nb_prev_actions = nb_prev_actions
        self.label_nr_actions = label_nr_actions
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.random_state = random_state
        self.scores_clf = None
        self.concedes_clf = None
        self._feature_columns: list[str] | None = None

    # ---------------------------------------------------------------
    # Feature engineering: per-game so `gamestates(nb_prev_actions=3)` has
    # a valid local window. Concatenate across games at the end.
    # ---------------------------------------------------------------

    def _features_for_actions(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Concatenate socceraction's 11 canonical VAEP feature blocks.
        Actions MUST have been enriched with `spadl.add_names` first."""
        from socceraction.vaep import features as fs

        game_states = fs.gamestates(actions, nb_prev_actions=self.nb_prev_actions)
        parts = [
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
        ]
        return pd.concat(parts, axis=1)

    def _features_and_labels_per_game(
        self, actions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loop over games. Return (X, y) concatenated across games."""
        import socceraction.spadl as spadl
        from socceraction.vaep import labels as lb

        X_parts: list[pd.DataFrame] = []
        y_parts: list[pd.DataFrame] = []

        for _, game_actions in actions.groupby("game_id", sort=False):
            game_actions = game_actions.sort_values(
                ["period_id", "time_seconds", "action_id"]
            ).reset_index(drop=True)
            named = spadl.add_names(game_actions)
            X_game = self._features_for_actions(named)
            # `lb.scores` / `lb.concedes` each return a 1-col DataFrame; concat
            # them and rename to a stable ("scores", "concedes") schema.
            y_scores = lb.scores(named, nr_actions=self.label_nr_actions)
            y_concedes = lb.concedes(named, nr_actions=self.label_nr_actions)
            y_game = pd.concat([y_scores, y_concedes], axis=1)
            y_game.columns = ["scores", "concedes"]
            X_parts.append(X_game.reset_index(drop=True))
            y_parts.append(y_game.reset_index(drop=True))

        X = pd.concat(X_parts, ignore_index=True) if X_parts else pd.DataFrame()
        y = pd.concat(y_parts, ignore_index=True) if y_parts else pd.DataFrame()
        return X, y

    # ---------------------------------------------------------------

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame | None = None) -> None:
        from xgboost import XGBClassifier

        X, y = self._features_and_labels_per_game(actions)
        if len(X) == 0:
            self.scores_clf = None
            self.concedes_clf = None
            return

        self._feature_columns = X.columns.tolist()

        # base_score=0.5 is a stability guard: when a training split contains
        # zero positives for `scores` or `concedes` (rare, but possible on
        # very small subsets like unit-test fixtures) XGBoost's auto-computed
        # base_score falls outside (0, 1) and the fit aborts. With real
        # StatsBomb training splits both classes are always present, so this
        # explicit prior has no effect on the final numbers.
        common = dict(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            base_score=0.5,
        )
        self.scores_clf = XGBClassifier(**common).fit(X.values, y["scores"].values)
        self.concedes_clf = XGBClassifier(**common).fit(X.values, y["concedes"].values)

    # ---------------------------------------------------------------
    # Prediction: same per-game loop; `value()` needs pandas Series
    # (numpy arrays trigger .shift() errors — verified in BM-6 exploration).
    # ---------------------------------------------------------------

    def _predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with cols [game_id, team_id, vaep_value] aligned to actions."""
        import socceraction.spadl as spadl
        from socceraction.vaep import formula as vaepf

        assert self.scores_clf is not None and self.concedes_clf is not None, "call fit() first"

        out_parts: list[pd.DataFrame] = []
        for _, game_actions in actions.groupby("game_id", sort=False):
            game_actions = game_actions.sort_values(
                ["period_id", "time_seconds", "action_id"]
            ).reset_index(drop=True)
            named = spadl.add_names(game_actions)
            X_game = self._features_for_actions(named)

            # Align features to the training column set if it drifted (rare —
            # can happen if a rare action-type only appears in one split).
            if self._feature_columns is not None:
                X_game = X_game.reindex(columns=self._feature_columns, fill_value=0)

            p_scores = pd.Series(
                self.scores_clf.predict_proba(X_game.values)[:, 1],
                name="scores",
            )
            p_concedes = pd.Series(
                self.concedes_clf.predict_proba(X_game.values)[:, 1],
                name="concedes",
            )
            values = vaepf.value(named, p_scores, p_concedes)
            values["game_id"] = named["game_id"].values
            values["team_id"] = named["team_id"].values
            out_parts.append(values[["game_id", "team_id", "vaep_value"]])

        return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(
            columns=["game_id", "team_id", "vaep_value"]
        )

    def predict_per_action(self, actions: pd.DataFrame) -> pd.DataFrame:
        """T1 projection. VAEP delta (offensive − defensive) per SPADL action.

        Rows are ordered to match the internal per-game sort key
        (period_id, time_seconds, action_id) — the caller can re-align
        via `action_id` if it needs the original input order.
        """
        import socceraction.spadl as spadl
        from socceraction.vaep import formula as vaepf

        assert self.scores_clf is not None and self.concedes_clf is not None, "call fit()"

        out_parts: list[pd.DataFrame] = []
        for _, ga in actions.groupby("game_id", sort=False):
            ga = ga.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True)
            named = spadl.add_names(ga)
            X = self._features_for_actions(named)
            if self._feature_columns is not None:
                X = X.reindex(columns=self._feature_columns, fill_value=0)
            p_scores   = pd.Series(self.scores_clf.predict_proba(X.values)[:, 1],   name="scores")
            p_concedes = pd.Series(self.concedes_clf.predict_proba(X.values)[:, 1], name="concedes")
            v = vaepf.value(named, p_scores, p_concedes)
            out_parts.append(pd.DataFrame({
                "game_id":   named["game_id"].values,
                "team_id":   named["team_id"].values,
                "action_id": named["action_id"].values,
                "value":     v["vaep_value"].values.astype(float),
            }))
        return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(
            columns=["game_id", "team_id", "action_id", "value"],
        )

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        per_action = self._predict_per_action(actions)

        agg = (per_action.groupby(["game_id", "team_id"], as_index=False)["vaep_value"]
                          .sum()
                          .rename(columns={"vaep_value": "value"}))
        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
