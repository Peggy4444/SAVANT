"""Goals-per-90 baseline.

For T3 (team-level value explains match xG): the baseline emits, per (game,
team), the total goals that team scored in that match. This is the simplest
possible "value" signal — how well does raw goal-count-per-match predict
match xG differential? Every reviewer expects to see it as the naive floor.

The "per 90" naming is inherited from the football-analytics convention;
at the team-per-match level a single match is already 90 minutes (~=), so
the aggregate value IS the per-match count.
"""

from __future__ import annotations

import pandas as pd


class GoalsPer90Baseline:
    """Trivial baseline. No fitting. Team-match value = total goals scored."""

    id: str = "goals_per_90"
    name: str = "Goals per 90"
    reports_on_tasks: list[str] = [
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame) -> None:
        """No-op — trivial baseline."""

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return [game_id, team_id, value] where value = goals scored.

        Expects SPADL-shaped `actions` with columns:
            game_id, team_id, type_id, result_id
        A goal is a shot (type_id label = 'shot') with a successful result.
        We resolve the label indices from the SPADL config so this stays
        robust against socceraction version changes.
        """
        from socceraction.spadl import config as spadlcfg

        shot_type_ids = {
            i for i, name in enumerate(spadlcfg.actiontypes)
            if name in ("shot", "shot_freekick", "shot_penalty")
        }
        success_result_id = spadlcfg.results.index("success")

        goals = actions[
            actions["type_id"].isin(shot_type_ids)
            & (actions["result_id"] == success_result_id)
        ]

        # Every (game, team) that ever appears in `actions` should get a row
        # even if they scored zero — otherwise T3's Pearson ρ regression is
        # biased by systematically dropping zero-goal team-matches.
        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        counts = (goals.groupby(["game_id", "team_id"], as_index=False)
                       .size()
                       .rename(columns={"size": "value"}))
        out = all_pairs.merge(counts, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0).astype(int)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
