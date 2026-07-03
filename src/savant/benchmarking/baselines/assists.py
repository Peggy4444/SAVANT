"""Assists-per-90 baseline.

For T3: team-match value = total assists that team's players registered in
that match. An assist here is the LAST pass action before a successful shot
by the same team, per socceraction's SPADL semantics.
"""

from __future__ import annotations

import pandas as pd


class AssistsPer90Baseline:
    """Trivial baseline. Team-match value = number of assists."""

    id: str = "assists_per_90"
    name: str = "Assists per 90"
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
        """Return [game_id, team_id, value] where value = assist count."""
        from socceraction.spadl import config as spadlcfg

        shot_type_ids = {
            i for i, name in enumerate(spadlcfg.actiontypes)
            if name in ("shot", "shot_freekick", "shot_penalty")
        }
        pass_type_ids = {
            i for i, name in enumerate(spadlcfg.actiontypes)
            if name in ("pass", "cross", "throw_in", "freekick_crossed",
                        "freekick_short", "corner_crossed", "corner_short")
        }
        success_result_id = spadlcfg.results.index("success")

        # An assist: successful shot preceded by a successful pass from a
        # teammate. We iterate by game to keep the sequence local.
        assists = []
        for gid, g in actions.groupby("game_id"):
            g = g.sort_values(["period_id", "time_seconds", "action_id"]).reset_index()
            for i in range(1, len(g)):
                row = g.iloc[i]
                if row["type_id"] not in shot_type_ids:
                    continue
                if row["result_id"] != success_result_id:
                    continue
                prev = g.iloc[i - 1]
                if (prev["type_id"] in pass_type_ids
                        and prev["result_id"] == success_result_id
                        and prev["team_id"] == row["team_id"]):
                    assists.append({
                        "game_id": int(gid),
                        "team_id": int(prev["team_id"]),
                    })

        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        assists_df = (pd.DataFrame(assists)
                        .groupby(["game_id", "team_id"], as_index=False)
                        .size()
                        .rename(columns={"size": "value"})
                      ) if assists else pd.DataFrame(columns=["game_id", "team_id", "value"])

        out = all_pairs.merge(assists_df, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0).astype(int)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
