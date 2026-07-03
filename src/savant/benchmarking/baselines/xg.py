"""Raw xG-per-shot baseline.

For T3: team-match value = sum of StatsBomb's built-in xG for the team's shots.
This is the strongest of the three baselines and every method must beat it
on team-level match-xG correlation to be interesting.

StatsBomb events carry `shot.statsbomb_xg` (via statsbombpy). SPADL drops
this field, so this baseline consumes raw events rather than SPADL actions.
"""

from __future__ import annotations

import re

import pandas as pd


# statsbombpy flattens nested StatsBomb JSON into columns like
# "shot" with a dict-of-str. We pull xG out of that with a small regex helper.
_XG_KEY_RE = re.compile(r"['\"]statsbomb_xg['\"]\s*:\s*([0-9.]+)")


def _xg_from_shot_str(shot_str: str) -> float:
    """Extract the numeric xG from statsbombpy's stringified shot dict.

    Robust to both single- and double-quoted representations because
    statsbombpy's dtype-coercion path can produce either."""
    if not isinstance(shot_str, str):
        return 0.0
    m = _XG_KEY_RE.search(shot_str)
    return float(m.group(1)) if m else 0.0


class XGPerShotBaseline:
    """Team-match value = sum of built-in StatsBomb xG for that team's shots.

    Consumes RAW events (from `StatsBombOpenLoader`) rather than SPADL actions,
    because SPADL drops the xG field.
    """

    id: str = "xg_per_shot"
    name: str = "Raw xG per shot"
    reports_on_tasks: list[str] = [
        "t1_action_value_predicts_xg",
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    def fit(self, actions: pd.DataFrame, matches: pd.DataFrame) -> None:
        """No-op."""

    def predict_team_match_values(
        self,
        events: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return [game_id, team_id, value] where value = ∑ StatsBomb xG per shot.

        NB: `events` here is the RAW StatsBomb DataFrame from
        `StatsBombOpenLoader`, not SPADL actions. statsbombpy uses lowercase
        column names and either flattens `shot` or leaves it as a stringified
        dict; we handle both."""
        # Isolate shot events. statsbombpy uses lowercase `type` = "Shot".
        shots = events[events["type"].astype(str) == "Shot"].copy()

        # xG can live in either a flat column ("shot_statsbomb_xg") or embedded
        # inside a stringified `shot` dict — statsbombpy is inconsistent.
        if "shot_statsbomb_xg" in shots.columns:
            shots["xg"] = pd.to_numeric(shots["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
        elif "shot" in shots.columns:
            shots["xg"] = shots["shot"].astype(str).map(_xg_from_shot_str)
        else:
            shots["xg"] = 0.0

        # Prefer the integer `team_id` so downstream merges (T3 eval, leaderboard)
        # match the (int) team_id emitted by every SPADL-based method wrapper.
        # Fall back to the team name only if statsbombpy dropped team_id (rare).
        team_col = "team_id" if "team_id" in shots.columns else "team"
        game_col = "match_id" if "match_id" in shots.columns else "game_id"

        agg = (shots.groupby([game_col, team_col], as_index=False)["xg"]
                    .sum()
                    .rename(columns={game_col: "game_id",
                                     team_col: "team_id",
                                     "xg": "value"}))

        # Include every team that appears in the match, even zero-shot teams,
        # for the same reason as the goals baseline.
        all_pairs = events[[game_col, team_col]].drop_duplicates().rename(
            columns={game_col: "game_id", team_col: "team_id"}
        )
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
