"""Ground-truth targets for Track-B benchmark tasks.

Currently ships targets for T3 (team-match xG differential). T1 and T2
targets are per-action / per-state — they require the per-method wrappers
to expose per-action predictions, which they don't yet. See TODOs at the
bottom.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Robust to statsbombpy's stringified-dict quirks (single- vs double-quoted).
_XG_KEY_RE = re.compile(r"['\"]statsbomb_xg['\"]\s*:\s*([0-9.]+)")


def _xg_from_shot_str(shot_str: object) -> float:
    if not isinstance(shot_str, str):
        return 0.0
    m = _XG_KEY_RE.search(shot_str)
    return float(m.group(1)) if m else 0.0


def team_match_xg(events: pd.DataFrame) -> pd.DataFrame:
    """Sum StatsBomb-native xG per (game_id, team_id) from raw events.

    Handles statsbombpy's inconsistent schema: xG can live either in a
    flat `shot_statsbomb_xg` column or embedded inside a stringified
    `shot` dict. Uses `team_id` (int) as the canonical join key.
    """
    shots = events[events["type"].astype(str) == "Shot"].copy()

    if "shot_statsbomb_xg" in shots.columns:
        shots["xg"] = pd.to_numeric(shots["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
    elif "shot" in shots.columns:
        shots["xg"] = shots["shot"].astype(str).map(_xg_from_shot_str)
    else:
        shots["xg"] = 0.0

    game_col = "match_id" if "match_id" in shots.columns else "game_id"
    if "team_id" not in shots.columns:
        raise KeyError(
            "team_match_xg requires int team_id in events — got "
            f"columns {list(events.columns)[:20]}...",
        )

    agg = (shots.groupby([game_col, "team_id"], as_index=False)["xg"]
                .sum()
                .rename(columns={game_col: "game_id", "xg": "team_xg"}))
    # Include zero-shot teams so downstream merges never lose (game, team) pairs.
    all_pairs = events[[game_col, "team_id"]].drop_duplicates().rename(
        columns={game_col: "game_id"},
    )
    out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
    out["team_xg"] = out["team_xg"].fillna(0.0).astype(float)
    return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)


def team_match_xg_diff(events: pd.DataFrame) -> pd.DataFrame:
    """T3 ground truth: xG differential = team_xg − opponent_xg per (game, team).

    Positive values → team out-shot its opponent by xG.
    """
    tx = team_match_xg(events)
    total = tx.groupby("game_id")["team_xg"].transform("sum")
    tx["xg_diff"] = tx["team_xg"] - (total - tx["team_xg"])
    return tx[["game_id", "team_id", "team_xg", "xg_diff"]]


# --- T1 target: goal in next k=10 same-team actions --------------


def t1_goal_next_k_actions(
    actions: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Return per-action binary label: `Y[i] = 1` iff the same team scores
    in one of the next k=10 SPADL actions in the same game.

    Column set matches per-action prediction parquets so the two merge
    cleanly on (game_id, action_id):

        game_id, team_id, action_id, y_goal_next_k
    """
    from socceraction.spadl import config as spadlcfg
    shot_ids = {spadlcfg.actiontypes.index(n)
                for n in ("shot", "shot_penalty", "shot_freekick")}
    success_id = spadlcfg.results.index("success")

    label_parts: list[pd.DataFrame] = []
    for _, ga in actions.groupby("game_id", sort=False):
        ga = ga.sort_values(
            ["period_id", "time_seconds", "action_id"]
        ).reset_index(drop=True)
        types    = ga["type_id"].values
        results  = ga["result_id"].values
        team_ids = ga["team_id"].values
        n = len(ga)

        is_goal = (
            pd.Series(types).isin(list(shot_ids)).values
            & (results == success_id)
        )

        y = pd.Series(0, index=range(n), dtype=int)
        for i in range(n):
            hi = min(i + k + 1, n)
            for j in range(i + 1, hi):
                if is_goal[j] and team_ids[j] == team_ids[i]:
                    y.iloc[i] = 1
                    break
        label_parts.append(pd.DataFrame({
            "game_id":       ga["game_id"].values,
            "team_id":       ga["team_id"].values,
            "action_id":     ga["action_id"].values,
            "y_goal_next_k": y.values,
        }))
    return pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame(
        columns=["game_id", "team_id", "action_id", "y_goal_next_k"],
    )


# --- T2 target: possession outcome -----------------------------------


# Ordinal encoding used across evaluation + calibration heads.
T2_OUTCOME_CLASSES: tuple[str, ...] = ("goal", "no_goal", "turnover", "foul")
T2_OUTCOME_INDEX: dict[str, int] = {c: i for i, c in enumerate(T2_OUTCOME_CLASSES)}


def t2_possession_outcome(actions: pd.DataFrame) -> pd.DataFrame:
    """Return per-action possession-terminal outcome ∈ T2_OUTCOME_CLASSES.

    Possession segmentation rule (event-data proxy — no player tracking
    is required):

        Iterate actions in temporal order per game. Track the current
        possession's team. A new action starts a new possession if:

          - the acting team differs from the previous possession's team
            (→ previous ends as `turnover`), OR
          - the previous action was terminal (shot / foul / period end).

        A possession terminates with:
            goal      — successful shot / shot_penalty / shot_freekick
            no_goal   — unsuccessful shot
            foul      — foul action
            turnover  — team change without an intervening shot/foul,
                        OR period end / end-of-game

    The returned outcome is the terminal outcome of the possession the
    action belongs to — same label for every action in a possession.

    Columns: game_id, team_id, action_id, possession_outcome (str),
             y_possession_outcome (int, per T2_OUTCOME_INDEX).
    """
    from socceraction.spadl import config as spadlcfg
    shot_ids = {spadlcfg.actiontypes.index(n)
                for n in ("shot", "shot_penalty", "shot_freekick")}
    foul_id  = spadlcfg.actiontypes.index("foul")
    success_id = spadlcfg.results.index("success")

    parts: list[pd.DataFrame] = []
    for _, ga in actions.groupby("game_id", sort=False):
        ga = ga.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True)
        n = len(ga)
        types   = ga["type_id"].values
        results = ga["result_id"].values
        teams   = ga["team_id"].values
        periods = ga["period_id"].values

        outcomes = ["turnover"] * n         # default; will be overwritten by pass 2
        # Pass 1: assign a possession_id per action.
        poss_id = np.zeros(n, dtype=int)
        current_id = 0
        for i in range(n):
            if i == 0:
                poss_id[i] = 0
                continue
            new_poss = False
            # Period change → new possession.
            if periods[i] != periods[i - 1]:
                new_poss = True
            # Team change → new possession (previous ended as turnover).
            elif teams[i] != teams[i - 1]:
                new_poss = True
            # Previous action was terminal → new possession.
            elif types[i - 1] in shot_ids or types[i - 1] == foul_id:
                new_poss = True

            if new_poss:
                current_id += 1
            poss_id[i] = current_id

        # Pass 2: derive terminal outcome per possession from its LAST action.
        # (Iterate one possession at a time — cheap; possessions rarely > 20 actions.)
        for pid in np.unique(poss_id):
            rows = np.where(poss_id == pid)[0]
            last = rows[-1]
            last_type = types[last]
            last_result = results[last]
            if last_type in shot_ids:
                outcome = "goal" if last_result == success_id else "no_goal"
            elif last_type == foul_id:
                outcome = "foul"
            else:
                outcome = "turnover"
            for r in rows:
                outcomes[r] = outcome

        parts.append(pd.DataFrame({
            "game_id":                ga["game_id"].values,
            "team_id":                ga["team_id"].values,
            "action_id":              ga["action_id"].values,
            "possession_outcome":     outcomes,
            "y_possession_outcome":   [T2_OUTCOME_INDEX[o] for o in outcomes],
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["game_id", "team_id", "action_id",
                 "possession_outcome", "y_possession_outcome"],
    )


