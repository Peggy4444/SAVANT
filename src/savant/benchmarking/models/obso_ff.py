"""OBSO_FF — Off-Ball Scoring Opportunity, freeze-frame approximation.

Reimplemented from William Spearman (2018), "Beyond Expected Goals"
(MIT Sloan). Corpus paper: `spearman_2018_beyondxg`.

Deviations from paper spec
--------------------------
The paper uses 25 Hz Hudl tracking data with per-player positions AND
velocities. StatsBomb 360 gives us only sparse freeze-frames at
on-ball events (visible players' positions, no velocities, no continuous
stream). This wrapper therefore computes OBSO **only at the events for
which 360 has a freeze-frame**, and treats every visible player as having
zero velocity. Named `OBSO_FF` (freeze-frame) so downstream code doesn't
confuse it with the tracking-native OBSO.

Concretely we compute, for each freeze-frame event:

    OBSO(D) ≈ Σ_r  P_ctrl(r|D) · P_trans(r|D) · P_goal(r|D) · dA

with

    P_ctrl(r) = σ((t_D(r) − t_A(r)) / s)      logistic pitch control
    P_trans(r) ∝ N(r; ball, σ=23.9 m)          2D normal around ball, then
                                                normalised to integrate to 1
    P_goal(r) = exp(−d_goal(r) / 10 m)          monotonic score-model proxy

t_A(r) / t_D(r) are the minimum arrival times of attackers / defenders
from their frozen positions, using the paper's constant-acceleration
kinematics (a = 7 m/s², v_max = 5 m/s, zero initial velocity).

We use the paper's MAP hyperparameters (Table 1) directly — the wrapper
does NOT re-fit them from data. The paper itself learned them from only
5 training matches with strong priors.

Score model deviation
---------------------
The paper's score model is `f(d_goal)^β` with β = 0.48. We use an
exponential in d_goal instead of a power law. Both are monotonic and
peak at the goal; both give the same qualitative OBSO rankings. Full
power-law replication would require fitting a d_ref scale from event
data, which for a benchmark-slot wrapper is out of scope.

Coordinate system
-----------------
Freeze-frame `location` columns are StatsBomb pixel coordinates
(0–120 × 0–80). We convert to metres (105 × 68) so the paper's σ and
a/v_max are dimensionally consistent. The attacking goal is at
(105 m, 34 m) — StatsBomb events are already reoriented so the actor's
team attacks left → right.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paper MAP values (§ Table 1) — no re-fit in this wrapper.
_TIME_UNCERTAINTY_S = 0.54
_SIGMA_TRANS_M       = 23.9
_ACCEL_M_PER_S2      = 7.0
_MAX_SPEED_M_PER_S   = 5.0

# Pitch dimensions.
_SB_PITCH_X = 120.0
_SB_PITCH_Y = 80.0
_PITCH_LENGTH_M = 105.0
_PITCH_WIDTH_M  = 68.0

# Attacking goal (all StatsBomb events are actor-oriented l→r).
_GOAL_X_M = _PITCH_LENGTH_M
_GOAL_Y_M = _PITCH_WIDTH_M / 2

_SCORE_DECAY_M = 10.0    # exponential score model — see docstring


class OBSO_FF:
    """Reference specification: Spearman (2018), MIT Sloan.
    Corpus paper: spearman_2018_beyondxg."""

    id: str = "obso_ff_spearman"
    name: str = "OBSO_FF (Spearman 2018, freeze-frame approximation)"
    corpus_paper_id: str = "spearman_2018_beyondxg"
    reports_on_tasks: list[str] = [
        "t3_team_value_explains_match_xg",
        "t4_cross_competition_generalisation",
    ]

    # Required data source — the orchestrator uses this to decide whether
    # to pass 360 freeze frames to fit/predict.
    requires_360: bool = True

    def __init__(self, grid_x: int = 21, grid_y: int = 14) -> None:
        self.grid_x = grid_x    # ~5 m cells in x
        self.grid_y = grid_y    # ~5 m cells in y

    # ---------------------------------------------------------------

    def fit(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
        frames: pd.DataFrame | None = None,
    ) -> None:
        """No fitting — OBSO_FF is physics-based with paper-MAP parameters."""
        return

    # ---------------------------------------------------------------

    def _time_to_intercept(
        self,
        player_x: np.ndarray,   # (n_players,) in metres
        player_y: np.ndarray,
        target_x: np.ndarray,   # (n_grid,)   in metres
        target_y: np.ndarray,
    ) -> np.ndarray:
        """Zero-velocity constant-acceleration TTI, (n_players, n_grid)."""
        dx = target_x[None, :] - player_x[:, None]
        dy = target_y[None, :] - player_y[:, None]
        d = np.sqrt(dx * dx + dy * dy)          # (n_players, n_grid)
        d0 = _MAX_SPEED_M_PER_S ** 2 / (2 * _ACCEL_M_PER_S2)
        t_ramp = _MAX_SPEED_M_PER_S / _ACCEL_M_PER_S2
        t = np.where(
            d <= d0,
            np.sqrt(2 * np.clip(d, 0, None) / _ACCEL_M_PER_S2),
            t_ramp + (d - d0) / _MAX_SPEED_M_PER_S,
        )
        return t

    def _obso_at_frame(
        self,
        attackers_xy: np.ndarray,   # (n_attackers, 2) in metres
        defenders_xy: np.ndarray,
        ball_xy: np.ndarray,        # (2,) in metres
    ) -> float:
        """Integrated OBSO scalar for one freeze frame."""
        # No attackers or defenders → don't credit anything.
        if len(attackers_xy) == 0 or len(defenders_xy) == 0:
            return 0.0

        # Coarse grid over the pitch.
        gx = np.linspace(0, _PITCH_LENGTH_M, self.grid_x)
        gy = np.linspace(0, _PITCH_WIDTH_M,  self.grid_y)
        XX, YY = np.meshgrid(gx, gy, indexing="xy")   # (grid_y, grid_x)
        target_x = XX.ravel()
        target_y = YY.ravel()
        cell_area = (_PITCH_LENGTH_M / self.grid_x) * (_PITCH_WIDTH_M / self.grid_y)

        # Time-to-intercept per team.
        t_A = self._time_to_intercept(
            attackers_xy[:, 0], attackers_xy[:, 1], target_x, target_y,
        ).min(axis=0)
        t_D = self._time_to_intercept(
            defenders_xy[:, 0], defenders_xy[:, 1], target_x, target_y,
        ).min(axis=0)

        # Pitch control: attacker wins where t_A < t_D.
        P_ctrl = 1.0 / (1.0 + np.exp((t_A - t_D) / _TIME_UNCERTAINTY_S))

        # 2D transition density around the ball (unnormalised, then normalise
        # over the grid so ∫ P_trans dA ≈ 1).
        d_ball_sq = (target_x - ball_xy[0]) ** 2 + (target_y - ball_xy[1]) ** 2
        P_trans = np.exp(-d_ball_sq / (2 * _SIGMA_TRANS_M ** 2))
        Z = P_trans.sum() * cell_area
        if Z > 0:
            P_trans = P_trans / Z

        # Score model: monotonic in distance to attacker goal.
        d_goal = np.sqrt((target_x - _GOAL_X_M) ** 2 + (target_y - _GOAL_Y_M) ** 2)
        P_goal = np.exp(-d_goal / _SCORE_DECAY_M)

        return float((P_ctrl * P_trans * P_goal).sum() * cell_area)

    # ---------------------------------------------------------------

    def _iterate_frame_groups(self, frames: pd.DataFrame):
        """Yield (event_id, match_id, actor_row, attackers_xy_m,
        defenders_xy_m, ball_xy_m) for each freeze-frame event."""
        # Orchestrator renames match_id → game_id; upstream tests may still
        # pass match_id directly. Accept either.
        match_col = "game_id" if "game_id" in frames.columns else "match_id"
        for (event_id, match_id), grp in frames.groupby(["id", match_col]):
            actor_rows = grp[grp["actor"]]
            if len(actor_rows) == 0:
                continue    # no actor tag → can't identify the acting team
            actor_row = actor_rows.iloc[0]
            # Location = [x_sb, y_sb]. Frames come out with a stringified list
            # if we round-tripped them through parquet; handle both.
            def _to_xy(loc: Any) -> np.ndarray:
                if isinstance(loc, str):
                    return np.array(eval(loc), dtype=float)   # cached repr
                return np.array(loc, dtype=float)

            def _to_m(xy_sb: np.ndarray) -> np.ndarray:
                return np.array([
                    xy_sb[0] * _PITCH_LENGTH_M / _SB_PITCH_X,
                    xy_sb[1] * _PITCH_WIDTH_M  / _SB_PITCH_Y,
                ])

            ball_xy = _to_m(_to_xy(actor_row["location"]))

            attackers = grp[grp["teammate"] & ~grp["keeper"]]
            defenders = grp[~grp["teammate"]]
            att_xy = np.array([_to_m(_to_xy(l)) for l in attackers["location"]]) \
                     if len(attackers) else np.zeros((0, 2))
            def_xy = np.array([_to_m(_to_xy(l)) for l in defenders["location"]]) \
                     if len(defenders) else np.zeros((0, 2))
            yield event_id, match_id, actor_row, att_xy, def_xy, ball_xy

    # ---------------------------------------------------------------

    def predict_per_action(
        self,
        actions: pd.DataFrame,
        frames: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """T1 projection. OBSO for actions with a matching 360 freeze-frame;
        NaN elsewhere. Note the `frames` kwarg — the orchestrator passes 360
        data explicitly."""
        out = pd.DataFrame({
            "game_id":   actions["game_id"].values,
            "team_id":   actions["team_id"].values,
            "action_id": actions["action_id"].values,
            "original_event_id": actions.get(
                "original_event_id", pd.Series([""] * len(actions))
            ).astype(str).values,
            "value":     np.nan,
        })
        if frames is None or len(frames) == 0:
            return out[["game_id", "team_id", "action_id", "value"]]

        # Compute OBSO per event id.
        per_frame_rows: list[dict[str, object]] = []
        for event_id, _, _, att_xy, def_xy, ball_xy in \
                self._iterate_frame_groups(frames):
            per_frame_rows.append({
                "original_event_id": str(event_id),
                "value":             self._obso_at_frame(att_xy, def_xy, ball_xy),
            })
        if not per_frame_rows:
            return out[["game_id", "team_id", "action_id", "value"]]
        obso_by_event = pd.DataFrame(per_frame_rows)

        out = out.drop(columns=["value"]).merge(
            obso_by_event, on="original_event_id", how="left",
        )
        return out[["game_id", "team_id", "action_id", "value"]]

    def predict_team_match_values(
        self,
        actions: pd.DataFrame,
        matches: pd.DataFrame | None = None,
        frames: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Aggregate per-frame OBSO by (game_id, team_id).

        Frames without an actor tag are silently skipped. Actions in
        matches with no 360 data get 0-value rows so the output always
        covers the same (game, team) pairs as the SPADL input.
        """
        if frames is None or len(frames) == 0:
            logger.warning("OBSO_FF: no 360 frames provided — returning zeros.")
            all_pairs = actions[["game_id", "team_id"]].drop_duplicates().copy()
            all_pairs["value"] = 0.0
            return all_pairs.sort_values(["game_id", "team_id"]).reset_index(drop=True)

        # Map event_id → (game_id, team_id) via SPADL's original_event_id.
        # 360 keys are UUIDs; SPADL keeps them intact.
        event_index = (actions[["original_event_id", "game_id", "team_id"]]
                       .dropna(subset=["original_event_id"])
                       .drop_duplicates(subset=["original_event_id"])
                       .set_index("original_event_id"))

        rows: list[dict[str, Any]] = []
        for event_id, match_id, actor_row, att_xy, def_xy, ball_xy in \
                self._iterate_frame_groups(frames):
            if event_id not in event_index.index:
                continue    # 360 event without a SPADL sibling — skip
            gid = int(event_index.loc[event_id, "game_id"])
            tid = int(event_index.loc[event_id, "team_id"])
            obso = self._obso_at_frame(att_xy, def_xy, ball_xy)
            rows.append({"game_id": gid, "team_id": tid, "value": obso})

        if not rows:
            all_pairs = actions[["game_id", "team_id"]].drop_duplicates().copy()
            all_pairs["value"] = 0.0
            return all_pairs.sort_values(["game_id", "team_id"]).reset_index(drop=True)

        per_frame = pd.DataFrame(rows)
        agg = (per_frame.groupby(["game_id", "team_id"], as_index=False)["value"]
                          .sum())
        all_pairs = actions[["game_id", "team_id"]].drop_duplicates()
        out = all_pairs.merge(agg, on=["game_id", "team_id"], how="left")
        out["value"] = out["value"].fillna(0.0).astype(float)
        return out.sort_values(["game_id", "team_id"]).reset_index(drop=True)
