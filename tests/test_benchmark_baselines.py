"""Baselines unit tests (BM-4). Uses synthetic SPADL / event DataFrames so
this test file runs fully offline. A slow end-to-end CLI smoke test lives
below, guarded by @pytest.mark.slow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from savant.benchmarking.baselines import (
    AssistsPer90Baseline, BASELINES_BY_ID, GoalsPer90Baseline,
    XGPerShotBaseline,
)

REPO = Path(__file__).resolve().parents[1]


# --- Fixtures --------------------------------------------------------

@pytest.fixture
def spadl_actions() -> pd.DataFrame:
    """Two games × two teams. Team A scores 2 goals in game 1 (one assisted),
    Team B scores 1 in game 2 (assisted). Team A also has one missed shot."""
    from socceraction.spadl import config as spadlcfg
    shot_id = spadlcfg.actiontypes.index("shot")
    pass_id = spadlcfg.actiontypes.index("pass")
    success = spadlcfg.results.index("success")
    fail = spadlcfg.results.index("fail")

    rows = [
        # --- game 1: Team A (team_id=1) scores twice, once assisted ---
        # (period, time, team, type, result)
        (1, 10.0, 1, pass_id, success),   # pass by team A
        (1, 11.0, 1, shot_id, success),   # goal by A (assisted)
        (1, 200.0, 1, shot_id, fail),      # missed shot
        (1, 300.0, 1, shot_id, success),   # unassisted goal
        (1, 400.0, 2, pass_id, success),   # random team B pass
        # --- game 2: Team B (team_id=2) scores once, assisted ---
        (1, 50.0, 2, pass_id, success),    # pass by B
        (1, 51.0, 2, shot_id, success),    # goal by B (assisted)
    ]
    df = pd.DataFrame(rows, columns=[
        "period_id", "time_seconds", "team_id", "type_id", "result_id",
    ])
    df["game_id"] = [10, 10, 10, 10, 10, 20, 20]
    df["action_id"] = range(len(df))
    df["player_id"] = 0
    return df


@pytest.fixture
def raw_events() -> pd.DataFrame:
    """Minimum raw-event DataFrame the xG baseline consumes."""
    return pd.DataFrame([
        # game 10, team A: two shots with xG 0.3 + 0.1
        {"match_id": 10, "team": "A", "type": "Shot", "shot_statsbomb_xg": 0.3},
        {"match_id": 10, "team": "A", "type": "Shot", "shot_statsbomb_xg": 0.1},
        # game 10, team B: 1 shot xG 0.05
        {"match_id": 10, "team": "B", "type": "Shot", "shot_statsbomb_xg": 0.05},
        # non-shot events that must be filtered out
        {"match_id": 10, "team": "A", "type": "Pass", "shot_statsbomb_xg": None},
        # game 20 has one shot for team B xG 0.5
        {"match_id": 20, "team": "B", "type": "Shot", "shot_statsbomb_xg": 0.5},
    ])


@pytest.fixture
def matches_meta() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [10, 20]})


# --- Goals baseline --------------------------------------------------

def test_goals_baseline_counts_goals_per_team_match(
    spadl_actions, matches_meta,
):
    preds = GoalsPer90Baseline().predict_team_match_values(spadl_actions, matches_meta)
    d = dict(zip(zip(preds["game_id"], preds["team_id"]), preds["value"]))
    assert d[(10, 1)] == 2      # Team A scored 2 in game 10
    assert d[(20, 2)] == 1      # Team B scored 1 in game 20


def test_goals_baseline_emits_row_for_zero_goal_teams(
    spadl_actions, matches_meta,
):
    """Every (game, team) that appears anywhere in the events must have a row
    with `value=0` if they didn't score — otherwise T3's Pearson ρ regression
    silently drops team-matches, biasing the correlation upward."""
    preds = GoalsPer90Baseline().predict_team_match_values(spadl_actions, matches_meta)
    zero_rows = preds[preds["value"] == 0]
    assert len(zero_rows) >= 1, (
        "expected at least one zero-goal team-match row (Team B in game 10)"
    )


# --- Assists baseline -----------------------------------------------

def test_assists_baseline_counts_last_pass_before_goal(
    spadl_actions, matches_meta,
):
    preds = AssistsPer90Baseline().predict_team_match_values(spadl_actions, matches_meta)
    d = dict(zip(zip(preds["game_id"], preds["team_id"]), preds["value"]))
    assert d[(10, 1)] == 1      # First goal was preceded by team A's pass
    assert d[(20, 2)] == 1      # Team B's goal was preceded by team B's pass


# --- xG baseline ----------------------------------------------------

def test_xg_baseline_sums_statsbomb_xg_per_team_match(raw_events, matches_meta):
    preds = XGPerShotBaseline().predict_team_match_values(raw_events, matches_meta)
    d = dict(zip(zip(preds["game_id"], preds["team_id"]), preds["value"]))
    assert d[(10, "A")] == pytest.approx(0.4, abs=1e-6)      # 0.3 + 0.1
    assert d[(10, "B")] == pytest.approx(0.05, abs=1e-6)
    assert d[(20, "B")] == pytest.approx(0.5, abs=1e-6)


# --- Registry -------------------------------------------------------

def test_baselines_registry_has_the_three_configured_ids():
    assert set(BASELINES_BY_ID) == {"goals_per_90", "assists_per_90", "xg_per_shot"}
    for cls in BASELINES_BY_ID.values():
        assert cls().id in {"goals_per_90", "assists_per_90", "xg_per_shot"}


# --- Slow: end-to-end CLI ------------------------------------------

@pytest.mark.slow
def test_run_benchmark_cli_end_to_end(tmp_path: Path):
    """Run scripts/run_benchmark.py with a tiny --max-matches on real data
    and verify the manifest + artefacts land where expected."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_benchmark.py"),
         "--max-matches", "3",
         "--out-dir", str(tmp_path)],
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"CLI failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.exists()
    import json
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_matches_total"] >= 1
    assert set(manifest["baselines"]) == {"goals_per_90", "assists_per_90", "xg_per_shot"}
    for bid in manifest["baselines"]:
        assert (tmp_path / "baselines" / f"{bid}__t3_team_match_values.parquet").exists()
