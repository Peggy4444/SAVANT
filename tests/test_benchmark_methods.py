"""xT-State + xT-Action unit tests (BM-5). Synthetic SPADL DataFrames, offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from savant.benchmarking.models import (
    DAxT, Liu2020, METHODS_BY_ID, OBSO_FF, PlayeRank, VAEP,
    VanRoyLeavingGoals, VanRoyMDP, XTAction, XTState,
)

REPO = Path(__file__).resolve().parents[1]


# --- Fixture: minimal SPADL DataFrame with 3 games, 2 teams -------

@pytest.fixture
def spadl_sample() -> pd.DataFrame:
    """Two games, two teams.  Some passes advancing the ball + a few shots."""
    from socceraction.spadl import config as spadlcfg
    pass_id = spadlcfg.actiontypes.index("pass")
    shot_id = spadlcfg.actiontypes.index("shot")
    dribble_id = spadlcfg.actiontypes.index("dribble")
    success = spadlcfg.results.index("success")
    fail = spadlcfg.results.index("fail")

    # Actions arranged so Team A does many high-value forward passes in game 10.
    rows = []
    # --- game 10, team 1 (attacking): forward passes near opponent goal ---
    for i, (sx, sy, ex, ey) in enumerate([
        (30, 34, 60, 34), (60, 34, 90, 34),
        (90, 34, 95, 30), (95, 30, 100, 40),
    ]):
        rows.append({
            "period_id": 1, "time_seconds": 10.0 + i,
            "team_id": 1, "type_id": pass_id, "result_id": success,
            "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
            "game_id": 10, "action_id": i, "player_id": 100 + i,
            "original_event_id": f"evt{i}",
        })
    rows.append({
        "period_id": 1, "time_seconds": 30.0,
        "team_id": 1, "type_id": shot_id, "result_id": success,
        "start_x": 100, "start_y": 40, "end_x": 105, "end_y": 34,
        "game_id": 10, "action_id": 100, "player_id": 200,
        "original_event_id": "shot1",
    })
    # --- game 10, team 2 (defending): mostly backward passes ---
    for i, (sx, sy, ex, ey) in enumerate([
        (60, 34, 30, 34), (30, 34, 10, 40),
    ]):
        rows.append({
            "period_id": 1, "time_seconds": 40.0 + i,
            "team_id": 2, "type_id": pass_id, "result_id": success,
            "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
            "game_id": 10, "action_id": 200 + i, "player_id": 300 + i,
            "original_event_id": f"back{i}",
        })
    # --- game 20, team 2 (attacking): one shot from close range ---
    rows.append({
        "period_id": 1, "time_seconds": 60.0,
        "team_id": 2, "type_id": pass_id, "result_id": success,
        "start_x": 60, "start_y": 34, "end_x": 100, "end_y": 34,
        "game_id": 20, "action_id": 400, "player_id": 500,
        "original_event_id": "evtA",
    })
    rows.append({
        "period_id": 1, "time_seconds": 61.0,
        "team_id": 2, "type_id": shot_id, "result_id": success,
        "start_x": 100, "start_y": 34, "end_x": 105, "end_y": 34,
        "game_id": 20, "action_id": 401, "player_id": 501,
        "original_event_id": "shot2",
    })
    # --- game 20, team 1 (defending): failed pass in own half ---
    rows.append({
        "period_id": 1, "time_seconds": 70.0,
        "team_id": 1, "type_id": pass_id, "result_id": fail,
        "start_x": 20, "start_y": 34, "end_x": 40, "end_y": 34,
        "game_id": 20, "action_id": 500, "player_id": 600,
        "original_event_id": "evtB",
    })
    df = pd.DataFrame(rows)
    # SPADL requires a bodypart_id — default to foot (0) so VAEP's bodypart
    # one-hot features have a valid column to fire on.
    df["bodypart_id"] = 0
    return df


# --- xT-State tests -------------------------------------------------


def test_xt_state_fit_produces_valid_map(spadl_sample):
    model = XTState(n_cols=6, n_rows=4, n_iterations=3)
    model.fit(spadl_sample)
    assert model.xt_map is not None
    assert model.xt_map.shape == (4, 6)
    assert (model.xt_map >= 0).all()
    assert (model.xt_map <= 1).all()


def test_xt_state_forward_pass_has_positive_value(spadl_sample):
    """A pass from far-from-goal to near-goal should have positive xT delta."""
    model = XTState(n_cols=6, n_rows=4, n_iterations=3)
    model.fit(spadl_sample)
    values = model._per_action_values(spadl_sample)
    forward_passes = spadl_sample[
        (spadl_sample["team_id"] == 1)
        & (spadl_sample["end_x"] > spadl_sample["start_x"])
    ]
    if len(forward_passes):
        assert values[forward_passes.index].sum() > 0


def test_xt_state_predict_returns_all_team_match_pairs(spadl_sample):
    model = XTState(n_cols=6, n_rows=4)
    model.fit(spadl_sample)
    preds = model.predict_team_match_values(spadl_sample)
    # Both teams appear in both games in the fixture except game 20 where
    # only team 2 attacks — but team 1 has a failed pass, so it should appear.
    pairs = set(zip(preds["game_id"], preds["team_id"]))
    assert (10, 1) in pairs and (10, 2) in pairs
    assert (20, 2) in pairs and (20, 1) in pairs


# --- xT-Action tests -----------------------------------------------


def test_xt_action_fit_builds_two_models(spadl_sample):
    model = XTAction(look_ahead=5)
    model.fit(spadl_sample)
    assert model.shot_clf is not None
    # There are 2 shots in the sample → enough training data for the xg regressor
    assert model.xg_reg is not None


def test_xt_action_predictions_are_nonnegative(spadl_sample):
    model = XTAction(look_ahead=5)
    model.fit(spadl_sample)
    preds = model.predict_team_match_values(spadl_sample)
    assert (preds["value"] >= 0).all()


def test_method_registry_keys():
    assert set(METHODS_BY_ID) == {
        "xt_state_singh", "xt_action_sumpter", "vaep_socceraction",
        "daxt_merhej", "vanroy_mdp", "vanroy_leaving_goals",
        "playerank_pappalardo", "obso_ff_spearman", "liu_2020_ttdp",
    }


def test_methods_advertise_correct_corpus_paper_id():
    expected = {
        "xt_state_singh":       "singh_2018_xt",
        "xt_action_sumpter":    "singh_2018_xt",
        "vaep_socceraction":    "decroos_2019_vaep",
        "daxt_merhej":          "merhej_2021_daxt",
        "vanroy_mdp":           "vanroy_2021_markov_reward",
        "vanroy_leaving_goals": "vanroy_2021_leaving_goals",
        "playerank_pappalardo": "pappalardo_2019_playerank",
        "obso_ff_spearman":     "spearman_2018_beyondxg",
        "liu_2020_ttdp":        "liu_2020_deep_soccer",
    }
    for mid, cls in METHODS_BY_ID.items():
        assert cls.corpus_paper_id == expected[mid]


# --- VAEP tests ----------------------------------------------------


def test_vaep_registry_and_metadata():
    assert VAEP.id == "vaep_socceraction"
    assert VAEP.corpus_paper_id == "decroos_2019_vaep"
    # T1/T2/T3/T4 all reported per Track-B spec.
    assert len(VAEP.reports_on_tasks) == 4


def test_vaep_fit_and_predict_smoke(spadl_sample):
    """Tiny hyperparameters — just check the pipeline runs and outputs the
    expected columns. Numeric quality is asserted on real data in BM-10."""
    model = VAEP(
        nb_prev_actions=1,
        label_nr_actions=3,
        xgb_n_estimators=5,
        xgb_max_depth=2,
    )
    model.fit(spadl_sample)
    assert model.scores_clf is not None
    assert model.concedes_clf is not None
    assert model._feature_columns is not None
    preds = model.predict_team_match_values(spadl_sample)
    assert list(preds.columns) == ["game_id", "team_id", "value"]
    # Every (game, team) pair present in the input is present in the output.
    pairs_in  = set(zip(spadl_sample["game_id"], spadl_sample["team_id"]))
    pairs_out = set(zip(preds["game_id"], preds["team_id"]))
    assert pairs_in == pairs_out


# --- DAxT tests ----------------------------------------------------


@pytest.fixture
def spadl_with_defensive(spadl_sample) -> pd.DataFrame:
    """Adds a tackle and an interception to the shared fixture so DAxT
    has at least one defensive action to value in each game."""
    from socceraction.spadl import config as spadlcfg
    tackle_id = spadlcfg.actiontypes.index("tackle")
    interception_id = spadlcfg.actiontypes.index("interception")
    success = spadlcfg.results.index("success")
    extra = pd.DataFrame([
        # tackle by team 2 mid-way through game 10 (after some passes → window has data)
        {"period_id": 1, "time_seconds": 35.0, "team_id": 2,
         "type_id": tackle_id, "result_id": success,
         "start_x": 80, "start_y": 34, "end_x": 80, "end_y": 34,
         "game_id": 10, "action_id": 300, "player_id": 400,
         "original_event_id": "tk1", "bodypart_id": 0},
        # interception by team 1 in game 20
        {"period_id": 1, "time_seconds": 65.0, "team_id": 1,
         "type_id": interception_id, "result_id": success,
         "start_x": 50, "start_y": 34, "end_x": 50, "end_y": 34,
         "game_id": 20, "action_id": 450, "player_id": 700,
         "original_event_id": "int1", "bodypart_id": 0},
    ])
    return pd.concat([spadl_sample, extra], ignore_index=True)


def test_daxt_registry_and_metadata():
    assert DAxT.id == "daxt_merhej"
    assert DAxT.corpus_paper_id == "merhej_2021_daxt"


def test_daxt_fit_and_predict_smoke(spadl_with_defensive):
    model = DAxT(
        window_size=2,
        xt_n_iterations=2,
        hidden_layer_sizes=(4, 4),
        max_iter=20,
    )
    model.fit(spadl_with_defensive)
    assert model.xt_state is not None
    assert model.mlp is not None
    preds = model.predict_team_match_values(spadl_with_defensive)
    assert list(preds.columns) == ["game_id", "team_id", "value"]
    # DAxT values only fire on defensive actions — every (game, team) pair
    # is still reported (with 0 for teams without any interceptions/tackles).
    pairs = set(zip(preds["game_id"], preds["team_id"]))
    assert (10, 2) in pairs and (20, 1) in pairs
    # At least the two teams that made a defensive action should have nonzero.
    nonzero = preds[preds["value"] != 0]
    assert len(nonzero) >= 1


# --- Van Roy MDP tests --------------------------------------------


def test_vanroy_mdp_registry_and_metadata():
    assert VanRoyMDP.id == "vanroy_mdp"
    assert VanRoyMDP.corpus_paper_id == "vanroy_2021_markov_reward"


def test_vanroy_mdp_fit_produces_bounded_state_values(spadl_sample):
    model = VanRoyMDP(n_cols=6, n_rows=4)
    model.fit(spadl_sample)
    assert model.V is not None
    assert model.V.shape == (24,)
    # V(s) is a probability → [0, 1].
    assert (model.V >= 0).all() and (model.V <= 1).all()


def test_vanroy_mdp_shots_produce_positive_action_values(spadl_sample):
    """A successful shot delivers `1 - V(start)` which is always > 0 when
    V(start) < 1 — i.e. finding the ball in the net is worth more than
    the state you left."""
    model = VanRoyMDP(n_cols=6, n_rows=4)
    model.fit(spadl_sample)
    vals = model._per_action_values(spadl_sample)
    shot_mask = spadl_sample["type_id"].isin(list(model._shot_ids)) & (
        spadl_sample["result_id"] == model._success_id
    )
    assert (vals[shot_mask.values] > 0).all()


def test_vanroy_mdp_predict_returns_all_team_match_pairs(spadl_sample):
    model = VanRoyMDP(n_cols=6, n_rows=4)
    model.fit(spadl_sample)
    preds = model.predict_team_match_values(spadl_sample)
    pairs_in  = set(zip(spadl_sample["game_id"], spadl_sample["team_id"]))
    pairs_out = set(zip(preds["game_id"], preds["team_id"]))
    assert pairs_in == pairs_out


# --- Van Roy Leaving Goals tests ----------------------------------


def test_leaving_goals_registry_and_metadata():
    assert VanRoyLeavingGoals.id == "vanroy_leaving_goals"
    assert VanRoyLeavingGoals.corpus_paper_id == "vanroy_2021_leaving_goals"


def test_leaving_goals_fits_per_team_v(spadl_sample):
    model = VanRoyLeavingGoals(n_cols=6, n_rows=4)
    model.fit(spadl_sample)
    # Two teams in the fixture — both should have their own V.
    assert set(model.V_by_team.keys()) == {1, 2}
    for V in model.V_by_team.values():
        assert V.shape == (24,)
        assert (V >= 0).all() and (V <= 1).all()
    # Pool fallback is populated so unseen teams don't fault.
    assert model.V_pool is not None


def test_leaving_goals_falls_back_to_pool_for_unseen_team(spadl_sample):
    """A team not in training must use the pool V, not raise."""
    model = VanRoyLeavingGoals(n_cols=6, n_rows=4)
    model.fit(spadl_sample)
    unseen = spadl_sample.copy()
    unseen["team_id"] = 999
    preds = model.predict_team_match_values(unseen)
    assert set(preds["team_id"]) == {999}
    # Value should be finite (used pool V).
    assert np.isfinite(preds["value"]).all()


def test_leaving_goals_predict_returns_all_team_match_pairs(spadl_sample):
    model = VanRoyLeavingGoals(n_cols=6, n_rows=4)
    model.fit(spadl_sample)
    preds = model.predict_team_match_values(spadl_sample)
    pairs_in  = set(zip(spadl_sample["game_id"], spadl_sample["team_id"]))
    pairs_out = set(zip(preds["game_id"], preds["team_id"]))
    assert pairs_in == pairs_out


# --- PlayeRank tests ----------------------------------------------


def test_playerank_registry_and_metadata():
    assert PlayeRank.id == "playerank_pappalardo"
    assert PlayeRank.corpus_paper_id == "pappalardo_2019_playerank"


def test_playerank_computes_binary_win_labels(spadl_sample):
    """Training relies on `_match_outcomes`. Verify it produces sane
    win/non-win labels — team 1 scores in game 10, team 2 scores in game 20."""
    model = PlayeRank()
    model._load_spadl_labels()
    outcomes = model._match_outcomes(spadl_sample).set_index(["game_id", "team_id"])["win"]
    assert outcomes.loc[(10, 1)] == 1     # team 1 shot & scored in game 10
    assert outcomes.loc[(10, 2)] == 0
    assert outcomes.loc[(20, 2)] == 1     # team 2 shot & scored in game 20
    assert outcomes.loc[(20, 1)] == 0


def test_playerank_fit_and_predict_smoke(spadl_sample):
    model = PlayeRank()
    model.fit(spadl_sample)
    preds = model.predict_team_match_values(spadl_sample)
    assert list(preds.columns) == ["game_id", "team_id", "value"]
    # Every (game, team) pair present in input is present in output.
    pairs_in  = set(zip(spadl_sample["game_id"], spadl_sample["team_id"]))
    pairs_out = set(zip(preds["game_id"], preds["team_id"]))
    assert pairs_in == pairs_out
    assert np.isfinite(preds["value"]).all()


def test_playerank_single_class_training_falls_back_gracefully(spadl_sample):
    """If the training split has only one outcome class (e.g. no draws
    that are non-wins), LSVC would refuse to fit. Wrapper must return
    zero predictions rather than crash."""
    # Force all rows to draws → both teams get 'non-win' → single class.
    from socceraction.spadl import config as spadlcfg
    shot_id = spadlcfg.actiontypes.index("shot")
    success = spadlcfg.results.index("success")
    df = spadl_sample.copy()
    # Kill both shots so no team scores → all teams are non-wins.
    df.loc[df["type_id"] == shot_id, "result_id"] = spadlcfg.results.index("fail")
    model = PlayeRank()
    model.fit(df)
    assert model.svc is None
    preds = model.predict_team_match_values(df)
    assert (preds["value"] == 0.0).all()


# --- OBSO_FF tests -------------------------------------------------


def _synthetic_360_row(event_id: str, match_id: int, teammate: bool,
                       actor: bool, keeper: bool,
                       loc_x: float, loc_y: float) -> dict:
    return {
        "id": event_id,
        "match_id": match_id,
        "teammate": teammate,
        "actor": actor,
        "keeper": keeper,
        "location": [loc_x, loc_y],
        "visible_area": [0, 0, 120, 0, 120, 80, 0, 80],
    }


def test_obso_ff_registry_and_metadata():
    assert OBSO_FF.id == "obso_ff_spearman"
    assert OBSO_FF.corpus_paper_id == "spearman_2018_beyondxg"
    assert OBSO_FF.requires_360 is True


def test_obso_ff_returns_zero_when_no_frames(spadl_sample):
    model = OBSO_FF()
    preds = model.predict_team_match_values(spadl_sample, frames=None)
    assert (preds["value"] == 0.0).all()
    pairs_in  = set(zip(spadl_sample["game_id"], spadl_sample["team_id"]))
    pairs_out = set(zip(preds["game_id"], preds["team_id"]))
    assert pairs_in == pairs_out


def test_obso_ff_scores_a_frame_near_goal_higher_than_midfield(spadl_sample):
    """The physics: freeze frame with attacker near the goal + no blocking
    defenders should score STRICTLY higher OBSO than the same setup at
    midfield."""
    model = OBSO_FF(grid_x=15, grid_y=10)

    # Attacker with the ball at (110, 40) [near goal] vs at (60, 40) [midfield].
    near_goal = [
        _synthetic_360_row("evt_near", 10, teammate=True,  actor=True,  keeper=False, loc_x=110, loc_y=40),
        _synthetic_360_row("evt_near", 10, teammate=True,  actor=False, keeper=False, loc_x=108, loc_y=44),
        _synthetic_360_row("evt_near", 10, teammate=False, actor=False, keeper=True,  loc_x=118, loc_y=40),
        _synthetic_360_row("evt_near", 10, teammate=False, actor=False, keeper=False, loc_x= 60, loc_y=40),
    ]
    midfield = [
        _synthetic_360_row("evt_mid", 10, teammate=True,  actor=True,  keeper=False, loc_x=60, loc_y=40),
        _synthetic_360_row("evt_mid", 10, teammate=True,  actor=False, keeper=False, loc_x=62, loc_y=44),
        _synthetic_360_row("evt_mid", 10, teammate=False, actor=False, keeper=True,  loc_x=118, loc_y=40),
        _synthetic_360_row("evt_mid", 10, teammate=False, actor=False, keeper=False, loc_x=70, loc_y=40),
    ]
    frames = pd.DataFrame(near_goal + midfield)

    # Attach synthetic SPADL rows so the event id → (game, team) join works.
    from socceraction.spadl import config as spadlcfg
    fake_actions = pd.DataFrame([
        {"game_id": 10, "team_id": 1, "type_id": spadlcfg.actiontypes.index("shot"),
         "result_id": spadlcfg.results.index("success"), "start_x": 100, "start_y": 40,
         "end_x": 105, "end_y": 40, "period_id": 1, "time_seconds": 10.0,
         "action_id": 1, "player_id": 100, "original_event_id": "evt_near", "bodypart_id": 0},
        {"game_id": 10, "team_id": 2, "type_id": spadlcfg.actiontypes.index("pass"),
         "result_id": spadlcfg.results.index("success"), "start_x": 60, "start_y": 40,
         "end_x": 70, "end_y": 40, "period_id": 1, "time_seconds": 20.0,
         "action_id": 2, "player_id": 200, "original_event_id": "evt_mid", "bodypart_id": 0},
    ])

    preds = OBSO_FF(grid_x=15, grid_y=10).predict_team_match_values(
        fake_actions, frames=frames,
    )
    team_near = preds[preds["team_id"] == 1]["value"].iloc[0]
    team_mid  = preds[preds["team_id"] == 2]["value"].iloc[0]
    assert team_near > team_mid > 0


# --- Liu 2020 tests ------------------------------------------------


def test_liu_2020_registry_and_metadata():
    assert Liu2020.id == "liu_2020_ttdp"
    assert Liu2020.corpus_paper_id == "liu_2020_deep_soccer"


def test_liu_2020_next_goal_labels_are_correct(spadl_sample):
    """Actions before team 1's goal in game 10 get scores=1 for team 1 and
    concedes=1 for team 2. Actions AFTER the goal have no subsequent goal
    in the game → both labels 0. Verifies the unbounded look-ahead logic."""
    model = Liu2020()
    model._load_spadl_labels()

    g10 = spadl_sample[spadl_sample["game_id"] == 10].sort_values(
        ["period_id", "time_seconds", "action_id"]
    ).reset_index(drop=True)
    labels = model._next_goal_labels(g10)

    from socceraction.spadl import config as spadlcfg
    shot_ids = {spadlcfg.actiontypes.index(n) for n in ("shot", "shot_penalty", "shot_freekick")}
    success = spadlcfg.results.index("success")
    is_goal = g10["type_id"].isin(shot_ids) & (g10["result_id"] == success)
    first_goal_pos = int(is_goal.idxmax())   # first row where goal fires

    # Before-goal: team 1 scores next → scores=1 for team 1, concedes=1 for team 2.
    for i in range(first_goal_pos):
        if g10["team_id"].iloc[i] == 1:
            assert labels["scores"].iloc[i] == 1 and labels["concedes"].iloc[i] == 0
        else:
            assert labels["scores"].iloc[i] == 0 and labels["concedes"].iloc[i] == 1
    # At-goal: same as before-goal (the goal action itself gets scores=1).
    assert labels["scores"].iloc[first_goal_pos] == 1
    # After-goal: no next goal → both labels 0.
    for i in range(first_goal_pos + 1, len(g10)):
        assert labels["scores"].iloc[i] == 0
        assert labels["concedes"].iloc[i] == 0


def test_liu_2020_fit_and_predict_smoke(spadl_sample):
    model = Liu2020(
        nb_prev_actions=1,
        xgb_n_estimators=5,
        xgb_max_depth=2,
    )
    model.fit(spadl_sample)
    assert model.scores_clf is not None
    assert model.concedes_clf is not None
    preds = model.predict_team_match_values(spadl_sample)
    assert list(preds.columns) == ["game_id", "team_id", "value"]
    pairs_in  = set(zip(spadl_sample["game_id"], spadl_sample["team_id"]))
    pairs_out = set(zip(preds["game_id"], preds["team_id"]))
    assert pairs_in == pairs_out


# --- BM-12: predict_per_action across all wrappers -------------


def _wrapper_returns_per_action_frame(model, spadl_df):
    """Every wrapper's predict_per_action must return a DataFrame with the
    canonical [game_id, team_id, action_id, value] schema."""
    df = model.predict_per_action(spadl_df)
    assert set(df.columns) >= {"game_id", "team_id", "action_id", "value"}
    # Row count matches or exceeds the input (some wrappers drop non-eligible
    # actions, others emit NaN — but must never produce cross-join duplicates).
    assert len(df) == len(spadl_df) or len(df) == 0


def test_xt_state_predict_per_action_shape(spadl_sample):
    m = XTState(n_cols=4, n_rows=3, n_iterations=2)
    m.fit(spadl_sample)
    _wrapper_returns_per_action_frame(m, spadl_sample)


def test_xt_action_predict_per_action_shape(spadl_sample):
    m = XTAction(look_ahead=3)
    m.fit(spadl_sample)
    _wrapper_returns_per_action_frame(m, spadl_sample)


def test_vaep_predict_per_action_shape(spadl_sample):
    m = VAEP(nb_prev_actions=1, label_nr_actions=3, xgb_n_estimators=5, xgb_max_depth=2)
    m.fit(spadl_sample)
    _wrapper_returns_per_action_frame(m, spadl_sample)


def test_daxt_predict_per_action_has_nans_for_non_defensive(spadl_with_defensive):
    m = DAxT(window_size=2, xt_n_iterations=2, hidden_layer_sizes=(4,), max_iter=20)
    m.fit(spadl_with_defensive)
    df = m.predict_per_action(spadl_with_defensive)
    _wrapper_returns_per_action_frame(m, spadl_with_defensive)
    # Non-defensive rows should have NaN values.
    from socceraction.spadl import config as spadlcfg
    defensive_types = {
        spadlcfg.actiontypes.index("tackle"),
        spadlcfg.actiontypes.index("interception"),
    }
    is_defensive = spadl_with_defensive["type_id"].isin(defensive_types).values
    # Merge on (game_id, action_id) to align — df row order may differ.
    merged = spadl_with_defensive.merge(
        df, on=["game_id", "action_id"], how="left", suffixes=("_orig", "")
    )
    non_def_mask = ~merged["type_id"].isin(defensive_types)
    assert merged.loc[non_def_mask, "value"].isna().all()


def test_daxt_predict_per_action_has_no_duplicate_rows(spadl_with_defensive):
    """Regression guard for the (game_id + action_id) merge bug — action_id
    is only unique within a game, so a single-key merge produces cross-game
    duplicates. Two games in the fixture with overlapping action_ids trigger
    this if the merge is broken."""
    m = DAxT(window_size=2, xt_n_iterations=2, hidden_layer_sizes=(4,), max_iter=20)
    m.fit(spadl_with_defensive)
    df = m.predict_per_action(spadl_with_defensive)
    assert len(df) == len(spadl_with_defensive)


def test_vanroy_mdp_predict_per_action_shape(spadl_sample):
    m = VanRoyMDP(n_cols=6, n_rows=4)
    m.fit(spadl_sample)
    _wrapper_returns_per_action_frame(m, spadl_sample)


def test_leaving_goals_predict_per_action_shape(spadl_sample):
    m = VanRoyLeavingGoals(n_cols=6, n_rows=4)
    m.fit(spadl_sample)
    _wrapper_returns_per_action_frame(m, spadl_sample)


def test_playerank_predict_per_action_shape(spadl_sample):
    m = PlayeRank()
    m.fit(spadl_sample)
    _wrapper_returns_per_action_frame(m, spadl_sample)


def test_liu_2020_predict_per_action_shape(spadl_sample):
    m = Liu2020(nb_prev_actions=1, xgb_n_estimators=5, xgb_max_depth=2)
    m.fit(spadl_sample)
    _wrapper_returns_per_action_frame(m, spadl_sample)


# --- Slow: end-to-end CLI with methods --------------------------


@pytest.mark.slow
def test_run_benchmark_dispatches_methods_end_to_end(tmp_path: Path):
    import os
    import subprocess
    import sys as _sys
    import json as _json
    r = subprocess.run(
        [_sys.executable, str(REPO / "scripts" / "run_benchmark.py"),
         "--max-matches", "3",
         "--out-dir", str(tmp_path)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, (
        f"CLI failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    manifest = _json.loads((tmp_path / "run_manifest.json").read_text())
    assert set(manifest["methods"]) == {
        "xt_state_singh", "xt_action_sumpter", "vaep_socceraction",
        "daxt_merhej", "vanroy_mdp", "vanroy_leaving_goals",
        "playerank_pappalardo", "obso_ff_spearman", "liu_2020_ttdp",
    }
    for mid in ("xt_state_singh", "xt_action_sumpter", "vaep_socceraction",
                "daxt_merhej", "vanroy_mdp", "vanroy_leaving_goals",
                "playerank_pappalardo", "obso_ff_spearman", "liu_2020_ttdp"):
        assert (tmp_path / "methods" / f"{mid}__t3_team_match_values.parquet").exists()
