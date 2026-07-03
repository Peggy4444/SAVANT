"""BM-10 evaluation module tests. Synthetic frames, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from savant.benchmarking.evaluation import (
    T2_OUTCOME_CLASSES, T2_OUTCOME_INDEX, bootstrap_ci_by_match, brier, ece,
    ece_multiclass, evaluate_t1, evaluate_t2, evaluate_t3, evaluate_t4,
    load_predictions, log_loss_binary, log_loss_multiclass, mae, pearson_rho,
    roc_auc, score, spearman_rho, t1_goal_next_k_actions,
    t2_possession_outcome, team_match_xg, team_match_xg_diff, top1_accuracy,
    wilcoxon_pairwise,
)


# --- metrics -------------------------------------------------------


def test_pearson_rho_perfect_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 3.0
    assert pearson_rho(x, y) == pytest.approx(1.0)


def test_pearson_rho_anti_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_rho(x, -x) == pytest.approx(-1.0)


def test_pearson_rho_handles_zero_variance():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([2.0, 3.0, 4.0, 5.0])
    assert np.isnan(pearson_rho(x, y))


def test_pearson_rho_drops_nans_pairwise():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_rho(x, y) == pytest.approx(1.0)


def test_spearman_rho_monotonic_but_nonlinear():
    """Spearman = 1 for any strictly monotonic transform; Pearson < 1."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x ** 3
    assert spearman_rho(x, y) == pytest.approx(1.0)
    assert pearson_rho(x, y) < 1.0


def test_mae_is_positive_and_zero_on_identical():
    x = np.array([1.0, 2.0, 3.0])
    assert mae(x, x) == pytest.approx(0.0)
    assert mae(x, x + 2.0) == pytest.approx(2.0)


def test_score_dispatch_by_name():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 2.5, 3.5])
    assert score("mae", x, y) == pytest.approx(0.5)
    assert score("pearson_rho", x, y) == pytest.approx(1.0)


def test_score_raises_on_unknown_metric():
    with pytest.raises(KeyError):
        score("mystery_metric", np.array([1]), np.array([2]))


# --- targets -------------------------------------------------------


@pytest.fixture
def synthetic_events() -> pd.DataFrame:
    """Two matches, two teams per match, with shots + xG values."""
    return pd.DataFrame([
        # game 10: team 1 outshoots team 2 by xG (0.6 vs 0.1)
        {"match_id": 10, "team_id": 1, "type": "Shot", "shot_statsbomb_xg": 0.4},
        {"match_id": 10, "team_id": 1, "type": "Shot", "shot_statsbomb_xg": 0.2},
        {"match_id": 10, "team_id": 2, "type": "Shot", "shot_statsbomb_xg": 0.1},
        {"match_id": 10, "team_id": 1, "type": "Pass", "shot_statsbomb_xg": None},
        {"match_id": 10, "team_id": 2, "type": "Pass", "shot_statsbomb_xg": None},
        # game 20: teams even
        {"match_id": 20, "team_id": 3, "type": "Shot", "shot_statsbomb_xg": 0.3},
        {"match_id": 20, "team_id": 4, "type": "Shot", "shot_statsbomb_xg": 0.3},
    ])


def test_team_match_xg_sums_shot_xg(synthetic_events):
    result = team_match_xg(synthetic_events).set_index(["game_id", "team_id"])["team_xg"]
    assert result.loc[(10, 1)] == pytest.approx(0.6)
    assert result.loc[(10, 2)] == pytest.approx(0.1)
    assert result.loc[(20, 3)] == pytest.approx(0.3)
    assert result.loc[(20, 4)] == pytest.approx(0.3)


def test_team_match_xg_diff_is_signed_symmetric(synthetic_events):
    result = team_match_xg_diff(synthetic_events).set_index(["game_id", "team_id"])["xg_diff"]
    assert result.loc[(10, 1)] == pytest.approx(0.5)     # 0.6 - 0.1
    assert result.loc[(10, 2)] == pytest.approx(-0.5)
    assert result.loc[(20, 3)] == pytest.approx(0.0)
    assert result.loc[(20, 4)] == pytest.approx(0.0)


# --- bootstrap CIs ------------------------------------------------


def test_bootstrap_point_estimate_matches_metric(synthetic_events):
    preds = pd.DataFrame([
        {"game_id": 10, "team_id": 1, "value": 5.0},
        {"game_id": 10, "team_id": 2, "value": 1.0},
        {"game_id": 20, "team_id": 3, "value": 3.0},
        {"game_id": 20, "team_id": 4, "value": 3.0},
    ])
    target = team_match_xg_diff(synthetic_events)
    r = bootstrap_ci_by_match(
        preds, target, target_col="xg_diff",
        metric_fn=pearson_rho, n_resamples=50, seed=0,
    )
    # Point estimate == Pearson ρ on full data.
    df = preds.merge(target, on=["game_id", "team_id"])
    expected = pearson_rho(df["value"].values, df["xg_diff"].values)
    assert r["point"] == pytest.approx(expected)


def test_bootstrap_ci_brackets_point_estimate():
    """CI should generally bracket the point estimate — trivial invariant."""
    rng = np.random.default_rng(42)
    n_matches = 30
    game_ids = np.repeat(np.arange(n_matches), 2)
    team_ids = np.tile([1, 2], n_matches)
    preds = pd.DataFrame({
        "game_id": game_ids, "team_id": team_ids,
        "value": rng.standard_normal(2 * n_matches),
    })
    target = pd.DataFrame({
        "game_id": game_ids, "team_id": team_ids,
        "xg_diff": preds["value"].values + 0.5 * rng.standard_normal(2 * n_matches),
    })
    r = bootstrap_ci_by_match(
        preds, target, target_col="xg_diff",
        metric_fn=pearson_rho, n_resamples=200, seed=0,
    )
    assert r["lo"] <= r["point"] <= r["hi"]
    assert r["hi"] > r["lo"]


def test_bootstrap_ci_returns_nan_on_single_match():
    preds = pd.DataFrame({"game_id": [1, 1], "team_id": [1, 2], "value": [1.0, 2.0]})
    target = pd.DataFrame({"game_id": [1, 1], "team_id": [1, 2], "xg_diff": [0.5, -0.5]})
    r = bootstrap_ci_by_match(
        preds, target, target_col="xg_diff",
        metric_fn=pearson_rho, n_resamples=50, seed=0,
    )
    assert np.isnan(r["lo"])
    assert np.isnan(r["hi"])


# --- Wilcoxon + BH correction ------------------------------------


def test_benjamini_hochberg_reduces_to_bh_by_hand():
    """Compare against hand-computed BH on 4 p-values."""
    from savant.benchmarking.evaluation.stats import _benjamini_hochberg
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = _benjamini_hochberg(p)
    # Sorted: 0.005, 0.01, 0.03, 0.04 → BH = min(p*n/rank, 1) then monotonicised
    # 0.005 * 4/1 = 0.02, 0.01 * 4/2 = 0.02, 0.03 * 4/3 = 0.04, 0.04 * 4/4 = 0.04
    # Monotonic from top: [0.02, 0.02, 0.04, 0.04].
    # Un-permute: original order was [0.01, 0.04, 0.03, 0.005] → ranks [2, 4, 3, 1]
    # → adjusted [0.02, 0.04, 0.04, 0.02].
    assert adj[3] == pytest.approx(0.02)   # smallest original p
    assert adj[0] == pytest.approx(0.02)
    assert adj[1] == pytest.approx(0.04)
    assert adj[2] == pytest.approx(0.04)


def test_wilcoxon_pairwise_detects_systematic_offset():
    """Two 'methods' where B is systematically off by a big constant should
    yield a significant p-value in the pairwise test."""
    rng = np.random.default_rng(0)
    n = 30
    game_ids = np.repeat(np.arange(n), 2)
    team_ids = np.tile([1, 2], n)
    target_diff = rng.standard_normal(2 * n) * 0.5

    preds_a = pd.DataFrame({"game_id": game_ids, "team_id": team_ids,
                            "value": target_diff + rng.standard_normal(2 * n) * 0.1})
    preds_b = pd.DataFrame({"game_id": game_ids, "team_id": team_ids,
                            "value": target_diff + 5.0})   # huge bias
    target = pd.DataFrame({"game_id": game_ids, "team_id": team_ids,
                           "xg_diff": target_diff})
    res = wilcoxon_pairwise(
        {"method_a": preds_a, "method_b": preds_b},
        target, target_col="xg_diff",
    )
    assert len(res) == 1
    assert res.iloc[0]["p_value_bh"] < 0.05
    assert res.iloc[0]["significant"]


# --- T3 evaluation end-to-end -----------------------------------


def test_evaluate_t3_orders_methods_by_pearson(synthetic_events):
    """A method that perfectly tracks xG_diff must top the leaderboard."""
    target = team_match_xg_diff(synthetic_events)
    perfect = target.rename(columns={"xg_diff": "value"})[["game_id", "team_id", "value"]]
    noisy = perfect.copy()
    noisy["value"] = noisy["value"] + np.array([0.5, -0.5, 0.5, -0.5])
    inverted = perfect.copy()
    inverted["value"] = -inverted["value"]

    leaderboard = evaluate_t3(
        {"perfect": perfect, "noisy": noisy, "inverted": inverted},
        synthetic_events,
        n_bootstrap=50,
    )
    # Best (highest ρ) at the top.
    assert leaderboard.iloc[0]["model_id"] == "perfect"
    assert leaderboard.iloc[-1]["model_id"] == "inverted"
    assert leaderboard.iloc[0]["pearson_rho_point"] > leaderboard.iloc[1]["pearson_rho_point"]


def test_evaluate_t4_signs_delta_correctly(synthetic_events):
    """T4 Δ = train − holdout. A method that ρ=0.9 on train but 0.4 on
    holdout should get Δ = +0.5."""
    t3_train = pd.DataFrame([{
        "model_id": "m1",
        "pearson_rho_point": 0.9, "pearson_rho_lo": 0.8, "pearson_rho_hi": 0.95,
        "spearman_rho_point": 0.85, "spearman_rho_lo": 0.7, "spearman_rho_hi": 0.9,
        "mae_point": 0.3, "mae_lo": 0.2, "mae_hi": 0.4,
    }])
    t3_holdout = pd.DataFrame([{
        "model_id": "m1",
        "pearson_rho_point": 0.4, "pearson_rho_lo": 0.2, "pearson_rho_hi": 0.5,
        "spearman_rho_point": 0.35, "spearman_rho_lo": 0.1, "spearman_rho_hi": 0.4,
        "mae_point": 0.5, "mae_lo": 0.4, "mae_hi": 0.6,
    }])
    t4 = evaluate_t4(t3_train, t3_holdout)
    assert t4.iloc[0]["pearson_rho_delta"] == pytest.approx(0.5)
    assert t4.iloc[0]["mae_delta"] == pytest.approx(-0.2)


# --- End-to-end runner on a tiny synthetic benchmark dir --------


def test_runner_writes_leaderboard(tmp_path, synthetic_events):
    """Simulate a benchmark run: drop two prediction parquets and confirm
    the runner writes leaderboard + per_task artefacts."""
    (tmp_path / "methods").mkdir()
    (tmp_path / "baselines").mkdir()

    target = team_match_xg_diff(synthetic_events)
    perfect = target.rename(columns={"xg_diff": "value"})[["game_id", "team_id", "value"]]
    perfect.to_parquet(tmp_path / "methods" / "perfect__t3_team_match_values.parquet",
                       index=False)
    inverted = perfect.copy()
    inverted["value"] = -inverted["value"]
    inverted.to_parquet(tmp_path / "baselines" / "inverted__t3_team_match_values.parquet",
                        index=False)

    from savant.benchmarking.evaluation import run
    manifest = run(
        bench_dir=tmp_path, events=synthetic_events,
        out_dir=tmp_path, n_bootstrap=50,
    )
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "per_task" / "t3_pearson_bootstrap.csv").exists()
    assert (tmp_path / "per_task" / "t3_wilcoxon_pairwise.csv").exists()

    leaderboard = pd.read_csv(tmp_path / "leaderboard.csv")
    assert set(leaderboard["model_id"]) == {"perfect", "inverted"}
    assert leaderboard.iloc[0]["model_id"] == "perfect"


# --- T1: binary metrics ---------------------------------------------


def test_brier_perfect_predictor_is_zero():
    y = np.array([0, 0, 1, 1])
    assert brier(y.astype(float), y) == pytest.approx(0.0)


def test_brier_always_half_is_a_quarter():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    assert brier(p, y) == pytest.approx(0.25)


def test_log_loss_perfect_predictor_is_zero():
    y = np.array([0, 1])
    # Use 0/1 exactly to hit the clip boundary; must return ~0.
    assert log_loss_binary(y.astype(float), y) == pytest.approx(0.0, abs=1e-6)


def test_ece_perfectly_calibrated_returns_zero():
    """If every bin's confidence == its accuracy, ECE == 0."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    # Sample y with P = p → perfect calibration in the limit.
    y = (rng.uniform(0, 1, 1000) < p).astype(int)
    assert ece(p, y, n_bins=10) < 0.05    # small n → some finite-sample noise


def test_ece_worst_case_is_bounded():
    """A predictor that always says 0.5 on all-1 labels has ECE = 0.5."""
    p = np.full(20, 0.5)
    y = np.ones(20, dtype=int)
    assert ece(p, y, n_bins=1) == pytest.approx(0.5)


def test_roc_auc_perfect_ranker_is_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.linspace(0.1, 0.9, 6)       # strictly monotone in y
    assert roc_auc(p, y) == pytest.approx(1.0)


def test_roc_auc_random_ranker_is_around_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = rng.uniform(0, 1, 5000)
    assert 0.45 < roc_auc(p, y) < 0.55


def test_roc_auc_returns_nan_on_single_class():
    y = np.zeros(10, dtype=int)
    p = np.linspace(0, 1, 10)
    assert np.isnan(roc_auc(p, y))


# --- T1 target construction --------------------------------------


def _synthetic_spadl_with_goal():
    """3 team-1 passes then a team-1 goal → first 3 rows should get
    y_goal_next_k = 1. Subsequent no-op rows should get 0."""
    from socceraction.spadl import config as spadlcfg
    pass_id = spadlcfg.actiontypes.index("pass")
    shot_id = spadlcfg.actiontypes.index("shot")
    success = spadlcfg.results.index("success")
    fail    = spadlcfg.results.index("fail")
    rows = []
    for i in range(3):
        rows.append({"game_id": 10, "team_id": 1, "action_id": i,
                     "period_id": 1, "time_seconds": 10.0 + i,
                     "type_id": pass_id, "result_id": success,
                     "start_x": 30, "start_y": 34, "end_x": 60, "end_y": 34})
    rows.append({"game_id": 10, "team_id": 1, "action_id": 3,
                 "period_id": 1, "time_seconds": 15.0,
                 "type_id": shot_id, "result_id": success,     # GOAL
                 "start_x": 100, "start_y": 40, "end_x": 105, "end_y": 34})
    # Two failed passes after the goal → neither should be labelled.
    for i in range(2):
        rows.append({"game_id": 10, "team_id": 2, "action_id": 4 + i,
                     "period_id": 1, "time_seconds": 20.0 + i,
                     "type_id": pass_id, "result_id": fail,
                     "start_x": 30, "start_y": 34, "end_x": 40, "end_y": 34})
    return pd.DataFrame(rows)


def test_t1_target_labels_pre_goal_actions():
    labels = t1_goal_next_k_actions(_synthetic_spadl_with_goal(), k=10)
    # Actions 0,1,2 are pre-goal same-team → 1
    for aid in (0, 1, 2):
        assert labels.loc[labels["action_id"] == aid, "y_goal_next_k"].iloc[0] == 1
    # Action 3 is the goal itself — it has no "next" same-team goal within k → 0
    assert labels.loc[labels["action_id"] == 3, "y_goal_next_k"].iloc[0] == 0
    # Actions 4,5 are post-goal → 0
    for aid in (4, 5):
        assert labels.loc[labels["action_id"] == aid, "y_goal_next_k"].iloc[0] == 0


def test_t1_target_k_horizon_is_respected():
    """A goal 15 actions away should NOT be labelled when k=10."""
    from socceraction.spadl import config as spadlcfg
    pass_id = spadlcfg.actiontypes.index("pass")
    shot_id = spadlcfg.actiontypes.index("shot")
    success = spadlcfg.results.index("success")
    rows = []
    for i in range(15):
        rows.append({"game_id": 10, "team_id": 1, "action_id": i,
                     "period_id": 1, "time_seconds": 10.0 + i * 0.5,
                     "type_id": pass_id, "result_id": success,
                     "start_x": 30, "start_y": 34, "end_x": 60, "end_y": 34})
    rows.append({"game_id": 10, "team_id": 1, "action_id": 15,
                 "period_id": 1, "time_seconds": 30.0,
                 "type_id": shot_id, "result_id": success,
                 "start_x": 100, "start_y": 40, "end_x": 105, "end_y": 34})
    labels = t1_goal_next_k_actions(pd.DataFrame(rows), k=10)
    # Action 0 → distance 15 to the goal → NOT labelled
    assert labels.loc[labels["action_id"] == 0, "y_goal_next_k"].iloc[0] == 0
    # Action 5 → distance 10 exactly → labelled (inclusive of k)
    assert labels.loc[labels["action_id"] == 5, "y_goal_next_k"].iloc[0] == 1


# --- T1 evaluation end-to-end -----------------------------------


def test_evaluate_t1_returns_one_row_per_method():
    """Synthetic 4-match dataset with per-action predictions from two methods."""
    rng = np.random.default_rng(0)
    game_ids = [1, 2, 3, 4]
    rows = []
    for gid in game_ids:
        for aid in range(20):
            rows.append({
                "game_id": gid, "team_id": (aid % 2) + 1,
                "action_id": aid, "value": rng.uniform(-1, 1),
            })
    preds_a = pd.DataFrame(rows)
    preds_b = preds_a.copy()
    preds_b["value"] = preds_b["value"] + rng.standard_normal(len(preds_b)) * 0.1

    targets = preds_a[["game_id", "team_id", "action_id"]].copy()
    targets["y_goal_next_k"] = rng.integers(0, 2, len(targets))

    res = evaluate_t1(
        per_action_by_method={"method_a": preds_a, "method_b": preds_b},
        targets=targets,
        train_game_ids={1, 2},
        test_game_ids={3, 4},
        n_bootstrap=25,
    )
    assert set(res["model_id"]) == {"method_a", "method_b"}
    for col in ("brier_point", "log_loss_binary_point", "ece_point", "roc_auc_point"):
        assert col in res.columns


# --- T2: multiclass metrics ---------------------------------------


def test_log_loss_multiclass_perfect_predictor_is_zero():
    y = np.array([0, 1, 2, 3])
    P = np.eye(4)
    assert log_loss_multiclass(P, y, n_classes=4) == pytest.approx(0.0, abs=1e-6)


def test_log_loss_multiclass_uniform_prior():
    """Uniform 1/4 over 4 classes → log-loss = ln(4)."""
    y = np.array([0, 1, 2, 3])
    P = np.full((4, 4), 0.25)
    assert log_loss_multiclass(P, y, 4) == pytest.approx(np.log(4))


def test_top1_accuracy_matches_argmax():
    y = np.array([0, 1, 2])
    P = np.array([[0.9, 0.05, 0.05],   # correct
                  [0.1, 0.7, 0.2],     # correct
                  [0.5, 0.4, 0.1]])    # wrong
    assert top1_accuracy(P, y, n_classes=3) == pytest.approx(2 / 3)


def test_ece_multiclass_perfectly_calibrated_returns_near_zero():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 4, n)
    # Confident correct predictions for even indices, uncertain for odd.
    P = np.zeros((n, 4))
    for i in range(n):
        if i % 2 == 0:
            P[i, y[i]] = 0.99
            P[i, (y[i] + 1) % 4] = 0.01
        else:
            P[i] = 0.25
    # Both slices are self-consistent (confidence == accuracy) → ECE small.
    assert ece_multiclass(P, y, n_classes=4) < 0.02


# --- T2 target: possession segmentation --------------------------


def _synthetic_spadl_possession_scenarios():
    """Four possessions in one game: goal / turnover / foul / no_goal."""
    from socceraction.spadl import config as spadlcfg
    pass_id = spadlcfg.actiontypes.index("pass")
    shot_id = spadlcfg.actiontypes.index("shot")
    foul_id = spadlcfg.actiontypes.index("foul")
    success = spadlcfg.results.index("success")
    fail    = spadlcfg.results.index("fail")

    rows = []
    aid = 0
    # Possession 1 (team 1): pass → shot=goal
    for i, (typ, res) in enumerate([(pass_id, success), (shot_id, success)]):
        rows.append({"game_id": 10, "team_id": 1, "action_id": aid, "period_id": 1,
                     "time_seconds": 10.0 + i, "type_id": typ, "result_id": res,
                     "start_x": 30, "start_y": 34, "end_x": 60, "end_y": 34}); aid += 1
    # Possession 2 (team 2): pass → passes ...
    for i, (typ, res) in enumerate([(pass_id, success), (pass_id, success)]):
        rows.append({"game_id": 10, "team_id": 2, "action_id": aid, "period_id": 1,
                     "time_seconds": 20.0 + i, "type_id": typ, "result_id": res,
                     "start_x": 30, "start_y": 34, "end_x": 60, "end_y": 34}); aid += 1
    # ... then team 1 gets the ball (→ team 2's possession ended as turnover)
    # Possession 3 (team 1): pass → foul
    for i, (typ, res) in enumerate([(pass_id, success), (foul_id, success)]):
        rows.append({"game_id": 10, "team_id": 1, "action_id": aid, "period_id": 1,
                     "time_seconds": 30.0 + i, "type_id": typ, "result_id": res,
                     "start_x": 30, "start_y": 34, "end_x": 60, "end_y": 34}); aid += 1
    # Possession 4 (team 2): pass → shot=miss
    for i, (typ, res) in enumerate([(pass_id, success), (shot_id, fail)]):
        rows.append({"game_id": 10, "team_id": 2, "action_id": aid, "period_id": 1,
                     "time_seconds": 40.0 + i, "type_id": typ, "result_id": res,
                     "start_x": 30, "start_y": 34, "end_x": 60, "end_y": 34}); aid += 1
    return pd.DataFrame(rows)


def test_t2_target_labels_each_possession_correctly():
    df = _synthetic_spadl_possession_scenarios()
    labels = t2_possession_outcome(df)
    # aid 0 and 1 belong to possession 1 (goal).
    assert labels.loc[labels["action_id"].isin([0, 1]), "possession_outcome"].tolist() == ["goal", "goal"]
    # aid 2, 3 → possession 2 (turnover — because team 1 acts next)
    assert labels.loc[labels["action_id"].isin([2, 3]), "possession_outcome"].tolist() == ["turnover", "turnover"]
    # aid 4, 5 → possession 3 (foul)
    assert labels.loc[labels["action_id"].isin([4, 5]), "possession_outcome"].tolist() == ["foul", "foul"]
    # aid 6, 7 → possession 4 (no_goal)
    assert labels.loc[labels["action_id"].isin([6, 7]), "possession_outcome"].tolist() == ["no_goal", "no_goal"]


def test_t2_target_encodes_outcome_class_indices():
    """Verify string ↔ integer mapping stays consistent."""
    df = _synthetic_spadl_possession_scenarios()
    labels = t2_possession_outcome(df)
    for _, row in labels.iterrows():
        assert row["y_possession_outcome"] == T2_OUTCOME_INDEX[row["possession_outcome"]]
    assert T2_OUTCOME_CLASSES == ("goal", "no_goal", "turnover", "foul")


# --- T2 evaluation end-to-end -----------------------------------


def test_evaluate_t2_returns_one_row_per_method():
    rng = np.random.default_rng(0)
    n_matches = 6
    rows = []
    for gid in range(1, n_matches + 1):
        for aid in range(30):
            rows.append({
                "game_id": gid, "team_id": (aid % 2) + 1,
                "action_id": aid, "value": rng.uniform(-1, 1),
            })
    preds_a = pd.DataFrame(rows)
    preds_b = preds_a.copy()
    preds_b["value"] = preds_b["value"] + rng.standard_normal(len(preds_b)) * 0.1

    targets = preds_a[["game_id", "team_id", "action_id"]].copy()
    targets["y_possession_outcome"] = rng.integers(0, 4, len(targets))

    res = evaluate_t2(
        per_action_by_method={"m_a": preds_a, "m_b": preds_b},
        targets=targets,
        train_game_ids={1, 2, 3, 4},
        test_game_ids={5, 6},
        n_bootstrap=25,
    )
    assert set(res["model_id"]) == {"m_a", "m_b"}
    for col in ("log_loss_multiclass_point", "top1_accuracy_point",
                "ece_multiclass_point"):
        assert col in res.columns


def test_load_predictions_picks_up_both_dirs(tmp_path):
    (tmp_path / "methods").mkdir()
    (tmp_path / "baselines").mkdir()
    df = pd.DataFrame({"game_id": [1], "team_id": [1], "value": [0.5]})
    df.to_parquet(tmp_path / "methods" / "m1__t3_team_match_values.parquet", index=False)
    df.to_parquet(tmp_path / "baselines" / "b1__t3_team_match_values.parquet", index=False)
    preds = load_predictions(tmp_path)
    assert set(preds.keys()) == {"m1", "b1"}
