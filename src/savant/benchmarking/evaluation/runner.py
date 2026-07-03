"""Track-B evaluation runner.

Reads a benchmark output directory (per-method / per-baseline parquets +
raw StatsBomb events) and produces:

    - Point estimates + 95% bootstrap CIs for T3 (Pearson ρ, Spearman ρ, MAE)
      per method against team match-xG differential.
    - Pairwise Wilcoxon signed-rank on per-match errors with BH correction.
    - Δ metrics for T4 (train-vs-holdout) when a holdout output dir is given.

Writes leaderboard.csv + per_task/ artefacts.

T1 (per-action Brier) and T2 (multiclass possession outcome) are NOT in
this runner — they require per-action prediction outputs that the current
wrappers don't emit. See BM-10.5.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_ci_by_match
from .metrics import score
from .per_competition import evaluate_per_competition, robustness_stats
from .stats import wilcoxon_pairwise
from .targets import (
    T2_OUTCOME_CLASSES, t1_goal_next_k_actions, t2_possession_outcome,
    team_match_xg_diff,
)

logger = logging.getLogger(__name__)


# --- Predictions loader ------------------------------------------------


def load_predictions(bench_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every {baseline,method}/*.parquet under a benchmark run dir.

    Returns dict of id → DataFrame with columns [game_id, team_id, value].
    Method IDs and baseline IDs share one flat dict — the leaderboard
    doesn't distinguish them (they both compete on T3).
    """
    preds: dict[str, pd.DataFrame] = {}
    for sub in ("baselines", "methods"):
        for p in sorted((bench_dir / sub).glob("*__t3_team_match_values.parquet")):
            model_id = p.name.replace("__t3_team_match_values.parquet", "")
            preds[model_id] = pd.read_parquet(p)
    return preds


def load_per_action_predictions(bench_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every methods_per_action/*.parquet under a benchmark run dir.
    Returns dict of method_id → DataFrame with cols
    [game_id, team_id, action_id, value]."""
    preds: dict[str, pd.DataFrame] = {}
    d = bench_dir / "methods_per_action"
    if not d.exists():
        return preds
    for p in sorted(d.glob("*__per_action_values.parquet")):
        model_id = p.name.replace("__per_action_values.parquet", "")
        preds[model_id] = pd.read_parquet(p)
    return preds


# --- T3 evaluation -----------------------------------------------------


T3_METRICS = ("pearson_rho", "spearman_rho", "mae")


def evaluate_t3(
    predictions_by_method: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Return one row per method with T3 point + bootstrap CIs.

    Ground truth: team_match_xg_diff (StatsBomb-native xG differential per
    (game, team)). Merge is on the (game_id, team_id) integer keys — any
    predictions using team names will drop out.
    """
    target = team_match_xg_diff(events)

    rows: list[dict] = []
    for model_id, preds in predictions_by_method.items():
        # Sanity: predictions must carry integer team ids to merge with
        # StatsBomb events. String ids silently produce empty joins.
        if preds["team_id"].dtype == object:
            logger.warning(
                "T3: %s has string team ids — coercing to int if numeric, "
                "else skipping merge with statsbomb team_id.", model_id,
            )

        row = {"model_id": model_id}
        for metric_name in T3_METRICS:
            def metric_fn(p, t, _mn=metric_name):
                return score(_mn, p, t)
            r = bootstrap_ci_by_match(
                predictions=preds,
                targets=target,
                target_col="xg_diff",
                metric_fn=metric_fn,
                n_resamples=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            row[f"{metric_name}_point"] = r["point"]
            row[f"{metric_name}_lo"]    = r["lo"]
            row[f"{metric_name}_hi"]    = r["hi"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "pearson_rho_point", ascending=False,
    ).reset_index(drop=True)


# --- T1 evaluation -----------------------------------------------------


T1_METRICS = ("brier", "log_loss_binary", "ece", "roc_auc")


def _isotonic_calibrate(
    raw_train: np.ndarray, y_train: np.ndarray, raw_test: np.ndarray,
) -> np.ndarray:
    """Fit an isotonic regression on (raw_V, y) from train, return calibrated
    probabilities in [0, 1] for `raw_test`. Rows with NaN in either raw or y
    are dropped before fitting."""
    from sklearn.isotonic import IsotonicRegression
    mask = np.isfinite(raw_train) & np.isfinite(y_train)
    if mask.sum() < 10 or len(np.unique(y_train[mask])) < 2:
        # Not enough signal to calibrate — return the sigmoid of standardised
        # raw values as a bland fallback, so downstream metrics still fire.
        z = raw_test.astype(float)
        z_mean, z_std = np.nanmean(z), np.nanstd(z)
        if z_std == 0 or not np.isfinite(z_std):
            return np.full_like(z, 0.5, dtype=float)
        return 1.0 / (1.0 + np.exp(-(z - z_mean) / z_std))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_train[mask], y_train[mask])
    calibrated = iso.predict(np.where(np.isfinite(raw_test), raw_test, 0.0))
    return np.clip(calibrated, 0.0, 1.0)


def evaluate_t1(
    per_action_by_method: dict[str, pd.DataFrame],
    targets: pd.DataFrame,               # cols: game_id, team_id, action_id, y_goal_next_k
    train_game_ids: set[int],
    test_game_ids: set[int],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per method with T1 point estimates + bootstrap CIs.
    Calibration: isotonic regression fit on train, evaluated on test."""
    rows: list[dict] = []
    for model_id, preds in per_action_by_method.items():
        df = preds.merge(targets, on=["game_id", "team_id", "action_id"], how="inner")
        if len(df) == 0:
            rows.append({"model_id": model_id, **{f"{m}_point": float("nan") for m in T1_METRICS}})
            continue

        train_mask = df["game_id"].isin(train_game_ids).values
        test_mask  = df["game_id"].isin(test_game_ids).values
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning(
                "T1: method %s has 0 rows in train or test slice — skipping.",
                model_id,
            )
            rows.append({"model_id": model_id, **{f"{m}_point": float("nan") for m in T1_METRICS}})
            continue

        p_calibrated = _isotonic_calibrate(
            raw_train=df["value"].values[train_mask],
            y_train=df["y_goal_next_k"].values[train_mask],
            raw_test=df["value"].values[test_mask],
        )
        # Build a test-only frame for bootstrap-by-match.
        test_df = pd.DataFrame({
            "game_id":       df["game_id"].values[test_mask],
            "team_id":       df["team_id"].values[test_mask],
            "action_id":     df["action_id"].values[test_mask],
            "value":         p_calibrated,
            "y_goal_next_k": df["y_goal_next_k"].values[test_mask],
        })
        # Point + CI per metric.
        row = {"model_id": model_id}
        for metric_name in T1_METRICS:
            def metric_fn(p, t, _mn=metric_name):
                return score(_mn, p, t)
            r = bootstrap_ci_by_match(
                predictions=test_df[["game_id", "team_id", "action_id", "value"]],
                targets=test_df[["game_id", "team_id", "action_id", "y_goal_next_k"]],
                target_col="y_goal_next_k",
                merge_on=("game_id", "team_id", "action_id"),
                metric_fn=metric_fn,
                n_resamples=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            row[f"{metric_name}_point"] = r["point"]
            row[f"{metric_name}_lo"]    = r["lo"]
            row[f"{metric_name}_hi"]    = r["hi"]
        rows.append(row)

    lb = pd.DataFrame(rows)
    # Primary rank by Brier ascending (lower = better).
    return lb.sort_values("brier_point").reset_index(drop=True)


# --- T2 evaluation -----------------------------------------------------


T2_METRICS = ("log_loss_multiclass", "top1_accuracy", "ece_multiclass")


def _multiclass_calibrator(
    raw_train: np.ndarray, y_train: np.ndarray, raw_test: np.ndarray,
    n_classes: int = 4,
) -> np.ndarray:
    """Fit a multinomial LogisticRegression V → 4-way P on train, return
    P for test. If train is degenerate (single class or all NaN) we fall
    back to the empirical class prior."""
    from sklearn.linear_model import LogisticRegression
    mask = np.isfinite(raw_train) & (y_train >= 0) & (y_train < n_classes)
    if mask.sum() < 20 or len(np.unique(y_train[mask])) < 2:
        # Empirical prior over ALL train rows (including out-of-range) is safe
        # because the class-index domain is bounded.
        prior = np.array([
            (y_train == c).sum() for c in range(n_classes)
        ], dtype=float)
        prior = prior / max(prior.sum(), 1.0)
        return np.tile(prior, (len(raw_test), 1))

    X = raw_train[mask].reshape(-1, 1)
    y = y_train[mask]
    # `multi_class` kwarg removed from sklearn ≥ 1.8 — the default is now
    # always multinomial with lbfgs, matching our previous explicit setting.
    lr = LogisticRegression(
        solver="lbfgs", max_iter=200, class_weight="balanced",
    ).fit(X, y)
    # Align predict-columns with T2_OUTCOME_CLASSES ordering: LogReg outputs
    # in sorted-class order, which matches range(n_classes).
    raw_test_2d = np.where(np.isfinite(raw_test), raw_test, 0.0).reshape(-1, 1)
    proba = lr.predict_proba(raw_test_2d)
    # If some class was absent from training, pad with a zero column.
    full = np.zeros((len(raw_test), n_classes))
    for j, c in enumerate(lr.classes_):
        full[:, int(c)] = proba[:, j]
    # Re-normalise rows so any padding doesn't inflate row sums.
    row_sums = full.sum(axis=1, keepdims=True)
    full = np.where(row_sums > 0, full / row_sums, 1.0 / n_classes)
    return full


def _t2_metric_on(
    df: pd.DataFrame,
    metric_name: str,
    n_classes: int,
) -> float:
    """Recompute a T2 metric on an arbitrary (bootstrap-sampled) DataFrame."""
    from .metrics import ece_multiclass, log_loss_multiclass, top1_accuracy
    p_cols = [f"p{c}" for c in range(n_classes)]
    P = df[p_cols].values
    y = df["y_possession_outcome"].values
    if metric_name == "log_loss_multiclass":
        return log_loss_multiclass(P, y, n_classes)
    if metric_name == "top1_accuracy":
        return top1_accuracy(P, y, n_classes)
    if metric_name == "ece_multiclass":
        return ece_multiclass(P, y, n_classes)
    raise KeyError(f"unknown T2 metric {metric_name!r}")


def _bootstrap_t2(
    test_df: pd.DataFrame,
    metric_name: str,
    n_classes: int = 4,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Match-level bootstrap for a T2 multiclass metric."""
    point = _t2_metric_on(test_df, metric_name, n_classes)
    match_ids = test_df["game_id"].unique()
    if len(match_ids) < 2:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan")}

    grouped = {mid: test_df[test_df["game_id"] == mid] for mid in match_ids}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = rng.choice(match_ids, size=len(match_ids), replace=True)
        parts = [grouped[m] for m in sampled]
        resample = pd.concat(parts, ignore_index=True)
        stats[i] = _t2_metric_on(resample, metric_name, n_classes)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(stats, [100 * alpha, 100 * (1.0 - alpha)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def evaluate_t2(
    per_action_by_method: dict[str, pd.DataFrame],
    targets: pd.DataFrame,           # game_id, team_id, action_id, y_possession_outcome
    train_game_ids: set[int],
    test_game_ids: set[int],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per method with T2 point + bootstrap CIs.

    Calibration: multinomial LogisticRegression V(a) → P over
    {goal, no_goal, turnover, foul}, fit on train, evaluated on test.
    """
    n_classes = len(T2_OUTCOME_CLASSES)
    rows: list[dict] = []
    for model_id, preds in per_action_by_method.items():
        df = preds.merge(
            targets[["game_id", "team_id", "action_id", "y_possession_outcome"]],
            on=["game_id", "team_id", "action_id"], how="inner",
        )
        if len(df) == 0:
            rows.append({"model_id": model_id,
                         **{f"{m}_point": float("nan") for m in T2_METRICS}})
            continue

        train_mask = df["game_id"].isin(train_game_ids).values
        test_mask  = df["game_id"].isin(test_game_ids).values
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning("T2: method %s empty in train or test — skipping.", model_id)
            rows.append({"model_id": model_id,
                         **{f"{m}_point": float("nan") for m in T2_METRICS}})
            continue

        P_test = _multiclass_calibrator(
            raw_train=df["value"].values[train_mask],
            y_train=df["y_possession_outcome"].values[train_mask],
            raw_test=df["value"].values[test_mask],
            n_classes=n_classes,
        )
        test_df = pd.DataFrame({
            "game_id":              df["game_id"].values[test_mask],
            "action_id":            df["action_id"].values[test_mask],
            "y_possession_outcome": df["y_possession_outcome"].values[test_mask],
        })
        for c in range(n_classes):
            test_df[f"p{c}"] = P_test[:, c]

        row = {"model_id": model_id}
        for metric_name in T2_METRICS:
            r = _bootstrap_t2(
                test_df=test_df,
                metric_name=metric_name,
                n_classes=n_classes,
                n_resamples=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            row[f"{metric_name}_point"] = r["point"]
            row[f"{metric_name}_lo"]    = r["lo"]
            row[f"{metric_name}_hi"]    = r["hi"]
        rows.append(row)

    lb = pd.DataFrame(rows)
    return lb.sort_values("log_loss_multiclass_point").reset_index(drop=True)


# --- T4 evaluation -----------------------------------------------------


def evaluate_t4(
    t3_train: pd.DataFrame,
    t3_holdout: pd.DataFrame,
) -> pd.DataFrame:
    """Δ = metric_train − metric_holdout per method. `closer to zero is better`."""
    merged = t3_train.merge(t3_holdout, on="model_id", suffixes=("_train", "_holdout"))
    for m in T3_METRICS:
        merged[f"{m}_delta"] = merged[f"{m}_point_train"] - merged[f"{m}_point_holdout"]
    out_cols = ["model_id"] + [f"{m}_delta" for m in T3_METRICS]
    return merged[out_cols].sort_values("pearson_rho_delta", key=np.abs).reset_index(drop=True)


# --- End-to-end -------------------------------------------------------


def run(
    bench_dir: Path,
    events: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    holdout_bench_dir: Path | None = None,
    holdout_events: pd.DataFrame | None = None,
    train_partial_root: Path | None = None,
    holdout_partial_root: Path | None = None,
) -> dict:
    """One-shot evaluation. Writes CSV artefacts and returns a manifest dict.

    T3 fires whenever `events` is provided. T1 additionally requires the
    SPADL `actions` DataFrame (for `goal_next_k` label construction) and
    the per-action prediction parquets emitted by the benchmark run.
    """
    out_dir = out_dir or bench_dir
    (out_dir / "per_task").mkdir(parents=True, exist_ok=True)

    preds = load_predictions(bench_dir)
    if not preds:
        raise RuntimeError(f"No prediction parquets found under {bench_dir}")

    t3 = evaluate_t3(preds, events, n_bootstrap=n_bootstrap, seed=seed)
    t3_path = out_dir / "per_task" / "t3_pearson_bootstrap.csv"
    t3.to_csv(t3_path, index=False)

    # Pairwise Wilcoxon on T3.
    target = team_match_xg_diff(events)
    wilcoxon_df = wilcoxon_pairwise(preds, target, target_col="xg_diff")
    wilcoxon_path = out_dir / "per_task" / "t3_wilcoxon_pairwise.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)

    # T1: per-action Brier / log-loss / ECE / AUC — optional, requires actions.
    # T2: multiclass possession outcome — shares the same per-action prediction
    #     parquets, calibration head differs.
    t1_path: Path | None = None
    t2_path: Path | None = None
    if actions is not None:
        pa_preds = load_per_action_predictions(bench_dir)
        if pa_preds:
            # Re-derive the same train/test split the benchmark run used so
            # calibration honours the leakage boundary.
            from ..config import split_spec as _split_spec
            from ..splits import split_matches
            spec = _split_spec()
            game_ids = sorted(actions["game_id"].unique().tolist())
            split = split_matches(game_ids, spec=spec)
            train_set = set(split.train)
            test_set  = set(split.test) if split.test else set(split.val)

            t1_targets = t1_goal_next_k_actions(actions, k=10)
            t1 = evaluate_t1(
                per_action_by_method=pa_preds,
                targets=t1_targets,
                train_game_ids=train_set,
                test_game_ids=test_set,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            t1_path = out_dir / "per_task" / "t1_brier_bootstrap.csv"
            t1.to_csv(t1_path, index=False)

            t2_targets = t2_possession_outcome(actions)
            t2 = evaluate_t2(
                per_action_by_method=pa_preds,
                targets=t2_targets,
                train_game_ids=train_set,
                test_game_ids=test_set,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            t2_path = out_dir / "per_task" / "t2_multiclass_bootstrap.csv"
            t2.to_csv(t2_path, index=False)

    # T4 (optional).
    t4_path = None
    if holdout_bench_dir is not None and holdout_events is not None:
        preds_holdout = load_predictions(holdout_bench_dir)
        t3_holdout = evaluate_t3(preds_holdout, holdout_events,
                                  n_bootstrap=n_bootstrap, seed=seed)
        t4 = evaluate_t4(t3, t3_holdout)
        t4_path = out_dir / "per_task" / "t4_delta.csv"
        t4.to_csv(t4_path, index=False)

    # Per-competition breakdown + robustness — optional, requires
    # by-competition partial dirs from run_benchmark_by_competition.py.
    per_comp_path: Path | None = None
    robustness: pd.DataFrame | None = None
    if train_partial_root is not None or holdout_partial_root is not None:
        per_comp = evaluate_per_competition(
            train_partial_root=train_partial_root,
            holdout_partial_root=holdout_partial_root,
        )
        if not per_comp.empty:
            per_comp_path = out_dir / "per_task" / "t3_per_competition.csv"
            per_comp.to_csv(per_comp_path, index=False)
            robustness = robustness_stats(per_comp)

    # Composite leaderboard: rank by T3 Pearson ρ + carry robustness cols
    # when they're available so the paper's headline table is self-contained.
    leaderboard = t3[["model_id", "pearson_rho_point", "pearson_rho_lo",
                      "pearson_rho_hi", "spearman_rho_point", "mae_point"]]
    if robustness is not None and not robustness.empty:
        leaderboard = leaderboard.merge(
            robustness[["model_id", "robustness_mean_rho",
                        "robustness_std_rho", "n_competitions"]],
            on="model_id", how="left",
        )
    leaderboard_path = out_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    manifest = {
        "n_methods": len(preds),
        "n_matches": int(events["match_id"].nunique()
                         if "match_id" in events.columns
                         else events["game_id"].nunique()),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "artefacts": {
            "leaderboard": str(leaderboard_path),
            "t3_pearson_bootstrap": str(t3_path),
            "t3_wilcoxon_pairwise": str(wilcoxon_path),
            "t1_brier_bootstrap": str(t1_path) if t1_path else None,
            "t2_multiclass_bootstrap": str(t2_path) if t2_path else None,
            "t3_per_competition": str(per_comp_path) if per_comp_path else None,
            "t4_delta": str(t4_path) if t4_path else None,
        },
    }
    (out_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
    )
    return manifest
