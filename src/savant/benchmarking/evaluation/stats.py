"""Pairwise significance testing for Track-B.

Config specifies Wilcoxon signed-rank on paired match-level errors with
Benjamini-Hochberg correction across the (methods × methods) grid.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _match_level_errors(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Return DataFrame [game_id, match_error] with per-match absolute-error
    signal. The two team-rows per match are averaged so each match
    contributes ONE row — Wilcoxon signed-rank assumes independent pairs."""
    df = predictions.merge(
        targets[["game_id", "team_id", target_col]],
        on=["game_id", "team_id"], how="inner",
    )
    df["abs_err"] = (df["value"] - df[target_col]).abs()
    return (df.groupby("game_id", as_index=False)["abs_err"]
              .mean()
              .rename(columns={"abs_err": "match_error"}))


def wilcoxon_pairwise(
    predictions_by_method: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    target_col: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank test on paired match-level errors.

    For every ordered pair (A, B) with |methods| distinct methods, tests
    whether A's per-match absolute error differs significantly from B's.
    Returns a DataFrame with columns:
        method_a, method_b, statistic, p_value, p_value_bh, significant

    p_value_bh is the Benjamini-Hochberg-corrected p-value at the given
    alpha across all pairs simultaneously.
    """
    from scipy.stats import wilcoxon

    errors_by_method: dict[str, pd.DataFrame] = {
        m: _match_level_errors(df, targets, target_col)
        for m, df in predictions_by_method.items()
    }

    rows: list[dict] = []
    for a, b in combinations(errors_by_method.keys(), 2):
        # Align on the intersection of matches — every pair is tested on
        # matches that both methods actually predicted on.
        merged = errors_by_method[a].merge(
            errors_by_method[b],
            on="game_id", suffixes=(f"_{a}", f"_{b}"),
        )
        if len(merged) < 5:
            rows.append({"method_a": a, "method_b": b,
                         "statistic": float("nan"), "p_value": float("nan")})
            continue
        try:
            res = wilcoxon(
                merged[f"match_error_{a}"].values,
                merged[f"match_error_{b}"].values,
                zero_method="wilcox",
                alternative="two-sided",
            )
            rows.append({"method_a": a, "method_b": b,
                         "statistic": float(res.statistic),
                         "p_value": float(res.pvalue)})
        except ValueError:
            # zero-difference degeneracy — all matches identical
            rows.append({"method_a": a, "method_b": b,
                         "statistic": float("nan"), "p_value": float("nan")})

    df = pd.DataFrame(rows)
    df["p_value_bh"] = _benjamini_hochberg(df["p_value"].values)
    df["significant"] = df["p_value_bh"] < alpha
    return df


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH-corrected p-values (a.k.a. q-values). NaNs pass through as NaN."""
    p = np.asarray(p_values, dtype=float).copy()
    n = np.sum(~np.isnan(p))
    if n == 0:
        return p
    # BH: sort ascending, multiply by n/rank, ensure monotone non-increasing
    # when scanned from the largest p downwards.
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranks = np.arange(1, len(p) + 1)
    adj = np.full_like(p, np.nan)
    p_sorted = p[order]
    valid = ~np.isnan(p_sorted)
    n_valid = valid.sum()
    q = p_sorted[valid] * n_valid / ranks[:n_valid]
    # Enforce monotonicity from the top.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    adj_sorted = adj.copy()
    adj_sorted[valid] = q
    adj[order] = adj_sorted
    return adj
