"""Match-level bootstrap CIs.

Track-B's `bootstrap.unit = match` means each resample draws MATCHES with
replacement (not per-row draws). This prevents CIs from being deflated by
the inherent correlation between a match's two team-rows: keeping matches
as the atomic unit preserves that dependence, and averaging over 1000
resamples gives an honest 95% CI.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _sample_matches(
    match_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a bootstrap sample of match ids with replacement."""
    return rng.choice(match_ids, size=len(match_ids), replace=True)


def bootstrap_ci_by_match(
    predictions: pd.DataFrame,          # cols include: game_id, value + `merge_on`
    targets: pd.DataFrame,              # cols include: game_id, <target_col> + `merge_on`
    target_col: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    merge_on: tuple[str, ...] = ("game_id", "team_id"),
) -> dict:
    """Bootstrap CI for a metric applied to (predictions, targets).

    Returns a dict with:
        - point:       metric on the full observed data
        - lo, hi:      lower / upper CI bounds
        - resamples:   np.ndarray of the n bootstrap statistics
    Resamples draw MATCHES with replacement — every row from a sampled
    match is kept, and metric is recomputed on the concatenated rows.

    For per-action T1 targets pass `merge_on=("game_id","team_id","action_id")`
    so N actions per team-match don't cartesian-product against N labels.
    """
    keep = list(merge_on) + [target_col]
    df = predictions.merge(
        targets[keep],
        on=list(merge_on), how="inner",
    )
    if len(df) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "resamples": np.array([])}

    point = metric_fn(df["value"].values, df[target_col].values)

    match_ids = df["game_id"].unique()
    if len(match_ids) < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "resamples": np.array([])}

    # Pre-group by match for fast resample assembly.
    grouped = {mid: df[df["game_id"] == mid] for mid in match_ids}

    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled_ids = _sample_matches(match_ids, rng)
        # Concatenate the sampled match blocks. Duplicate matches show up
        # more than once, which is the point of the bootstrap.
        parts = [grouped[m] for m in sampled_ids]
        resample = pd.concat(parts, ignore_index=True)
        stats[i] = metric_fn(resample["value"].values, resample[target_col].values)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(stats, [100 * alpha, 100 * (1.0 - alpha)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "resamples": stats}
