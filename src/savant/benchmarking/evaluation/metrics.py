"""Scalar metrics for Track-B evaluation.

Every function takes two 1-D numpy arrays of the same length: predictions
and ground-truth. NaNs in either side are dropped pairwise before scoring.
"""

from __future__ import annotations

import numpy as np


def _clean(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise drop any (pred, target) row with a NaN or non-finite value."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    mask = np.isfinite(pred) & np.isfinite(target)
    return pred[mask], target[mask]


def pearson_rho(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _clean(pred, target)
    if len(p) < 2:
        return float("nan")
    # Guard against zero-variance predictions — Pearson is undefined then.
    if np.std(p) == 0 or np.std(t) == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def spearman_rho(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman's ρ via pairwise Pearson on ranks — avoids the scipy import
    on the hot path but matches scipy.stats.spearmanr on numerical ties."""
    p, t = _clean(pred, target)
    if len(p) < 2:
        return float("nan")
    # `argsort().argsort()` produces the rank of each element; ties get
    # broken by original order (same as scipy's default 'average' up to
    # a small offset that doesn't affect Pearson on ranks).
    pr = np.argsort(np.argsort(p))
    tr = np.argsort(np.argsort(t))
    return pearson_rho(pr, tr)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _clean(pred, target)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


# --- Binary probability metrics (T1) --------------------------------


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    """Brier score = mean((p − y)²).  Lower is better.
    `pred` is a probability in [0, 1]; `target` is a 0/1 label."""
    p, t = _clean(pred, target)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - t) ** 2))


def log_loss_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """Binary log loss (aka cross-entropy). Predictions clipped to
    [1e-12, 1 − 1e-12] to avoid log(0). Lower is better."""
    p, t = _clean(pred, target)
    if len(p) == 0:
        return float("nan")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def ece(pred: np.ndarray, target: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over `n_bins` equal-width bins.
    Reports the sample-weighted mean of |accuracy_bin − avg_confidence_bin|."""
    p, t = _clean(pred, target)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    n_total = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Inclusive on the top edge for the final bin so p = 1.0 is counted.
        in_bin = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        n_in = in_bin.sum()
        if n_in == 0:
            continue
        acc_bin = t[in_bin].mean()
        conf_bin = p[in_bin].mean()
        ece_sum += (n_in / n_total) * abs(acc_bin - conf_bin)
    return float(ece_sum)


def roc_auc(pred: np.ndarray, target: np.ndarray) -> float:
    """Standard binary AUC.  Delegates to sklearn to avoid re-deriving the
    Mann-Whitney formulation. Returns NaN when only one class is present."""
    p, t = _clean(pred, target)
    if len(p) < 2 or len(np.unique(t)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(t, p))


# --- Multiclass metrics (T2) ---------------------------------------


def _clean_multiclass(
    pred: np.ndarray, target: np.ndarray, n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`pred` shape (n, n_classes) with row probabilities; `target` shape (n,)
    with int class labels. Drops rows where target is NaN or out of range."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target)
    if pred.ndim != 2 or pred.shape[1] != n_classes:
        raise ValueError(
            f"pred must be (n, {n_classes}); got {pred.shape}",
        )
    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            f"pred rows {pred.shape[0]} vs target rows {target.shape[0]}",
        )
    mask = (
        (target >= 0) & (target < n_classes)
        & np.isfinite(pred).all(axis=1)
    )
    return pred[mask], target[mask].astype(int)


def log_loss_multiclass(
    pred: np.ndarray, target: np.ndarray, n_classes: int = 4,
) -> float:
    """Cross-entropy over rows. `pred` clipped to [1e-12, 1] and re-normalised
    row-wise so log(0) doesn't fire even on adversarial inputs."""
    p, t = _clean_multiclass(pred, target, n_classes)
    if len(p) == 0:
        return float("nan")
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(t)), t])))


def top1_accuracy(
    pred: np.ndarray, target: np.ndarray, n_classes: int = 4,
) -> float:
    p, t = _clean_multiclass(pred, target, n_classes)
    if len(p) == 0:
        return float("nan")
    return float((p.argmax(axis=1) == t).mean())


def ece_multiclass(
    pred: np.ndarray, target: np.ndarray, n_classes: int = 4, n_bins: int = 15,
) -> float:
    """Max-probability ECE: bin by top-class confidence, compare to top-1
    accuracy in each bin. This is the standard reliability-diagram flavour
    used by Guo et al. (2017)."""
    p, t = _clean_multiclass(pred, target, n_classes)
    if len(p) == 0:
        return float("nan")
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == t).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    n_total = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= lo) & (
            confidence <= hi if hi == 1.0 else confidence < hi
        )
        n_in = in_bin.sum()
        if n_in == 0:
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = confidence[in_bin].mean()
        ece_sum += (n_in / n_total) * abs(acc_bin - conf_bin)
    return float(ece_sum)


# --- Metric registry -----------------------------------------------
METRIC_FUNCS = {
    # Continuous — T3
    "pearson_rho":     pearson_rho,
    "spearman_rho":    spearman_rho,
    "mae":             mae,
    # Binary probability — T1
    "brier":           brier,
    "log_loss_binary": log_loss_binary,
    "ece":             ece,
    "roc_auc":         roc_auc,
    # Multiclass — T2 (require matrix-shaped inputs — not routed through score())
    "log_loss_multiclass": log_loss_multiclass,
    "top1_accuracy":       top1_accuracy,
    "ece_multiclass":      ece_multiclass,
}


def score(name: str, pred: np.ndarray, target: np.ndarray) -> float:
    """Dispatch metric by name — for use in bootstrap_ci / dynamic setups."""
    if name not in METRIC_FUNCS:
        raise KeyError(f"unknown metric {name!r}; known: {list(METRIC_FUNCS)}")
    return METRIC_FUNCS[name](pred, target)
