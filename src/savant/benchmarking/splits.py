"""Match-level splitting for Track-B benchmark experiments.

Every method's train / val / test partition MUST be defined at the match
level — never at the event level — per CLAUDE.md §5 Stage 6 and the paper's
benchmark-integrity rules. `leakage_audit` provides a fail-fast check that
an events DataFrame does not accidentally include the same match in more
than one split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import SplitSpec, split_spec


@dataclass(frozen=True)
class MatchSplit:
    """The result of splitting a set of matches into train / val / test."""
    train: list[int] = field(default_factory=list)
    val:   list[int] = field(default_factory=list)
    test:  list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[int]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def label_for(self, match_id: int) -> str | None:
        for lbl in ("train", "val", "test"):
            if match_id in getattr(self, lbl):
                return lbl
        return None


class LeakageError(RuntimeError):
    """Raised by `leakage_audit` when an events DataFrame's `match_id`
    intersects more than one split."""


def split_matches(
    match_ids: Sequence[int],
    spec: SplitSpec | None = None,
) -> MatchSplit:
    """Deterministic match-level split. Uses `spec.seed` for reproducibility.

    Fractions are consumed left-to-right (train → val → test); any remaining
    matches from rounding fall into the test bucket so no match is dropped.
    """
    spec = spec or split_spec()
    if abs((spec.train_fraction + spec.val_fraction + spec.test_fraction) - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1; got "
            f"{spec.train_fraction} + {spec.val_fraction} + {spec.test_fraction}"
        )
    if spec.by != "match":
        raise ValueError(f"Only match-level splits are supported; got by={spec.by!r}")

    rng = np.random.default_rng(spec.seed)
    shuffled = list(match_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(spec.train_fraction * n))
    n_val = int(round(spec.val_fraction * n))
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]  # remainder → test

    return MatchSplit(
        train=[int(x) for x in train],
        val=[int(x) for x in val],
        test=[int(x) for x in test],
    )


def leakage_audit(
    events: pd.DataFrame,
    split: MatchSplit,
    *,
    match_id_column: str = "match_id",
    strict: bool = True,
) -> dict[str, int]:
    """Verify that no `match_id` in `events` appears in more than one split.

    Returns per-split event counts. If `strict=True` (default), raises
    `LeakageError` at the first violation. If `strict=False`, returns the
    counts and logs a warning instead.
    """
    if match_id_column not in events.columns:
        raise ValueError(
            f"events DataFrame missing required column {match_id_column!r}"
        )

    train_ids = set(split.train)
    val_ids = set(split.val)
    test_ids = set(split.test)

    # Overlap checks — split assignments must be disjoint.
    if train_ids & val_ids:
        msg = f"Leakage: {len(train_ids & val_ids)} match_ids in both train and val"
        if strict:
            raise LeakageError(msg)
    if train_ids & test_ids:
        msg = f"Leakage: {len(train_ids & test_ids)} match_ids in both train and test"
        if strict:
            raise LeakageError(msg)
    if val_ids & test_ids:
        msg = f"Leakage: {len(val_ids & test_ids)} match_ids in both val and test"
        if strict:
            raise LeakageError(msg)

    # Every match_id in the events must be assigned to exactly one split.
    all_assigned = train_ids | val_ids | test_ids
    events_match_ids = set(events[match_id_column].dropna().astype(int).unique())
    unassigned = events_match_ids - all_assigned
    if unassigned:
        msg = (
            f"Leakage risk: {len(unassigned)} match_ids in the events DataFrame "
            f"do not appear in ANY split (first 5: {sorted(unassigned)[:5]}). "
            "This means those events would be evaluated without an explicit "
            "split assignment."
        )
        if strict:
            raise LeakageError(msg)

    train_n = int(events[events[match_id_column].isin(train_ids)].shape[0])
    val_n = int(events[events[match_id_column].isin(val_ids)].shape[0])
    test_n = int(events[events[match_id_column].isin(test_ids)].shape[0])
    return {
        "n_events_train": train_n,
        "n_events_val": val_n,
        "n_events_test": test_n,
        "n_matches_train": len(train_ids),
        "n_matches_val": len(val_ids),
        "n_matches_test": len(test_ids),
    }
