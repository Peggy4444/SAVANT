"""Benchmark integrity tests.

TODO (Stage 6):
- [ ] Test that match-level splits leak zero events between train/test.
- [ ] Test xT grid runs end-to-end on a 1-match StatsBomb sample.
- [ ] Test VAEP / Atomic-VAEP train+predict on a 1-match sample.
- [ ] Test that bootstrap CIs widen monotonically as `n` shrinks.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Stage 6 not implemented yet.")
def test_match_level_split_has_no_event_leakage() -> None:
    raise NotImplementedError
