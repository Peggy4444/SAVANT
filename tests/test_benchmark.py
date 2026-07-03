"""Benchmark integrity tests (Track B — BM-2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from savant.benchmarking import (
    LeakageError,
    MatchSplit,
    SPADLLoader,
    SplitSpec,
    holdout_competitions,
    leakage_audit,
    spadl_action_types,
    spadl_bodyparts,
    spadl_field_dimensions,
    spadl_results,
    split_matches,
    split_spec,
    train_competitions,
)
from savant.benchmarking.loaders import StatsBombOpenLoader

# statsbombpy hits the network. Tests that resolve competitions run offline
# via the cached competitions.json inside statsbombpy — no download — but
# tests that pull match events do fetch from GitHub. Mark those `slow` so
# CI can skip if unavailable.


# --- Config-driven metadata (offline) --------------------------------


def test_train_and_holdout_competitions_are_in_open_data() -> None:
    """Cheap check: every competition listed in configs/benchmark_tasks.yaml
    resolves against statsbombpy's competitions() catalogue."""
    loader = StatsBombOpenLoader()
    for spec in train_competitions() + holdout_competitions():
        comp = loader.resolve(spec.name, spec.season)
        assert comp.competition_id > 0
        assert comp.season_id > 0
        assert comp.competition_name == spec.name
        assert comp.season_name == spec.season


def test_resolve_raises_on_missing_season() -> None:
    loader = StatsBombOpenLoader()
    with pytest.raises(ValueError, match="not found in StatsBomb-Open"):
        loader.resolve("FIFA World Cup", "1848")


def test_split_spec_from_config() -> None:
    s = split_spec()
    assert s.by == "match"
    assert abs((s.train_fraction + s.val_fraction + s.test_fraction) - 1.0) < 1e-6
    assert s.require_leakage_audit_pass is True


# --- Match-level splitter (pure) ------------------------------------


def test_split_is_deterministic_and_partitions_all_ids() -> None:
    ids = list(range(100))
    s1 = split_matches(ids)
    s2 = split_matches(ids)
    assert s1.train == s2.train
    assert s1.val == s2.val
    assert s1.test == s2.test
    assert set(s1.train) | set(s1.val) | set(s1.test) == set(ids)
    assert set(s1.train) & set(s1.val) == set()
    assert set(s1.train) & set(s1.test) == set()
    assert set(s1.val) & set(s1.test) == set()


def test_split_respects_fractions() -> None:
    spec = SplitSpec(by="match", train_fraction=0.5, val_fraction=0.25,
                     test_fraction=0.25, seed=42,
                     require_leakage_audit_pass=True)
    s = split_matches(list(range(400)), spec=spec)
    assert 195 <= len(s.train) <= 205
    assert 95 <= len(s.val) <= 105
    assert 95 <= len(s.test) <= 105
    assert len(s.train) + len(s.val) + len(s.test) == 400


def test_split_rejects_non_match_level() -> None:
    spec = SplitSpec(by="event", train_fraction=0.8, val_fraction=0.1,
                     test_fraction=0.1, seed=0,
                     require_leakage_audit_pass=True)
    with pytest.raises(ValueError, match="match-level"):
        split_matches([1, 2, 3], spec=spec)


def test_split_rejects_fraction_mismatch() -> None:
    spec = SplitSpec(by="match", train_fraction=0.5, val_fraction=0.5,
                     test_fraction=0.5, seed=0,
                     require_leakage_audit_pass=True)
    with pytest.raises(ValueError, match="sum to 1"):
        split_matches([1, 2, 3], spec=spec)


# --- Leakage audit ---------------------------------------------------


def test_leakage_audit_passes_on_clean_split() -> None:
    split = MatchSplit(train=[1, 2], val=[3], test=[4])
    events = pd.DataFrame({"match_id": [1, 1, 2, 3, 4, 4, 4]})
    counts = leakage_audit(events, split)
    assert counts["n_events_train"] == 3
    assert counts["n_events_val"] == 1
    assert counts["n_events_test"] == 3
    assert counts["n_matches_train"] == 2


def test_leakage_audit_catches_train_test_overlap() -> None:
    split = MatchSplit(train=[1, 2], val=[3], test=[2, 4])  # 2 in both!
    events = pd.DataFrame({"match_id": [1, 2, 3, 4]})
    with pytest.raises(LeakageError, match="train and test"):
        leakage_audit(events, split, strict=True)


def test_leakage_audit_catches_unassigned_events() -> None:
    split = MatchSplit(train=[1], val=[2], test=[3])
    events = pd.DataFrame({"match_id": [1, 2, 3, 99]})  # 99 never assigned
    with pytest.raises(LeakageError, match="do not appear in ANY split"):
        leakage_audit(events, split, strict=True)


def test_leakage_audit_requires_match_id_column() -> None:
    split = MatchSplit(train=[1], val=[], test=[])
    events = pd.DataFrame({"event_id": [1, 2]})
    with pytest.raises(ValueError, match="match_id"):
        leakage_audit(events, split)


# --- Slow: real StatsBomb pull ---------------------------------------

pytestmark_slow = pytest.mark.skipif(
    False,  # flip to True in CI without network
    reason="requires network to reach StatsBomb Open data on GitHub",
)


@pytest.mark.slow
def test_matches_loader_hits_real_open_data(tmp_path: Path) -> None:
    """Live smoke test: FA Women's Super League 2020/2021 has ~87 matches
    on Open Data and NO 360 frames — cheap to verify."""
    loader = StatsBombOpenLoader(cache_root=tmp_path)
    comp = loader.resolve("FA Women's Super League", "2020/2021")
    assert comp.has_360 is False
    matches = loader.list_matches(comp)
    assert len(matches) > 50
    # Cached the second time
    assert loader._matches_cache_path(comp).exists()
    matches2 = loader.list_matches(comp)
    assert list(matches.columns) == list(matches2.columns)


# --- SPADL constants (offline) --------------------------------------


def test_spadl_field_dimensions_are_reasonable() -> None:
    length, width = spadl_field_dimensions()
    # SPADL uses metres — canonical pitch is ~105 × 68
    assert 100 < length < 110
    assert 60 < width < 72


def test_spadl_action_types_include_canonical_labels() -> None:
    types = set(spadl_action_types())
    for t in ("pass", "shot", "dribble", "cross", "tackle"):
        assert t in types, f"canonical SPADL action type {t!r} missing"


def test_spadl_bodyparts_and_results_are_lists() -> None:
    bps = spadl_bodyparts()
    results = spadl_results()
    assert isinstance(bps, list) and len(bps) >= 4
    assert isinstance(results, list) and "success" in results and "fail" in results


# --- SPADL conversion live test (slow, network) ---------------------


@pytest.mark.slow
def test_spadl_conversion_on_one_real_match(tmp_path: Path) -> None:
    """Load ONE FA-WSL match via socceraction, convert to SPADL, verify shape.
    Uses socceraction's own loader (which pulls from statsbomb/open-data on
    GitHub), not statsbombpy — the two are separate data paths."""
    loader = SPADLLoader(cache_root=tmp_path)
    comp = loader.resolve("FA Women's Super League", "2020/2021")
    matches = loader.list_matches(comp)
    assert len(matches) > 50
    game_id = int(matches.iloc[0]["game_id"])

    actions = loader.load_actions(comp, game_id)

    # SPADL schema — verify every canonical column is present
    required = {
        "game_id", "original_event_id", "period_id", "time_seconds",
        "team_id", "player_id",
        "start_x", "start_y", "end_x", "end_y",
        "type_id", "result_id", "bodypart_id", "action_id",
    }
    assert required.issubset(set(actions.columns)), (
        f"SPADL columns missing: {required - set(actions.columns)}"
    )

    # Bounded action count (a full match yields ~1500–2500 SPADL actions)
    assert 800 < len(actions) < 3500

    # Positions within the SPADL pitch (105 × 68 m)
    assert (actions["start_x"] >= 0).all() and (actions["start_x"] <= 105).all()
    assert (actions["start_y"] >= 0).all() and (actions["start_y"] <= 68).all()

    # Cache round-trip
    cache_path = loader._spadl_cache_path(comp, game_id)
    assert cache_path.exists()
    actions_from_cache = loader.load_actions(comp, game_id)
    assert list(actions.columns) == list(actions_from_cache.columns)
    assert len(actions) == len(actions_from_cache)


# --- SB-360 freeze-frames live test (slow, network) ----------------


@pytest.mark.slow
def test_360_freeze_frames_load_for_recent_tournament(tmp_path: Path) -> None:
    """UEFA Euro 2024 has 360 available. Load frames for one match and
    verify structural shape."""
    loader = StatsBombOpenLoader(cache_root=tmp_path)
    comp = loader.resolve("UEFA Euro", "2024")
    assert comp.has_360 is True
    matches = loader.list_matches(comp)
    game_id = int(matches.iloc[0]["match_id"])

    frames = loader.load_360(comp, game_id)
    assert frames is not None, "Expected 360 frames for a 360-enabled competition"
    assert len(frames) > 500  # a full match has thousands of freeze-frame rows
    # Freeze frames should carry the event_uuid so we can join back to events.
    assert "event_uuid" in frames.columns or "match_id" in frames.columns
    # Cache round-trip
    frames_cached = loader.load_360(comp, game_id)
    assert len(frames) == len(frames_cached)


@pytest.mark.slow
def test_360_returns_none_when_not_available(tmp_path: Path) -> None:
    """FA Women's Super League 2020/21 has NO 360 data → should return None."""
    loader = StatsBombOpenLoader(cache_root=tmp_path)
    comp = loader.resolve("FA Women's Super League", "2020/2021")
    assert comp.has_360 is False
    matches = loader.list_matches(comp)
    game_id = int(matches.iloc[0]["match_id"])
    frames = loader.load_360(comp, game_id)
    assert frames is None
