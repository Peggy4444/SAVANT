"""StatsBomb-Open event + 360 freeze-frame loader.

Wraps `statsbombpy` with:
- name/season → competition_id/season_id resolution
- per-match caching to `data/statsbomb/{competition_id}/{season_id}/`
- freeze-frame loader for SB-360 (when available)
- batch loader that returns one long DataFrame across a whole competition-season

Public API:
    loader = StatsBombOpenLoader()
    comp_id, season_id, has_360 = loader.resolve("FIFA World Cup", "2022")
    matches = loader.list_matches(comp_id, season_id)
    events = loader.load_events_batch(matches["match_id"].tolist())
    frames = loader.load_360_batch(matches["match_id"].tolist())    # if has_360

Or the module-level convenience:
    events_df, matches_df = load_events_for_competitions([("FIFA World Cup","2022")])
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import cache_dir as _default_cache_dir

# Silence statsbombpy's "no credentials" warning at import time — the open
# data path does not need credentials by design.
warnings.filterwarnings(
    "ignore",
    message="credentials were not supplied",
    category=UserWarning,
    module="statsbombpy.api_client",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Competition:
    """Resolved competition-season identifier."""
    competition_id: int
    season_id: int
    competition_name: str
    season_name: str
    has_360: bool


class StatsBombOpenLoader:
    """Cached wrapper around `statsbombpy.sb`. All heavy calls memoise to disk."""

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = cache_root or _default_cache_dir()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # --- Competition lookup --------------------------------------------

    def _load_competitions_df(self) -> pd.DataFrame:
        from statsbombpy import sb
        return sb.competitions()

    def resolve(self, name: str, season: str) -> Competition:
        """(name, season) → resolved `Competition`. Raises if not found."""
        comps = self._load_competitions_df()
        m = comps[
            (comps["competition_name"] == name)
            & (comps["season_name"] == season)
        ]
        if len(m) == 0:
            candidates = comps[comps["competition_name"] == name][
                "season_name"
            ].tolist()
            raise ValueError(
                f"Competition {name!r} / season {season!r} not found in "
                f"StatsBomb-Open. Available seasons for that competition: "
                f"{candidates or '(none)'}"
            )
        row = m.iloc[0]
        return Competition(
            competition_id=int(row["competition_id"]),
            season_id=int(row["season_id"]),
            competition_name=name,
            season_name=season,
            has_360=bool(row.get("match_available_360")),
        )

    def resolve_many(self, pairs: Iterable[tuple[str, str]]) -> list[Competition]:
        return [self.resolve(name, season) for name, season in pairs]

    # --- Match listing --------------------------------------------------

    def _matches_cache_path(self, comp: Competition) -> Path:
        return (self.cache_root
                / str(comp.competition_id) / str(comp.season_id)
                / "matches.parquet")

    def list_matches(self, comp: Competition, use_cache: bool = True) -> pd.DataFrame:
        path = self._matches_cache_path(comp)
        if use_cache and path.exists():
            return pd.read_parquet(path)
        from statsbombpy import sb
        df = sb.matches(competition_id=comp.competition_id,
                        season_id=comp.season_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df

    # --- Events (per match, cached) ------------------------------------

    def _events_cache_path(self, comp: Competition, match_id: int) -> Path:
        return (self.cache_root
                / str(comp.competition_id) / str(comp.season_id)
                / "events" / f"{match_id}.parquet")

    def load_events(
        self, comp: Competition, match_id: int, use_cache: bool = True
    ) -> pd.DataFrame:
        """Raw StatsBomb events for one match (cached)."""
        path = self._events_cache_path(comp, match_id)
        if use_cache and path.exists():
            return pd.read_parquet(path)
        from statsbombpy import sb
        df = sb.events(match_id=match_id)
        # statsbombpy returns object-heavy dtypes; coerce columns that would
        # break parquet round-trip.
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df

    def load_events_batch(
        self,
        comp: Competition,
        match_ids: Iterable[int],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Concatenated events across many matches; `match_id` column preserved."""
        parts = []
        for mid in match_ids:
            df = self.load_events(comp, int(mid), use_cache=use_cache)
            if "match_id" not in df.columns:
                df = df.assign(match_id=int(mid))
            parts.append(df)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    # --- 360 freeze frames (per match, cached) -------------------------

    def _360_cache_path(self, comp: Competition, match_id: int) -> Path:
        return (self.cache_root
                / str(comp.competition_id) / str(comp.season_id)
                / "three_sixty" / f"{match_id}.parquet")

    def load_360(
        self,
        comp: Competition,
        match_id: int,
        use_cache: bool = True,
    ) -> pd.DataFrame | None:
        """Freeze-frame data for one match. Returns None if unavailable."""
        if not comp.has_360:
            return None
        path = self._360_cache_path(comp, match_id)
        if use_cache and path.exists():
            return pd.read_parquet(path)
        from statsbombpy import sb
        try:
            df = sb.frames(match_id=match_id)
        except Exception as exc:
            logger.warning("360 frames unavailable for match %d: %s", match_id, exc)
            return None
        if df is None or len(df) == 0:
            return None
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df

    def load_360_batch(
        self,
        comp: Competition,
        match_ids: Iterable[int],
        use_cache: bool = True,
    ) -> pd.DataFrame | None:
        if not comp.has_360:
            return None
        parts = []
        for mid in match_ids:
            df = self.load_360(comp, int(mid), use_cache=use_cache)
            if df is not None and len(df):
                if "match_id" not in df.columns:
                    df = df.assign(match_id=int(mid))
                parts.append(df)
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)


# --- Module-level convenience -----------------------------------------


def resolve_competition(name: str, season: str) -> Competition:
    """One-shot resolver, uses the default cache dir."""
    return StatsBombOpenLoader().resolve(name, season)


def load_events_for_competitions(
    pairs: Iterable[tuple[str, str]],
    include_360: bool = False,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: return `(events_df, matches_df)` across one or more
    competition-seasons. Adds a `competition_name` + `season_name` column
    to both frames so downstream splits are competition-aware."""
    loader = StatsBombOpenLoader()
    events_parts, matches_parts = [], []
    for name, season in pairs:
        comp = loader.resolve(name, season)
        matches = loader.list_matches(comp, use_cache=use_cache)
        matches = matches.assign(
            competition_name=comp.competition_name,
            season_name=comp.season_name,
            has_360=comp.has_360,
        )
        matches_parts.append(matches)
        events = loader.load_events_batch(
            comp, matches["match_id"].tolist(), use_cache=use_cache
        )
        events = events.assign(
            competition_name=comp.competition_name,
            season_name=comp.season_name,
        )
        events_parts.append(events)
    events_df = pd.concat(events_parts, ignore_index=True) if events_parts else pd.DataFrame()
    matches_df = pd.concat(matches_parts, ignore_index=True) if matches_parts else pd.DataFrame()
    return events_df, matches_df
