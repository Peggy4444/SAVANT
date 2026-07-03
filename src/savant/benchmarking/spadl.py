"""StatsBomb → SPADL conversion via socceraction, with per-match caching.

SPADL (Soccer Player Action Description Language) is the canonical action
representation used by VAEP, Atomic-VAEP, and many downstream methods. This
module is the single seam between raw StatsBomb events and every corpus
method that wants SPADL actions.

Design:
- We use `socceraction.data.statsbomb.StatsBombLoader` (NOT statsbombpy) to
  fetch events, because it returns the exact shape socceraction's SPADL
  converter expects (verified in BM-3 investigation).
- Every per-match SPADL DataFrame is cached to `data/statsbomb/{competition_id}/
  {season_id}/spadl/{match_id}.parquet` so subsequent calls are instant.
- Batch helpers return one long DataFrame with `game_id` preserved for
  match-level splits (see `savant.benchmarking.splits`).

Public API:
    loader = SPADLLoader()
    comp = loader.resolve("FA Women's Super League", "2020/2021")
    matches = loader.list_matches(comp)
    actions = loader.load_actions_batch(comp, matches["game_id"].tolist())
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import cache_dir as _default_cache_dir

warnings.filterwarnings(
    "ignore",
    message="credentials were not supplied",
    category=UserWarning,
    module="statsbombpy.api_client",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SPADLCompetition:
    """Resolved StatsBomb-Open competition (socceraction schema)."""
    competition_id: int
    season_id: int
    competition_name: str
    season_name: str


class SPADLLoader:
    """Load StatsBomb-Open matches and convert them to SPADL, cached to disk."""

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = cache_root or _default_cache_dir()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._sb_loader = None  # lazy — the import chain is slow

    # --- Lazy import of socceraction's loader --------------------------

    def _loader(self):
        if self._sb_loader is None:
            from socceraction.data.statsbomb import StatsBombLoader
            self._sb_loader = StatsBombLoader(getter="remote")
        return self._sb_loader

    # --- Competition lookup --------------------------------------------

    def resolve(self, name: str, season: str) -> SPADLCompetition:
        comps = self._loader().competitions()
        m = comps[
            (comps["competition_name"] == name)
            & (comps["season_name"] == season)
        ]
        if len(m) == 0:
            available = comps[comps["competition_name"] == name]["season_name"].tolist()
            raise ValueError(
                f"Competition {name!r} / season {season!r} not found via socceraction. "
                f"Available seasons for that competition: {available or '(none)'}"
            )
        row = m.iloc[0]
        return SPADLCompetition(
            competition_id=int(row["competition_id"]),
            season_id=int(row["season_id"]),
            competition_name=name,
            season_name=season,
        )

    # --- Matches --------------------------------------------------------

    def _matches_cache_path(self, comp: SPADLCompetition) -> Path:
        return (self.cache_root
                / str(comp.competition_id) / str(comp.season_id)
                / "matches_socceraction.parquet")

    def list_matches(self, comp: SPADLCompetition, use_cache: bool = True) -> pd.DataFrame:
        path = self._matches_cache_path(comp)
        if use_cache and path.exists():
            return pd.read_parquet(path)
        df = self._loader().games(
            competition_id=comp.competition_id, season_id=comp.season_id
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df

    # --- SPADL actions (per match, cached) -----------------------------

    def _spadl_cache_path(self, comp: SPADLCompetition, game_id: int) -> Path:
        return (self.cache_root
                / str(comp.competition_id) / str(comp.season_id)
                / "spadl" / f"{game_id}.parquet")

    def load_actions(
        self, comp: SPADLCompetition, game_id: int, use_cache: bool = True
    ) -> pd.DataFrame:
        """Load one match's SPADL actions. Cached to parquet after first call."""
        path = self._spadl_cache_path(comp, game_id)
        if use_cache and path.exists():
            return pd.read_parquet(path)

        from socceraction.spadl import statsbomb as sb_spadl

        loader = self._loader()
        games = loader.games(
            competition_id=comp.competition_id, season_id=comp.season_id
        )
        row = games[games["game_id"] == game_id]
        if len(row) == 0:
            raise ValueError(f"Game {game_id} not in {comp}")
        home_team_id = int(row.iloc[0]["home_team_id"])

        events = loader.events(game_id)
        actions = sb_spadl.convert_to_actions(events, home_team_id)
        # Cast object → str for parquet round-trip.
        for col in actions.select_dtypes(include=["object"]).columns:
            actions[col] = actions[col].astype(str)

        path.parent.mkdir(parents=True, exist_ok=True)
        actions.to_parquet(path, index=False)
        return actions

    def load_actions_batch(
        self,
        comp: SPADLCompetition,
        game_ids: Iterable[int],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Concatenated SPADL actions across many matches. `game_id` preserved."""
        parts = []
        for gid in game_ids:
            df = self.load_actions(comp, int(gid), use_cache=use_cache)
            if "game_id" not in df.columns:
                df = df.assign(game_id=int(gid))
            parts.append(df)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)


# --- SPADL schema constants (from socceraction, exposed for convenience) ---

def spadl_field_dimensions() -> tuple[float, float]:
    """Pitch dimensions used by socceraction's SPADL (metres)."""
    from socceraction.spadl import config as spadlcfg
    return float(spadlcfg.field_length), float(spadlcfg.field_width)


def spadl_action_types() -> list[str]:
    """Canonical action type labels (index → label)."""
    from socceraction.spadl import config as spadlcfg
    return list(spadlcfg.actiontypes)


def spadl_bodyparts() -> list[str]:
    from socceraction.spadl import config as spadlcfg
    return list(spadlcfg.bodyparts)


def spadl_results() -> list[str]:
    from socceraction.spadl import config as spadlcfg
    return list(spadlcfg.results)
