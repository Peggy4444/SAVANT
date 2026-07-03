"""Data loaders for the SAVANT Track-B benchmark."""

from .statsbomb_open import (
    StatsBombOpenLoader,
    load_events_for_competitions,
    resolve_competition,
)

__all__ = [
    "StatsBombOpenLoader",
    "load_events_for_competitions",
    "resolve_competition",
]
