"""Paper metadata: load and validate the corpus CSV.

The CSV is the single source of truth for "which papers are in the corpus and
what do we know about each at the bibliographic level." It is hand-curated by
Axum and Pegah (see docs/phase1_workflow.md) until Stage 2 ingestion can
auto-extract bibliographic data from PDFs.

This module does NOT call any external APIs. It only reads CSVs from disk and
returns typed records.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.config import repo_root

__all__ = [
    "PaperRecord",
    "load_paper_metadata",
    "default_papers_csv",
    "default_papers_template_csv",
]

ScreeningDecision = Literal["include", "exclude", "uncertain"]
CorpusPool = Literal["candidate", "held-out"]
VenueType = Literal[
    "journal",
    "conference",
    "workshop",
    "preprint",
    "blog",
    "thesis",
    "report",
    "other",
]

# Subset of scope.tags from configs/extraction_schema.json. Kept in sync via
# tests/test_schema.py — if the schema enum drifts, the test fails loudly.
PrimaryTag = Literal[
    "action-valuation",
    "expected-threat",
    "VAEP",
    "atomic-VAEP",
    "EPV",
    "possession-value",
    "pass-value",
    "shot-value",
    "defensive-action",
    "off-ball-action",
    "tracking-data",
    "event-data",
    "graph-neural-network",
    "deep-learning",
    "reinforcement-learning",
    "markov-decision-process",
    "interpretability",
    "calibration",
    "ranking-players",
    "tactical-analysis",
    "other",
]

_PLACEHOLDER_MARKERS: tuple[str, ...] = ("<PLACEHOLDER", "<TBD")
_TRUE_STRINGS = {"true", "t", "yes", "y", "1"}
_FALSE_STRINGS = {"false", "f", "no", "n", "0"}


def _looks_like_placeholder(value: str) -> bool:
    s = value.strip()
    return any(s.startswith(marker) for marker in _PLACEHOLDER_MARKERS)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip()
    if not s or _looks_like_placeholder(s):
        return None
    return s


def _parse_bool(value: str | None) -> bool | None:
    s = _clean(value)
    if s is None:
        return None
    low = s.lower()
    if low in _TRUE_STRINGS:
        return True
    if low in _FALSE_STRINGS:
        return False
    if low in {"unknown", "uncertain", "na", "n/a"}:
        return None
    raise ValueError(f"Cannot parse boolean from {value!r}")


def _split_authors(raw: str | None) -> list[str]:
    s = _clean(raw)
    if s is None:
        return []
    return [a.strip() for a in s.split(";") if a.strip()]


def _parse_date(raw: str | None) -> date | None:
    s = _clean(raw)
    if s is None:
        return None
    return date.fromisoformat(s)


class PaperRecord(BaseModel):
    """One row of `data/papers/papers.csv` (or the seeded `.template.csv`).

    Placeholder values (`<PLACEHOLDER...>`, empty cells) are normalised to
    `None` so downstream code can distinguish "unknown" from a real value.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    paper_id: str = Field(min_length=1)
    title: str | None
    authors: list[str]
    year: int | None = Field(default=None, ge=1990, le=2100)
    venue: str | None
    venue_type: VenueType | None
    doi: str | None
    url: str | None
    open_access: bool | None
    file_path: str | None
    primary_tag: PrimaryTag | None
    screening_decision: ScreeningDecision | None
    screening_decision_by: str | None
    screening_decision_date: date | None
    needs_full_text: bool | None
    pegah_verified: bool | None
    pegah_verified_date: date | None
    david_verified: bool | None
    david_verified_date: date | None
    corpus_pool: CorpusPool | None
    notes: str | None

    @field_validator("paper_id")
    @classmethod
    def _paper_id_is_slug(cls, v: str) -> str:
        if " " in v:
            raise ValueError(f"paper_id must be a slug (no spaces); got {v!r}")
        return v

    @property
    def is_complete_bibliographic(self) -> bool:
        """True iff all bibliographic fields needed by the extraction schema's
        `bibliographic.required` list are present and non-placeholder."""
        return all(
            x is not None and x != []
            for x in (self.title, self.authors, self.year, self.venue)
        )


def default_papers_csv() -> Path:
    return repo_root() / "data" / "papers" / "papers.csv"


def default_papers_template_csv() -> Path:
    return repo_root() / "data" / "papers" / "papers.template.csv"


def _row_to_record(row: dict[str, str]) -> PaperRecord:
    year_raw = _clean(row.get("year"))
    payload: dict[str, Any] = {
        "paper_id": row["paper_id"].strip(),
        "title": _clean(row.get("title")),
        "authors": _split_authors(row.get("authors")),
        "year": int(year_raw) if year_raw is not None else None,
        "venue": _clean(row.get("venue")),
        "venue_type": _clean(row.get("venue_type")),
        "doi": _clean(row.get("doi")),
        "url": _clean(row.get("url")),
        "open_access": _parse_bool(row.get("open_access")),
        "file_path": _clean(row.get("file_path")),
        "primary_tag": _clean(row.get("primary_tag")),
        "screening_decision": _clean(row.get("screening_decision")),
        "screening_decision_by": _clean(row.get("screening_decision_by")),
        "screening_decision_date": _parse_date(row.get("screening_decision_date")),
        "needs_full_text": _parse_bool(row.get("needs_full_text")),
        "pegah_verified": _parse_bool(row.get("pegah_verified")),
        "pegah_verified_date": _parse_date(row.get("pegah_verified_date")),
        "david_verified": _parse_bool(row.get("david_verified")),
        "david_verified_date": _parse_date(row.get("david_verified_date")),
        "corpus_pool": _clean(row.get("corpus_pool")),
        "notes": _clean(row.get("notes")),
    }
    return PaperRecord.model_validate(payload)


def load_paper_metadata(path: str | Path | None = None) -> list[PaperRecord]:
    """Load the paper-metadata CSV.

    Defaults to `data/papers/papers.csv` if it exists, otherwise falls back to
    the tracked `papers.template.csv` so notebooks work out-of-the-box on a
    fresh clone.
    """
    if path is None:
        live = default_papers_csv()
        path = live if live.is_file() else default_papers_template_csv()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Paper metadata CSV not found: {p}")

    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        seen_ids: set[str] = set()
        records: list[PaperRecord] = []
        for i, row in enumerate(reader, start=2):  # header is line 1
            try:
                record = _row_to_record(row)
            except Exception as exc:
                raise ValueError(f"Error parsing row {i} of {p}: {exc}") from exc
            if record.paper_id in seen_ids:
                raise ValueError(f"Duplicate paper_id {record.paper_id!r} in {p} (row {i}).")
            seen_ids.add(record.paper_id)
            records.append(record)
    return records


def filter_by_screening(
    records: Iterable[PaperRecord],
    *,
    decisions: tuple[ScreeningDecision, ...] = ("include",),
) -> list[PaperRecord]:
    return [r for r in records if r.screening_decision in decisions]
