"""Paper metadata CSV loader tests (Phase 1).

These tests exercise the tracked `data/papers/papers.template.csv` so they
work on a fresh clone with no data populated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from savant.ingestion import (
    PaperRecord,
    default_papers_template_csv,
    filter_by_screening,
    load_paper_metadata,
)


def test_template_csv_loads() -> None:
    records = load_paper_metadata(default_papers_template_csv())
    assert len(records) >= 20, (
        "Template should ship with the full Phase-1 corpus (28 papers as of "
        "the 2026-06-29 ingest)."
    )
    for r in records:
        assert isinstance(r, PaperRecord)
        assert r.paper_id  # non-empty


def test_template_csv_includes_canonical_av_methods() -> None:
    """The canonical action-valuation methods must always be in the corpus —
    if one of these slugs disappears the manuscript loses its baseline coverage."""
    records = {r.paper_id: r for r in load_paper_metadata(default_papers_template_csv())}
    canonical = {
        "singh_2018_xt",
        "decroos_2019_vaep",
        "fernandez_2019_epv",
        "spearman_2018_beyondxg",
        "rudd_2011_markov",
        "brooks_2016_passrank",
    }
    missing = canonical - records.keys()
    assert not missing, f"Canonical AV methods missing from template: {missing}"


def test_placeholders_are_normalised_to_none(tmp_path: Path) -> None:
    """The loader must turn <PLACEHOLDER...> cells and empty cells into None
    rather than raw strings. Tested on a synthetic CSV because the current
    corpus contains no placeholders."""
    csv = tmp_path / "placeholders.csv"
    csv.write_text(
        "paper_id,title,authors,year,venue,venue_type,doi,url,open_access,file_path,"
        "primary_tag,screening_decision,screening_decision_by,screening_decision_date,"
        "needs_full_text,pegah_verified,pegah_verified_date,david_verified,david_verified_date,"
        "corpus_pool,notes\n"
        "fake_paper,<PLACEHOLDER: title>,<PLACEHOLDER: authors>,,<PLACEHOLDER: venue>,"
        ",,,,,VAEP,uncertain,axum,2026-06-29,,false,,false,,candidate,test\n"
    )
    records = {r.paper_id: r for r in load_paper_metadata(csv)}
    fake = records["fake_paper"]
    assert fake.title is None
    assert fake.authors == []
    assert fake.year is None
    assert fake.venue is None
    assert fake.is_complete_bibliographic is False


def test_decroos_row_is_fully_bibliographic() -> None:
    records = {r.paper_id: r for r in load_paper_metadata(default_papers_template_csv())}
    vaep = records["decroos_2019_vaep"]
    assert vaep.title is not None
    assert "Tom Decroos" in vaep.authors
    assert vaep.year == 2019
    assert vaep.doi == "10.1145/3292500.3330758"
    assert vaep.is_complete_bibliographic is True


def test_filter_by_screening() -> None:
    records = load_paper_metadata(default_papers_template_csv())
    included = filter_by_screening(records, decisions=("include",))
    uncertain = filter_by_screening(records, decisions=("uncertain",))
    assert all(r.screening_decision == "include" for r in included)
    assert all(r.screening_decision == "uncertain" for r in uncertain)
    assert len(included) + len(uncertain) <= len(records)


def test_every_template_row_has_a_corpus_pool() -> None:
    records = load_paper_metadata(default_papers_template_csv())
    for r in records:
        assert r.corpus_pool in {"candidate", "held-out"}, (
            f"{r.paper_id!r} has corpus_pool={r.corpus_pool!r}; "
            "every template row must be classified as candidate or held-out."
        )
    pools = {r.corpus_pool for r in records}
    assert "candidate" in pools and "held-out" in pools, (
        "Template should contain both candidate and held-out rows after the "
        "Xarles-2025 survey expansion."
    )


def test_duplicate_paper_id_is_rejected(tmp_path: Path) -> None:
    csv = tmp_path / "dup.csv"
    csv.write_text(
        "paper_id,title,authors,year,venue,venue_type,doi,url,open_access,file_path,"
        "primary_tag,screening_decision,screening_decision_by,screening_decision_date,"
        "needs_full_text,pegah_verified,pegah_verified_date,david_verified,david_verified_date,"
        "corpus_pool,notes\n"
        "x,T,A,2020,V,journal,,,,,VAEP,include,axum,2026-06-26,false,false,,false,,candidate,\n"
        "x,T,A,2020,V,journal,,,,,VAEP,include,axum,2026-06-26,false,false,,false,,candidate,\n"
    )
    with pytest.raises(ValueError, match="Duplicate paper_id"):
        load_paper_metadata(csv)
