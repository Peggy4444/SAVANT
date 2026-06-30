"""Paper ingestion: CSV-driven metadata today; PDF → text → chunks in Stage 2.

Stage 1 status:
- [x] `metadata.PaperRecord` + `metadata.load_paper_metadata(csv_path)`

Stage 2 (not yet implemented):
- [ ] `parse_pdf(path) -> RawPaper` using PyMuPDF, falling back to pdfminer.
- [ ] `detect_sections(raw) -> SectionedPaper` with heuristics + regex.
- [ ] `chunk(sectioned, max_tokens) -> list[Chunk]` preserving section labels.
- [ ] `dedupe(papers) -> list[Paper]` on DOI / title-similarity.
- [ ] Persist artefacts to `data/raw/` and `data/processed/`.
"""

from .metadata import (
    PaperRecord,
    default_papers_csv,
    default_papers_template_csv,
    filter_by_screening,
    load_paper_metadata,
)

__all__ = [
    "PaperRecord",
    "default_papers_csv",
    "default_papers_template_csv",
    "filter_by_screening",
    "load_paper_metadata",
]
