"""Thin extraction-side wrapper around `savant.utils.io.validate_extraction_record`.

Lives under `extraction/` so the extraction runner (Stage 3) can import a
single symbol without reaching into utils.
"""

from __future__ import annotations

from ..utils.io import (
    ExtractionValidationError,
    load_extraction_schema,
    validate_extraction_record,
)

__all__ = [
    "ExtractionValidationError",
    "load_extraction_schema",
    "validate_extraction_record",
]
