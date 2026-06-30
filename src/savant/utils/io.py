"""JSON IO helpers with atomic write and JSON Schema validation.

Reads/writes are deliberately small and explicit. Validation uses
jsonschema's Draft 2020-12 validator — matching the `$schema` declared in
`configs/extraction_schema.json`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .config import repo_root

__all__ = [
    "read_json",
    "write_json",
    "load_extraction_schema",
    "validate_extraction_record",
    "ExtractionValidationError",
]


class ExtractionValidationError(ValueError):
    """Raised when a record fails validation against `extraction_schema.json`.

    `errors` holds the full list of jsonschema validation errors so the
    critic / runner can surface every problem in one pass instead of
    fix-rerun-fix-rerun.
    """

    def __init__(self, message: str, errors: list[ValidationError]):
        super().__init__(message)
        self.errors = errors


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Atomic JSON write: write to a sibling temp file, fsync, rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_extraction_schema() -> dict[str, Any]:
    return read_json(repo_root() / "configs" / "extraction_schema.json")


def validate_extraction_record(record: dict[str, Any]) -> None:
    """Validate an extraction record against `configs/extraction_schema.json`.

    Raises `ExtractionValidationError` carrying every jsonschema error found.
    Returns None on success.
    """
    schema = load_extraction_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    if errors:
        summary = "; ".join(f"{list(e.absolute_path) or '<root>'}: {e.message}" for e in errors)
        raise ExtractionValidationError(
            f"Extraction record failed schema validation ({len(errors)} issue(s)): {summary}",
            errors=errors,
        )
