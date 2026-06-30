"""Validate `configs/extraction_schema.json` and any tracked fixture records.

Run: `pytest tests/test_schema.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from savant.utils.io import (
    ExtractionValidationError,
    load_extraction_schema,
    validate_extraction_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "extraction_schema.json"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "extractions"


def test_schema_is_valid_json() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert schema["title"] == "SAVANT Paper Extraction Record"
    assert schema["properties"]["schema_version"]["const"] == "0.3.0"


def test_schema_is_self_consistent() -> None:
    """The schema itself must conform to the JSON Schema meta-schema."""
    schema = load_extraction_schema()
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    "fixture_path",
    sorted(FIXTURES_DIR.glob("*.json")),
    ids=lambda p: p.name,
)
def test_fixture_record_validates(fixture_path: Path) -> None:
    record = json.loads(fixture_path.read_text())
    validate_extraction_record(record)


def test_invalid_record_is_rejected() -> None:
    """A record missing a required field must raise ExtractionValidationError."""
    fixture = json.loads((FIXTURES_DIR / "decroos_2019_vaep.json").read_text())
    fixture.pop("model")  # required at root
    with pytest.raises(ExtractionValidationError):
        validate_extraction_record(fixture)


def test_primary_tag_enum_matches_schema() -> None:
    """The metadata loader's PrimaryTag Literal must equal scope.tags enum from
    the schema. If the schema drifts, this test fails so the loader gets updated.
    """
    from typing import get_args

    from savant.ingestion.metadata import PrimaryTag

    schema = load_extraction_schema()
    schema_enum = set(schema["properties"]["scope"]["properties"]["tags"]["items"]["enum"])
    literal_values = set(get_args(PrimaryTag))
    assert literal_values == schema_enum, (
        "PrimaryTag in src/savant/ingestion/metadata.py is out of sync with "
        "configs/extraction_schema.json scope.tags enum."
    )
