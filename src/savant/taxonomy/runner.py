"""Taxonomy-agent runner.

Canonicalises free-text fields across the SAVANT extraction corpus so the
knowledge graph can compare papers on a like-for-like basis.

Typical usage:
    raw = collect_raw_values("evaluation.metrics", load_all_extractions())
    runner = TaxonomyRunner(client)
    mapping = runner.run(field="evaluation.metrics", values=raw,
                         controlled_vocabulary=[], examples=[])
    save_taxonomy(mapping)

The `apply_taxonomy_to_record` helper writes the canonical values to a
`*_canonical` sibling field on the record (e.g. `evaluation.metrics_canonical`)
WITHOUT mutating the original — so the gold-standard records are preserved.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from ..agents import AgentRunner, AgentValidationError, LLMClient, format_input_block
from ..utils.config import repo_root

DEFAULT_TAXONOMY_DIR = repo_root() / "data" / "outputs" / "taxonomy"


class TaxonomyRunner(AgentRunner):
    """One field-normalisation call → one mapping dict."""

    AGENT_NAME = "taxonomy"

    def _render_user_message(
        self,
        *,
        field: str,
        values: list[str],
        controlled_vocabulary: list[str] | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        return "\n".join([
            format_input_block("Field", field),
            format_input_block("Raw values (deduplicated)", sorted(set(values))),
            format_input_block(
                "Controlled vocabulary (current canonicals; may be empty)",
                controlled_vocabulary or [],
            ),
            format_input_block(
                "Examples (paper_id + raw value, optional context)",
                examples or [],
            ),
        ])

    def _validate(self, response: dict[str, Any]) -> dict[str, Any]:
        if response.get("field") in (None, ""):
            raise AgentValidationError("taxonomy response missing 'field'")
        if not isinstance(response.get("mappings"), list):
            raise AgentValidationError(
                "taxonomy response 'mappings' must be a list of {raw, canonical} dicts"
            )
        for i, m in enumerate(response["mappings"]):
            if not isinstance(m, dict) or "raw" not in m or "canonical" not in m:
                raise AgentValidationError(
                    f"mappings[{i}] must be a dict with 'raw' and 'canonical' keys"
                )
            canonical = m["canonical"]
            if not isinstance(canonical, str) or not canonical:
                raise AgentValidationError(
                    f"mappings[{i}].canonical must be a non-empty string; got {canonical!r}"
                )
            # Style check: canonicals must be snake_case-ish ASCII (per prompt).
            if not canonical.replace("_", "").replace("-", "").isalnum():
                raise AgentValidationError(
                    f"mappings[{i}].canonical must be ASCII alnum + _ -; got {canonical!r}"
                )
        for key in ("new_vocabulary", "unresolved"):
            if key in response and not isinstance(response[key], list):
                raise AgentValidationError(
                    f"taxonomy response '{key}' must be a list when present"
                )
        return response


# --- Corpus-wide raw-value collection ---------------------------------------

def _walk(record: dict, path: list[str]) -> Any:
    """Walk a dot-separated path into a nested dict. Returns None if any
    segment is missing."""
    cur: Any = record
    for seg in path:
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def collect_raw_values(field_path: str,
                       records: Iterable[dict[str, Any]]) -> list[str]:
    """Walk `field_path` (e.g. "evaluation.metrics") across every record and
    return a deduplicated list of non-empty string values. Lists of strings
    are flattened; scalar strings are kept as-is."""
    path = field_path.split(".")
    seen: set[str] = set()
    out: list[str] = []
    for rec in records:
        value = _walk(rec, path)
        if value is None:
            continue
        if isinstance(value, str):
            iterable = [value]
        elif isinstance(value, list):
            iterable = [v for v in value if isinstance(v, str)]
        else:
            continue
        for s in iterable:
            s2 = s.strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                out.append(s2)
    return out


def collect_examples(field_path: str,
                     records: Iterable[dict[str, Any]],
                     limit: int = 5) -> list[dict[str, Any]]:
    """Small `{paper_id, raw}` list per field, for the agent's context."""
    path = field_path.split(".")
    out: list[dict[str, Any]] = []
    for rec in records:
        if len(out) >= limit:
            break
        value = _walk(rec, path)
        pid = rec.get("paper_id")
        if value is None or pid is None:
            continue
        out.append({"paper_id": pid, "raw": value})
    return out


# --- Persistence + application ----------------------------------------------

def save_taxonomy(result: dict[str, Any],
                  out_dir: Path | None = None) -> Path:
    """Persist a taxonomy result to `data/outputs/taxonomy/{field_slug}.json`."""
    out_dir = out_dir or DEFAULT_TAXONOMY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    field = result.get("field", "unknown")
    slug = field.replace(".", "_")
    path = out_dir / f"{slug}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return path


def apply_taxonomy_to_record(
    record: dict[str, Any],
    field_path: str,
    mappings: dict[str, str],
) -> dict[str, Any]:
    """Patch a copy of `record` so the value at `{field_path}_canonical` is
    the canonicalised version of the raw value at `field_path`.

    Lists of strings → list of canonicals (preserving order, dropping
    unmapped values). Scalar strings → scalar canonical (or original if
    unmapped). Original `field_path` value is NOT mutated.
    """
    out = deepcopy(record)
    *parent_path, leaf = field_path.split(".")
    parent = out
    for seg in parent_path:
        if not isinstance(parent, dict) or seg not in parent:
            return out  # nothing to do
        parent = parent[seg]
    if not isinstance(parent, dict) or leaf not in parent:
        return out
    raw = parent[leaf]
    canonical_leaf = f"{leaf}_canonical"
    if isinstance(raw, str):
        parent[canonical_leaf] = mappings.get(raw.strip(), raw)
    elif isinstance(raw, list):
        parent[canonical_leaf] = [
            mappings[s.strip()] for s in raw
            if isinstance(s, str) and s.strip() in mappings
        ]
    return out
