"""Programmatic extraction-agent runner.

In the batch ingest we invoked the extraction agent via Claude Code
sub-agents. This module wraps the same prompt in the standard `AgentRunner`
pattern so the UI (and any other script) can call it directly with an
`AnthropicClient` or `MockClient`.

Public API:
    ExtractionRunner(client).run(paper_id=..., bibliographic=..., chunks=...)
        -> dict matching configs/extraction_schema.json v0.3.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..agents import AgentRunner, AgentValidationError, LLMClient, format_input_block
from ..utils.io import validate_extraction_record


class ExtractionRunner(AgentRunner):
    """Reads a paper's chunks + bibliographic metadata, returns a v0.3.0 record."""

    AGENT_NAME = "extraction"

    def _render_user_message(
        self,
        *,
        paper_id: str,
        bibliographic: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> str:
        return "\n".join([
            format_input_block("paper_id", paper_id),
            format_input_block("bibliographic", bibliographic),
            format_input_block("chunks", chunks),
            format_input_block("instruction",
                "Produce ONE JSON object validating against extraction_schema.json v0.3.0. "
                "Set schema_version='0.3.0'. Use the paper_id and bibliographic block above verbatim. "
                "For provenance.source_chunks list every chunk_id you actually used."),
        ])

    def _validate(self, response: dict[str, Any]) -> dict[str, Any]:
        # Patch provenance.extracted_at if the agent left it null/missing.
        prov = response.setdefault("provenance", {})
        if not prov.get("extracted_at"):
            prov["extracted_at"] = datetime.now(timezone.utc).isoformat()
        if not prov.get("agent"):
            prov["agent"] = "extraction_agent"
        if not prov.get("model_id"):
            prov["model_id"] = "claude/extraction-runner/v0.3.0"
        if not prov.get("prompt_hash"):
            prov["prompt_hash"] = "ui-live-extraction"
        prov.setdefault("source_chunks", [])
        prov.setdefault("critic", {
            "verdict": "skipped", "model_id": "n/a",
            "notes": "Initial extraction; critic pending."
        })
        # Full schema validation
        try:
            validate_extraction_record(response)
        except Exception as exc:
            raise AgentValidationError(
                f"Extraction does not validate against v0.3.0 schema: {exc}"
            ) from exc
        return response
