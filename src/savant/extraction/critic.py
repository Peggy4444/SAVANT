"""Critic-agent runner.

Reviews one extraction record against its source PDF excerpt and returns a
structured verdict (accept / revise / reject) plus actionable issues. The
verdict can be patched back into the record's `provenance.critic` block via
`apply_verdict_to_record`.

Ergonomic one-liner:
    from savant.agents import MockClient
    from savant.extraction.critic import CriticRunner, critique_extraction

    result = critique_extraction(record, source_text, MockClient([{...}]))
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from ..agents import AgentRunner, AgentValidationError, LLMClient, format_input_block

VALID_VERDICTS = {"accept", "revise", "reject"}


class CriticRunner(AgentRunner):
    """One call → one structured critique."""

    AGENT_NAME = "critic"

    def _render_user_message(
        self,
        *,
        record: dict[str, Any],
        source: str,
        upstream_agent: str = "extraction",
        schema_hint: str = "extraction_schema.json v0.3.0",
    ) -> str:
        return "\n".join([
            format_input_block("Agent", upstream_agent),
            format_input_block("Output to critique", record),
            format_input_block("Source", source or "(no source excerpt provided)"),
            format_input_block("Schema", schema_hint),
        ])

    def _validate(self, response: dict[str, Any]) -> dict[str, Any]:
        verdict = response.get("verdict")
        if verdict not in VALID_VERDICTS:
            raise AgentValidationError(
                f"critic verdict must be one of {sorted(VALID_VERDICTS)}; got {verdict!r}"
            )
        # `schema_conformant` should be a boolean when present (the prompt
        # describes it as such); coerce missing to None silently.
        if "schema_conformant" in response and not isinstance(
            response["schema_conformant"], bool
        ):
            raise AgentValidationError(
                f"critic.schema_conformant must be boolean if present; "
                f"got {type(response['schema_conformant']).__name__}"
            )
        return response


# --- Convenience ------------------------------------------------------------

def critique_extraction(
    record: dict[str, Any],
    source: str,
    client: LLMClient,
) -> dict[str, Any]:
    """One-shot helper. Convenience over CriticRunner for callers that don't
    need to reuse the runner."""
    return CriticRunner(client).run(record=record, source=source)


def apply_verdict_to_record(
    record: dict[str, Any],
    verdict: dict[str, Any],
    *,
    model_id: str,
) -> dict[str, Any]:
    """Patch a copy of `record` with the critic's verdict in
    `provenance.critic`. Returns a NEW dict; original record is untouched.

    The patch overwrites the previous critic notes (typically the "skipped"
    placeholder from the batch extraction). The factual / completeness /
    style / revision_instructions lists are folded into a single readable
    notes string so the schema's existing field (a single string) stays
    valid.
    """
    out = deepcopy(record)
    provenance = out.setdefault("provenance", {})
    notes_parts = []
    for k in ("notes",):
        if verdict.get(k):
            notes_parts.append(str(verdict[k]))
    for k in ("factual_issues", "completeness_issues", "style_issues"):
        items = verdict.get(k) or []
        if items:
            notes_parts.append(f"{k}: " + json.dumps(items, ensure_ascii=False))
    if verdict.get("revision_instructions"):
        notes_parts.append("revision_instructions: " + str(verdict["revision_instructions"]))
    provenance["critic"] = {
        "verdict": verdict["verdict"],
        "model_id": model_id,
        "notes": "\n".join(notes_parts) if notes_parts else None,
    }
    return out
