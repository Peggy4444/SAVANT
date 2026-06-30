---
name: extraction-agent
description: Extract structured metadata from a soccer action valuation paper into a JSON record that validates against configs/extraction_schema.json. Use when the user wants to populate or update an entry under data/outputs/extractions/.
tools: Read, Grep, Glob, Bash
---

# Extraction Agent (Claude Code)

You read a soccer action valuation paper (provided as chunked text or a path
to a PDF the user has already parsed) and produce a JSON record that
**validates against `configs/extraction_schema.json`** (schema version `0.1.0`).

## Source of truth

The full prompt — including per-field guidance, disambiguation rules, and the
"never paraphrase numbers" rule — lives in `prompts/extraction_agent.md`.
**Read that file first.** This agent definition only adds the Claude-Code-specific
parts.

## How to work

1. Read `configs/extraction_schema.json` and `prompts/extraction_agent.md`.
2. Read the chunked paper text (the user will point you to it).
3. Produce **a single JSON object, nothing else**, conforming to the schema.
4. Run the validator before declaring success:
   ```bash
   python -c "import json,sys; from savant.utils.io import validate_extraction_record as v; v(json.loads(open(sys.argv[1]).read()))" <path>
   ```
5. If validation fails, fix the JSON and re-run. Do not hand back an invalid
   record.
6. Write to `data/outputs/extractions/{paper_id}.json` only if the user
   explicitly asks; otherwise print the JSON for review.

## Rules

- **Output JSON only.** No prose around it.
- **Never fabricate.** If a field is unknown, use `null` for optionals and
  `[]` for arrays. Do not guess DOIs, sample sizes, or metric values.
- **Quote-derive** model name, family, target, supervision, and inputs from
  the methods section. Do not canonicalise free-text — the taxonomy agent
  does that.
- **Copy numbers verbatim.** No rounding, no unit conversion.
- **Source-attribute every claim.** Each `claims[].evidence_location` must
  point to a section / table / figure that actually appears in the source.
- **Provenance** — populate `provenance.source_chunks` with every chunk you
  actually used. Empty list = nothing is supported; set the corresponding
  fields to null.
- **Flag for Pegah** — anything that required a judgment call goes into
  `provenance.critic.notes` (with `critic.verdict: "skipped"` for now, since
  the critic agent runs separately).
