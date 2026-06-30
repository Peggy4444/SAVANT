"""B8 — minimal live-LLM demo: run the critic agent on ONE paper.

This is the smallest possible end-to-end validation of the WP 2 live
pipeline. It:

1. Loads ONE extraction record (decroos_2019_vaep by default — the most-
   reviewed paper in the corpus, so the result is easy to sanity-check).
2. Calls `CriticRunner` with `AnthropicClient` (real API).
3. Prints the verdict + structured issues.
4. Reports approximate input/output token counts from the call log.
5. **Does NOT modify any files.** Pure read + print.

Prerequisites:
- `ANTHROPIC_API_KEY` set in env, or in a `.env` file at the repo root.
- ~$0.01–$0.05 of API tokens.

Usage:
    # Set the key in this session
    export ANTHROPIC_API_KEY=sk-ant-...

    # Then run:
    PYTHONPATH=src python scripts/demo_live_critic.py
    PYTHONPATH=src python scripts/demo_live_critic.py --paper-id singh_2018_xt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Best-effort load of .env if present and dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except ImportError:
    pass

from savant.agents import AnthropicClient  # noqa: E402
from savant.agents.client import default_log_dir  # noqa: E402
from savant.extraction.critic import CriticRunner  # noqa: E402


def _load_record(paper_id: str) -> dict:
    path = REPO / "data" / "outputs" / "extractions" / f"{paper_id}.json"
    if not path.exists():
        raise SystemExit(
            f"No extraction JSON found for {paper_id!r}. Available: "
            f"{[p.stem for p in path.parent.glob('*.json')]}"
        )
    return json.loads(path.read_text())


def _build_source_text(paper_id: str) -> str:
    """For the demo we cannot pass the full PDF (chunking not implemented
    yet), so we hand the critic the record's own claims + scope rationale
    as a stand-in for source material. That's an honest minimum-source
    test: the critic should accept verdicts where the record content is
    consistent with itself, and flag obvious internal contradictions."""
    record = _load_record(paper_id)
    parts = [
        f"PDF excerpt placeholder for {paper_id}.",
        f"Scope rationale (per the existing extraction): "
        f"{record['scope'].get('rationale', '(none)')}",
        "Claims with paper-section citations:",
    ]
    for c in record.get("claims", []):
        parts.append(f"  - [{c.get('evidence_location', '?')}] {c.get('statement', '')}")
    parts.append("(Full PDF text not available in this demo — chunking is WP 2 follow-up.)")
    return "\n".join(parts)


def _pretty_print_verdict(verdict: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  VERDICT: {verdict.get('verdict', '?').upper()}")
    print("=" * 60)
    print(f"  schema_conformant: {verdict.get('schema_conformant')}")
    for key in ("factual_issues", "completeness_issues", "style_issues"):
        items = verdict.get(key) or []
        if items:
            print(f"\n  {key}:")
            for item in items:
                print(f"    - {item}")
        else:
            print(f"  {key}: (none)")
    if verdict.get("revision_instructions"):
        print(f"\n  revision_instructions:\n    {verdict['revision_instructions']}")
    if verdict.get("notes"):
        print(f"\n  notes:\n    {verdict['notes']}")
    print()


def _report_latest_call_log() -> None:
    """Show the last line of today's call log so the user can see tokens used."""
    log_dir = default_log_dir()
    today_logs = sorted(log_dir.glob("*.jsonl"))
    if not today_logs:
        return
    latest = today_logs[-1]
    lines = latest.read_text().strip().splitlines()
    if not lines:
        return
    last = json.loads(lines[-1])
    print("--- Call log entry ---")
    print(f"  log file       : {latest}")
    print(f"  model          : {last.get('model')}")
    print(f"  input tokens   : {last.get('input_tokens')}")
    print(f"  output tokens  : {last.get('output_tokens')}")
    print(f"  duration       : {last.get('duration_s')}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-id", default="decroos_2019_vaep",
                    help="paper to critique (default: decroos_2019_vaep)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set in your environment.\n"
            "Set it in this session with: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or put it in a .env file at the repo root."
        )

    record = _load_record(args.paper_id)
    source = _build_source_text(args.paper_id)

    print(f"\nB8: live critic-agent demo on {args.paper_id!r}\n")
    print(f"  source length: {len(source)} chars (record claims, not full PDF)")
    print("  calling Anthropic API …")

    client = AnthropicClient()  # picks up ANTHROPIC_API_KEY from env
    runner = CriticRunner(client)
    verdict = runner.run(record=record, source=source)

    _pretty_print_verdict(verdict)
    _report_latest_call_log()
    print("\nNo files were modified by this demo (read-only).\n")


if __name__ == "__main__":
    main()
