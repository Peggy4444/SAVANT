"""End-to-end smoke test for the WP 2 orchestrator script.

Runs scripts/run_agentic_pipeline.run() in mock mode against tmp output
dirs so it doesn't touch the committed artifacts. Verifies every stage's
artifact lands where expected.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "run_agentic_pipeline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_agentic_pipeline", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_agentic_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script_module():
    return _load_script()


def test_mock_run_produces_all_stage_artifacts(tmp_path, script_module):
    """Default run in mock mode: critic verdicts (dry-run, not persisted)
    plus taxonomy files plus gap rationales."""
    summary = script_module.run(
        mode="mock",
        write=False,  # critic stage should NOT modify extraction JSONs
        extractions_dir=REPO / "data" / "outputs" / "extractions",  # read-only
        taxonomy_out_dir=tmp_path / "taxonomy",
        gap_out_path=tmp_path / "gaps" / "llm_rationales.json",
    )
    # Summary shape
    assert summary["mode"] == "mock"
    assert summary["write"] is False
    assert summary["critic"]["accept"] == 24
    assert isinstance(summary["taxonomy_files"], list)
    assert len(summary["taxonomy_files"]) == 4  # default fields list
    assert summary["gap"]["n_gaps"] == 1

    # Taxonomy files written
    for path_str in summary["taxonomy_files"]:
        p = Path(path_str)
        assert p.exists() and p.stat().st_size > 0

    # Gap rationale written
    gap_file = tmp_path / "gaps" / "llm_rationales.json"
    assert gap_file.exists()
    rationale = json.loads(gap_file.read_text())
    assert "gaps" in rationale
    assert "agent_notes" in rationale


def test_mock_run_skip_flags_respected(tmp_path, script_module):
    summary = script_module.run(
        mode="mock",
        skip_critic=True,
        skip_taxonomy=True,
        extractions_dir=REPO / "data" / "outputs" / "extractions",
        gap_out_path=tmp_path / "gaps" / "llm_rationales.json",
    )
    assert "critic" not in summary
    assert "taxonomy_files" not in summary
    assert "gap" in summary


def test_mock_run_dry_run_does_not_modify_extraction_jsons(
    tmp_path, script_module
):
    """The default critic stage is dry-run. Confirm no extraction JSON is
    re-written by reading mtimes before + after."""
    extractions = REPO / "data" / "outputs" / "extractions"
    before = {p.name: p.stat().st_mtime for p in extractions.glob("*.json")}
    script_module.run(
        mode="mock",
        write=False,
        extractions_dir=extractions,
        taxonomy_out_dir=tmp_path / "taxonomy",
        gap_out_path=tmp_path / "gaps" / "llm_rationales.json",
    )
    after = {p.name: p.stat().st_mtime for p in extractions.glob("*.json")}
    assert before == after, "Critic dry-run unexpectedly rewrote extraction JSONs"
