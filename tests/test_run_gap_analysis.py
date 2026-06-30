"""End-to-end smoke test for the Track-A pipeline script.

Exercises scripts/run_gap_analysis.run() against tmp output directories so
that running it doesn't clobber the committed artifacts under
data/outputs/.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "run_gap_analysis.py"


def _load_script():
    """Import scripts/run_gap_analysis.py as a module without polluting sys.path."""
    spec = importlib.util.spec_from_file_location("run_gap_analysis", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_gap_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script_module():
    return _load_script()


def test_run_produces_all_artifacts(tmp_path, script_module):
    """Default run (skip interactive — pyvis writes to a hardcoded location)
    produces a KG, three figures, an enumeration, and a ranking."""
    kg_dir = tmp_path / "kg"
    fig_dir = tmp_path / "figures"
    gaps_dir = tmp_path / "gaps"

    artifacts = script_module.run(
        out_kg_dir=kg_dir,
        out_figures_dir=fig_dir,
        out_gaps_dir=gaps_dir,
        skip_interactive=True,  # pyvis html is anchored to repo path; not tested here
    )

    expected = {
        "kg_json", "kg_graphml",
        "fig_lineage", "fig_coverage", "fig_dataset",
        "enumeration", "ranking",
    }
    assert set(artifacts) == expected, (
        f"Missing: {expected - artifacts.keys()}  Extra: {artifacts.keys() - expected}"
    )
    for label, path in artifacts.items():
        assert path.exists(), f"{label}: {path} not created"
        assert path.stat().st_size > 0, f"{label}: {path} is empty"


def test_run_with_skip_figures_still_produces_data_artifacts(tmp_path, script_module):
    """--skip-figures still rebuilds the data-side artifacts (KG, enumeration,
    ranking) — useful when you only want fresh metrics."""
    artifacts = script_module.run(
        out_kg_dir=tmp_path / "kg",
        out_gaps_dir=tmp_path / "gaps",
        skip_figures=True,
        skip_interactive=True,
    )
    for required in ("kg_json", "kg_graphml", "enumeration", "ranking"):
        assert required in artifacts and artifacts[required].exists()
    assert "fig_lineage" not in artifacts


def test_ranking_artifact_has_expected_structure(tmp_path, script_module):
    """The ranking JSON the script writes must validate against A6's expected
    shape, including the documented weights."""
    import json
    artifacts = script_module.run(
        out_kg_dir=tmp_path / "kg",
        out_figures_dir=tmp_path / "figures",
        out_gaps_dir=tmp_path / "gaps",
        skip_interactive=True,
    )
    ranking = json.loads(artifacts["ranking"].read_text())
    assert ranking["schema_version"] == "1.0"
    assert ranking["weights"]["methodological"] == 1.0  # primary per A6 design
    assert ranking["n_candidate_cells"] > 0
    assert len(ranking["ranked_gaps"]) == ranking["n_candidate_cells"]
    assert ranking["ranked_gaps"][0]["rank"] == 1
