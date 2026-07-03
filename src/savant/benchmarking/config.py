"""Load and parse `configs/benchmark_tasks.yaml`.

Single seam so every benchmarking submodule reads the config the same way.
Path resolution is anchored at `repo_root()` so scripts and tests behave
identically to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import repo_root


DEFAULT_CONFIG_PATH = repo_root() / "configs" / "benchmark_tasks.yaml"


@dataclass(frozen=True)
class CompetitionSpec:
    """One competition-season row from the config."""
    name: str
    season: str


@dataclass(frozen=True)
class SplitSpec:
    by: str
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int
    require_leakage_audit_pass: bool


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Return the parsed benchmark config."""
    path = path or DEFAULT_CONFIG_PATH
    return yaml.safe_load(path.read_text())


def train_competitions(cfg: dict | None = None) -> list[CompetitionSpec]:
    cfg = cfg or load_config()
    return [CompetitionSpec(c["name"], c["season"])
            for c in cfg["data"]["competitions_train"]]


def holdout_competitions(cfg: dict | None = None) -> list[CompetitionSpec]:
    cfg = cfg or load_config()
    return [CompetitionSpec(c["name"], c["season"])
            for c in cfg["data"]["competitions_holdout_for_t4"]]


def split_spec(cfg: dict | None = None) -> SplitSpec:
    cfg = cfg or load_config()
    s = cfg["data"]["split"]
    return SplitSpec(
        by=s["by"],
        train_fraction=float(s["train_fraction"]),
        val_fraction=float(s["val_fraction"]),
        test_fraction=float(s["test_fraction"]),
        seed=int(s["seed"]),
        require_leakage_audit_pass=bool(s["require_leakage_audit_pass"]),
    )


def cache_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return repo_root() / cfg["data"]["cache_dir"]


def three_sixty_enabled(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg["data"].get("three_sixty", {}).get("enabled", False))


def methods(cfg: dict | None = None) -> list[dict[str, Any]]:
    """Return the ordered list of method entries from configs/benchmark_tasks.yaml.
    Each entry has keys id / name / corpus_paper_id / family / valuation_unit /
    implementation / reference_specification / citation / reports_on_tasks."""
    cfg = cfg or load_config()
    return list(cfg.get("methods") or [])
