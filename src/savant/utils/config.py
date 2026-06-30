"""Config loading.

Loads YAML files from `configs/` and resolves paths relative to the repo root.
Repo root is detected by walking up from this file until a `pyproject.toml`
is found — this keeps the loader usable from scripts, notebooks, and tests
without any environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = ["repo_root", "load_yaml", "load_paths", "load_agents", "load_benchmark_tasks"]


@lru_cache(maxsize=1)
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root (no pyproject.toml found upwards).")


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {p}, got {type(data).__name__}.")
    return data


def load_paths() -> dict[str, Any]:
    return load_yaml("configs/paths.yaml")


def load_agents() -> dict[str, Any]:
    return load_yaml("configs/agents.yaml")


def load_benchmark_tasks() -> dict[str, Any]:
    return load_yaml("configs/benchmark_tasks.yaml")
