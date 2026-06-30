"""Shared utilities: config loading, logging, IO, schema validation.

Stage 1 status:
- [x] `config.load_yaml(path)` + repo-root detection
- [x] `logging.get_logger(name)` — Rich console; structured file logs land in Stage 3
- [x] `io.read_json / write_json` (atomic) and `validate_extraction_record`
- [ ] `seeding.seed_everything(seed)` — Stage 3, alongside the LLM client
- [ ] `cost.LedgerWriter` — Stage 3, append-only LLM spend log
"""

from .config import load_agents, load_benchmark_tasks, load_paths, load_yaml, repo_root
from .io import (
    ExtractionValidationError,
    load_extraction_schema,
    read_json,
    validate_extraction_record,
    write_json,
)
from .logging import get_logger

__all__ = [
    "ExtractionValidationError",
    "get_logger",
    "load_agents",
    "load_benchmark_tasks",
    "load_extraction_schema",
    "load_paths",
    "load_yaml",
    "read_json",
    "repo_root",
    "validate_extraction_record",
    "write_json",
]
