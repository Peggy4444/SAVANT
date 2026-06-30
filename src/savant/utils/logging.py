"""Logger factory. One module-level configuration; callers ask for a named logger.

Stage 1 minimum: rich-formatted console handler, level from $SAVANT_LOG_LEVEL
(default INFO). Structured JSON logs to disk land in Stage 3 once the LLM
client and cost ledger are in.
"""

from __future__ import annotations

import logging
import os

# Rich is optional: gives pretty CLI output when available, but is not
# required (so the package works in slim environments like the Streamlit
# Cloud or HF Spaces base image, or any Python install where rich
# happens not to be present).
try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only in slim envs
    RichHandler = None  # type: ignore[assignment]
    _HAS_RICH = False

__all__ = ["get_logger"]

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("SAVANT_LOG_LEVEL", "INFO").upper()
    if _HAS_RICH:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_time=True, show_path=False
        )
        log_format = "%(message)s"
    else:
        handler = logging.StreamHandler()
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
