"""SAVANT — Hugging Face Spaces entry point.

HF Spaces (and Streamlit Community Cloud) look for `streamlit_app.py` (or
`app.py`) at the repo root by default. This file is a thin shim that:

1. Makes `savant` importable (adds `src/` to sys.path) so the page modules
   can do `from savant.ui...` without a separately-set PYTHONPATH.
2. Delegates to the real entry point under `src/savant/ui/app.py`.

For local development the recommended launcher is still
`./scripts/run_ui.sh` (which sets PYTHONPATH=src). This file exists so the
Streamlit-on-HF deploy works without any additional shell setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inject src/ into sys.path BEFORE any savant import.
REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Re-export the home page's Streamlit script execution.
# Streamlit's multi-page mechanism discovers pages under the same directory
# as the entry script — to keep our nice package structure, we use
# Streamlit's programmatic Page API to mount the existing pages.
import streamlit as st  # noqa: E402

# If running on Streamlit < 1.30 the Page/Navigation API isn't available;
# in that case fall back to importing src/savant/ui/app.py inline.
try:
    from streamlit import navigation, Page  # type: ignore
    pages = [
        Page(SRC / "savant" / "ui" / "app.py",
             title="Home", icon="⚽", default=True),
        Page(SRC / "savant" / "ui" / "pages" / "1_📚_Corpus.py",
             title="Corpus", icon="📚"),
        Page(SRC / "savant" / "ui" / "pages" / "2_🕸_Knowledge_Graph.py",
             title="Knowledge Graph", icon="🕸"),
        Page(SRC / "savant" / "ui" / "pages" / "3_🔥_Coverage.py",
             title="Coverage", icon="🔥"),
        Page(SRC / "savant" / "ui" / "pages" / "4_🎯_Gaps.py",
             title="Gaps", icon="🎯"),
        Page(SRC / "savant" / "ui" / "pages" / "5_🤖_Agentic_Pipeline.py",
             title="Agentic Pipeline", icon="🤖"),
        Page(SRC / "savant" / "ui" / "pages" / "6_⚡_Live_Demo.py",
             title="Live Demo", icon="⚡"),
        Page(SRC / "savant" / "ui" / "pages" / "7_💬_Chat.py",
             title="Chat", icon="💬"),
        Page(SRC / "savant" / "ui" / "pages" / "8_ℹ️_About.py",
             title="About", icon="ℹ️"),
        Page(SRC / "savant" / "ui" / "pages" / "9_🏁_Leaderboard.py",
             title="Leaderboard", icon="🏁"),
    ]
    pg = navigation(pages)
    pg.run()
except ImportError:
    # Older Streamlit — best effort fallback.
    import runpy
    runpy.run_path(str(SRC / "savant" / "ui" / "app.py"), run_name="__main__")
