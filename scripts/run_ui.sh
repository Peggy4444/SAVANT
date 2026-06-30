#!/usr/bin/env bash
# Launch the SAVANT Streamlit UI locally.
#
# Usage:
#   ./scripts/run_ui.sh                  # opens at http://localhost:8501
#   ./scripts/run_ui.sh --port 9000
set -euo pipefail

cd "$(dirname "$0")/.."

# Ensure PYTHONPATH so streamlit can import the savant package without install.
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

exec streamlit run src/savant/ui/app.py "$@"
