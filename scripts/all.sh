#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
PYTHON_EXE="$WORKSPACE_ROOT/.venv-torch/Scripts/python.exe"

"$PYTHON_EXE" "$PROJECT_ROOT/data/generate_data.py"
"$PYTHON_EXE" "$PROJECT_ROOT/model/train.py"
"$PYTHON_EXE" "$PROJECT_ROOT/analysis/pca_analysis.py"
"$PYTHON_EXE" "$PROJECT_ROOT/analysis/jacobian_analysis.py"
"$PYTHON_EXE" "$PROJECT_ROOT/analysis/symbol_clustering.py"
"$PYTHON_EXE" "$PROJECT_ROOT/analysis/state_machine.py"

echo "All pipeline steps finished."