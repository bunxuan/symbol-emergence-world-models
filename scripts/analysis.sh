#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
PYTHON_EXE="$WORKSPACE_ROOT/.venv-torch/Scripts/python.exe"

"$PYTHON_EXE" "$PROJECT_ROOT/analysis/jacobian_analysis.py"
"$PYTHON_EXE" "$PROJECT_ROOT/analysis/symbol_clustering.py"
"$PYTHON_EXE" "$PROJECT_ROOT/analysis/state_machine.py"

if [ -f "$PROJECT_ROOT/model/flow.pth" ] || [ -f "$PROJECT_ROOT/model/flow_samples.npy" ]; then
	"$PYTHON_EXE" "$PROJECT_ROOT/analysis/manifold_plot.py" --mode flow
fi

if [ -f "$PROJECT_ROOT/model/diffusion.pth" ] || [ -f "$PROJECT_ROOT/model/diffusion_chain.npy" ]; then
	"$PYTHON_EXE" "$PROJECT_ROOT/analysis/manifold_plot.py" --mode diffusion
fi
