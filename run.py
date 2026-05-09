"""One-command pipeline for the symbol emergence project."""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(script_path: Path) -> None:
    result = subprocess.run([PYTHON, str(script_path)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    run_script(PROJECT_ROOT / "data" / "generate_data.py")
    run_script(PROJECT_ROOT / "model" / "train.py")
    run_script(PROJECT_ROOT / "analysis" / "pca_analysis.py")
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
