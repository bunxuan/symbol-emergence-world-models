"""One-command pipeline for the symbol emergence project."""

from pathlib import Path
import argparse
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(script_path: Path, *args: str) -> None:
    result = subprocess.run([PYTHON, str(script_path), *args], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_world_pipeline() -> None:
    run_script(PROJECT_ROOT / "data" / "generate_data.py")
    run_script(PROJECT_ROOT / "model" / "train_world.py")
    run_script(PROJECT_ROOT / "analysis" / "latent_pca.py")
    run_script(PROJECT_ROOT / "analysis" / "jacobian_analysis.py")
    run_script(PROJECT_ROOT / "analysis" / "clustering.py")
    run_script(PROJECT_ROOT / "analysis" / "state_machine.py")


def run_flow_pipeline() -> None:
    # run_script(PROJECT_ROOT / "model" / "train_flow.py")
    run_script(PROJECT_ROOT / "analysis" / "flow_geometry.py")
    run_script(PROJECT_ROOT / "analysis" / "flow_jacobian.py")
    run_script(PROJECT_ROOT / "analysis" / "flow_segmentation.py")
    run_script(PROJECT_ROOT / "analysis" / "latent_structure_comparison.py")


def run_diffusion_pipeline() -> None:
    run_script(PROJECT_ROOT / "model" / "train_diffusion.py")
    run_script(PROJECT_ROOT / "analysis" / "diffusion_dynamics.py")
    run_script(PROJECT_ROOT / "analysis" / "diffusion_figures.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the symbol emergence pipeline.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="world",
        choices=["world", "flow", "diffusion", "all"],
    )
    args = parser.parse_args()

    if args.mode == "world":
        run_world_pipeline()
    elif args.mode == "flow":
        run_flow_pipeline()
    elif args.mode == "diffusion":
        run_diffusion_pipeline()
    else:
        run_world_pipeline()
        run_flow_pipeline()
        run_diffusion_pipeline()

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
