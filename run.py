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


def run_randomization_pipeline() -> None:
    # Keep model, loss, and analysis logic fixed; only perturb trajectory order.
    run_script(
        PROJECT_ROOT / "analysis" / "randomization_study.py",
        "--ratios",
        "0,20,100",
        "--epochs",
        "50",
        "--batch-size",
        "32",
        "--latent-dim",
        "16",
    )


def run_test() -> None:
    run_script(PROJECT_ROOT / "data" / "generate_data.py")
    run_script(PROJECT_ROOT / "model" / "train_world.py")
    run_script(PROJECT_ROOT / "analysis" / "latent_pca.py")
    run_script(PROJECT_ROOT / "analysis" / "jacobian_analysis.py")
    run_script(PROJECT_ROOT / "analysis" / "clustering.py")
    run_script(PROJECT_ROOT / "analysis" / "state_machine.py")


def run_gridworld_pipeline() -> None:
    # 1) collect raw 2D states/actions
    run_script(PROJECT_ROOT / "envs" / "collect_2d.py")
    # 2) train the shared world model with state_dim=2
    run_script(
        PROJECT_ROOT / "model" / "train.py",
        "--mode",
        "world",
        "--data-path",
        "data/trajectories_2d.npy",
        "--state-dim",
        "2",
        "--epochs",
        "20",
        "--batch-size",
        "32",
        "--latent-dim",
        "16",
    )
    # 3) compute Jacobian / entropy / clustering / MI figures
    run_script(PROJECT_ROOT / "analysis" / "run_analysis.py")
    # 4) render the spatial segmentation map
    run_script(PROJECT_ROOT / "analysis" / "segmentation.py")


def run_chaos_pipeline() -> None:
    # 1. 生成混沌轨迹
    run_script(PROJECT_ROOT / "envs" / "generate_lorenz.py")
    # 2. 训练世界模型
    run_script(
        PROJECT_ROOT / "model" / "train.py",
        "--mode",
        "world",
        "--data-path",
        "data/lorenz_trajectories.npy",
        "--state-dim",
        "3",
        "--epochs",
        "50",
        "--batch-size",
        "32",
        "--latent-dim",
        "16",
    )
    # 3. 分析
    run_script(PROJECT_ROOT / "analysis" / "run_analysis.py")
    # 4. 可视化
    run_script(PROJECT_ROOT / "analysis" / "segmentation.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the symbol emergence pipeline.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="world",
        choices=[
            "world",
            "flow",
            "diffusion",
            "gridworld",
            "2d",
            "randomization",
            "random",
            "all",
        ],
    )
    args = parser.parse_args()

    if args.mode == "world":
        run_world_pipeline()
    elif args.mode == "flow":
        run_flow_pipeline()
    elif args.mode == "diffusion":
        run_diffusion_pipeline()
    elif args.mode in {"randomization", "random"}:
        run_randomization_pipeline()
    elif args.mode == "test":
        run_chaos_pipeline()
    elif args.mode in {"gridworld", "2d"}:
        run_gridworld_pipeline()
    else:
        run_world_pipeline()
        run_flow_pipeline()
        run_diffusion_pipeline()

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
