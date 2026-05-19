"""Diffusion dynamics comparison entry point."""

from pathlib import Path

from manifold_plot import plot_diffusion_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    plot_diffusion_comparison(
        save_path=PROJECT_ROOT / "analysis" / "plots" / "diffusion_vs_real.png"
    )


if __name__ == "__main__":
    main()
