"""Flow geometry comparison entry point."""

from pathlib import Path

from manifold_plot import plot_flow_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    plot_flow_comparison(
        save_path=PROJECT_ROOT / "analysis" / "plots" / "flow_vs_latent.png"
    )


if __name__ == "__main__":
    main()
