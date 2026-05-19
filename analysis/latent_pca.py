"""Latent PCA plot entry point."""

from pathlib import Path

from pca_analysis import plot_pca

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    plot_pca(save_path=PROJECT_ROOT / "analysis" / "plots" / "latent_pca.png")


if __name__ == "__main__":
    main()
