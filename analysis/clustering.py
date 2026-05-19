"""Latent clustering entry point."""

from pathlib import Path

from symbol_clustering import cluster_symbols

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cluster_symbols(
        save_clusters=PROJECT_ROOT / "analysis" / "clusters.npy",
        save_fig=PROJECT_ROOT / "analysis" / "plots" / "symbol_clusters.png",
    )


if __name__ == "__main__":
    main()
