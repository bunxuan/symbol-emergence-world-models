"""PCA analysis utilities."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_pca(
    latent_path=PROJECT_ROOT / "model" / "latent.npy",
    save_path=PROJECT_ROOT / "analysis" / "plots" / "latent_pca.png",
):
    # 加载 latent
    h = np.load(latent_path)  # shape (T, latent_dim)

    # PCA 到 2D
    pca = PCA(n_components=2)
    h2 = pca.fit_transform(h)

    # 画轨迹
    plt.figure(figsize=(6, 6))
    plt.scatter(h2[:, 0], h2[:, 1], linewidth=1.0, alpha=0.8)

    plt.title("PCA of Latent Trajectory")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_pca()
