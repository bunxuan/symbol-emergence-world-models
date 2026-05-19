from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATENT_PATH = PROJECT_ROOT / "model" / "latent.npy"
SAVE_CLUSTERS = PROJECT_ROOT / "analysis" / "clusters.npy"
SAVE_FIG = PROJECT_ROOT / "analysis" / "plots" / "symbol_clusters.png"


def cluster_symbols(
    latent_path=LATENT_PATH,
    n_clusters=4,
    save_clusters=SAVE_CLUSTERS,
    save_fig=SAVE_FIG,
):
    # 加载 latent
    h = np.load(latent_path)  # shape (T, latent_dim)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    labels = kmeans.fit_predict(h)

    np.save(save_clusters, labels)
    print(f"Saved {save_clusters}, shape = {labels.shape}")

    # PCA 可视化
    pca = PCA(n_components=2)
    h2 = pca.fit_transform(h)

    plt.figure(figsize=(6, 6))
    plt.scatter(h2[:, 0], h2[:, 1], c=labels, cmap="tab10", s=8)
    plt.title("Symbol Clusters in Latent Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    print(f"Saved {save_fig}")


enhanced_symbol_clustering = cluster_symbols


if __name__ == "__main__":
    cluster_symbols()
