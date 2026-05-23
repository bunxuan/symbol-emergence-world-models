from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    # plot each cluster separately so we can add a legend with sizes
    sizes = np.bincount(labels, minlength=n_clusters)
    cmap = plt.get_cmap("tab10")
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        ax.scatter(
            h2[idx, 0],
            h2[idx, 1],
            c=[cmap(k)],
            s=8,
            label=f"cluster {k} (n={sizes[k]})",
        )

    ax.set_title("Symbol Clusters in Latent Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    # overlay collision markers if available
    try:
        traj = np.load(PROJECT_ROOT / "data" / "trajectories.npy").squeeze()
        vel = np.diff(traj)
        coll_idx = np.where(np.sign(vel[:-1]) != np.sign(vel[1:]))[0] + 1
        coll_idx = coll_idx[(coll_idx >= 0) & (coll_idx < h2.shape[0])]
        if coll_idx.size > 0:
            ax.scatter(
                h2[coll_idx, 0], h2[coll_idx, 1], c="k", marker="x", label="collision"
            )
            ax.legend(loc="best", fontsize=8)
    except Exception:
        pass

    # inset: state assignment over time (compact and fully inside the panel)
    ax_in = inset_axes(ax, width="46%", height="16%", loc="lower left", borderpad=1.0)
    ax_in.plot(labels, drawstyle="steps-mid", color="tab:blue", linewidth=1.0)
    ax_in.set_facecolor("white")
    ax_in.set_title("State assignments", fontsize=8, pad=2)
    ax_in.set_xlabel("time", fontsize=7, labelpad=1)
    ax_in.set_ylabel("state", fontsize=7, labelpad=1)
    ax_in.set_yticks(range(min(n_clusters, 10)))
    ax_in.tick_params(axis="both", labelsize=7)
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.8)

    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    print(f"Saved {save_fig}")


enhanced_symbol_clustering = cluster_symbols


if __name__ == "__main__":
    cluster_symbols()
