from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATENT_PATH = PROJECT_ROOT / "model" / "latent.npy"
SAVE_CLUSTERS = PROJECT_ROOT / "analysis" / "data" / "clusters.npy"
SAVE_FIG = PROJECT_ROOT / "analysis" / "plots" / "symbol_clusters.png"
SAVE_SEGMENT_ENTROPY = PROJECT_ROOT / "analysis" / "data" / "segment_entropy.npz"
SAVE_SEGMENT_ENTROPY_FIG = (
    PROJECT_ROOT / "report" / "figures" / "entropy" / "fig9b_segment_entropy.png"
)


def _gaussian_entropy(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]

    variances = np.var(samples, axis=0, ddof=0)
    variances = np.clip(variances, 1e-8, None)
    return 0.5 * np.sum(np.log(2.0 * np.pi * np.e * variances))


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

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    cluster_entropies = np.array(
        [_gaussian_entropy(h[labels == k]) for k in range(n_clusters)], dtype=np.float64
    )
    global_entropy = float(_gaussian_entropy(h))
    weighted_entropy = float(np.sum((cluster_sizes / len(labels)) * cluster_entropies))
    information_gain = float(global_entropy - weighted_entropy)

    SAVE_SEGMENT_ENTROPY.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        SAVE_SEGMENT_ENTROPY,
        cluster_entropies=cluster_entropies,
        cluster_sizes=cluster_sizes,
        global_entropy=global_entropy,
        weighted_entropy=weighted_entropy,
        information_gain=information_gain,
    )
    print(f"Saved {SAVE_SEGMENT_ENTROPY}")

    cmap = plt.get_cmap("tab10")
    SAVE_SEGMENT_ENTROPY_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig_entropy, ax_entropy = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    bars = ax_entropy.bar(
        range(n_clusters),
        cluster_entropies,
        color=[cmap(k) for k in range(n_clusters)],
        edgecolor="#1f2937",
    )
    ax_entropy.axhline(
        global_entropy,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"global H(Z) = {global_entropy:.2f}",
    )
    ax_entropy.axhline(
        weighted_entropy,
        color="#ea580c",
        linestyle=":",
        linewidth=1.4,
        label=f"weighted H(Z|S) = {weighted_entropy:.2f}",
    )
    ax_entropy.set_xticks(range(n_clusters))
    ax_entropy.set_xticklabels(
        [f"cluster {k}\n(n={cluster_sizes[k]})" for k in range(n_clusters)]
    )
    ax_entropy.set_ylabel("differential entropy (nats)")
    ax_entropy.set_xlabel("symbolic cluster")
    ax_entropy.set_title("Segment internal entropy")
    ax_entropy.grid(True, axis="y", alpha=0.25)
    ax_entropy.legend(loc="best", fontsize=8)
    ax_entropy.text(
        0.02,
        0.98,
        f"I(Z;S) = {information_gain:.2f} nats",
        transform=ax_entropy.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.95),
    )
    fig_entropy.savefig(SAVE_SEGMENT_ENTROPY_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig_entropy)
    print(f"Saved {SAVE_SEGMENT_ENTROPY_FIG}")

    # PCA 可视化
    pca = PCA(n_components=2)
    h2 = pca.fit_transform(h)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    # plot each cluster separately so we can add a legend with sizes
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        ax.scatter(
            h2[idx, 0],
            h2[idx, 1],
            c=[cmap(k)],
            s=8,
            label=f"cluster {k} (n={cluster_sizes[k]})",
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
