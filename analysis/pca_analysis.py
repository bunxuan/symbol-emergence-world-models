"""PCA analysis utilities."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_pca(
    latent_path=PROJECT_ROOT / "model" / "latent.npy",
    save_path=PROJECT_ROOT / "analysis" / "plots" / "latent_pca.png",
):
    # 加载 latent
    h = np.load(latent_path)  # shape (T, latent_dim)
    # try to load raw trajectories to detect event times (collisions)
    try:
        traj = np.load(PROJECT_ROOT / "data" / "trajectories.npy").squeeze()
    except Exception:
        traj = None

    # PCA 到 2D
    pca = PCA(n_components=2)
    h2 = pca.fit_transform(h)

    # 画轨迹
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    sc = ax.scatter(
        h2[:, 0],
        h2[:, 1],
        c=range(h2.shape[0]),
        cmap="viridis",
        linewidth=0,
        alpha=0.8,
        s=8,
    )

    ax.set_title("PCA of Latent Trajectory")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)

    # Mark collision/turning points if trajectory available
    if traj is not None and traj.size >= 3:
        vel = np.diff(traj)
        coll_idx = np.where(np.sign(vel[:-1]) != np.sign(vel[1:]))[0] + 1
        coll_idx = coll_idx[(coll_idx >= 0) & (coll_idx < h2.shape[0])]
        if coll_idx.size > 0:
            ax.scatter(
                h2[coll_idx, 0],
                h2[coll_idx, 1],
                c="red",
                marker="x",
                label="collision",
                zorder=3,
            )
            for ci in coll_idx:
                ax.annotate(
                    "|",
                    (h2[ci, 0], h2[ci, 1]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=8,
                    color="red",
                )
            ax.legend(loc="best", fontsize=8)

            # inset zoom around first collision (use inset_axes to be compatible with constrained_layout)
            first = coll_idx[0]
            lo = max(0, first - 10)
            hi = min(h2.shape[0], first + 11)
            ax_in = inset_axes(ax, width="30%", height="30%", loc="upper right")
            ax_in.scatter(h2[lo:hi, 0], h2[lo:hi, 1], c=range(lo, hi), cmap="viridis", s=12)
            ax_in.scatter(h2[first, 0], h2[first, 1], c="red", marker="x")
            ax_in.set_title("Inset: collision region", fontsize=9)
            ax_in.tick_params(axis="both", which="major", labelsize=7)

    # colorbar (time progression)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("time index")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_pca()
