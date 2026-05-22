"""Flow latent trajectory segmentation with PCA and KMeans."""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from manifold_plot import _load_latent
from world_model import FlowModule

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
LATENT_PATH = MODEL_DIR / "latent.npy"
FLOW_MODEL_PATH = MODEL_DIR / "flow.pth"
SAVE_FIG = ROOT / "analysis" / "plots" / "flow_latent_segmentation.png"
SAVE_ARROW_FIG = ROOT / "analysis" / "plots" / "flow_latent_segmentation_arrows.png"
SAVE_LABELS = ROOT / "analysis" / "flow_segmentation_labels.npy"
FLOW_NUM_LAYERS = 8
FLOW_HIDDEN_DIM = 128


def _flow_latent_trajectory(flow, latent):
    flow.eval()
    with torch.no_grad():
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        flow_latent, _ = flow.flow(latent_tensor, reverse=False)
    return flow_latent.cpu().numpy().astype(np.float32)


def plot_flow_latent_segmentation(
    latent_path=LATENT_PATH,
    flow_path=FLOW_MODEL_PATH,
    save_path=SAVE_FIG,
    save_labels_path=SAVE_LABELS,
    n_clusters=4,
):
    latent = _load_latent(latent_path)
    latent_dim = latent.shape[1]

    if not flow_path.exists():
        raise FileNotFoundError(f"Missing flow checkpoint: {flow_path}")

    flow = FlowModule(
        latent_dim, num_layers=FLOW_NUM_LAYERS, hidden_dim=FLOW_HIDDEN_DIM
    )
    flow.load_state_dict(torch.load(flow_path, map_location="cpu"))
    flow.eval()

    flow_latent = _flow_latent_trajectory(flow, latent)

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(flow_latent)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    labels = kmeans.fit_predict(flow_latent)

    np.save(save_labels_path, labels)
    print(f"Saved {save_labels_path}, shape = {labels.shape}")

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(
        latent_2d[:, 0],
        latent_2d[:, 1],
        color="#cbd5e1",
        linewidth=1.0,
        alpha=0.7,
        zorder=1,
    )
    scatter = ax.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.95,
        zorder=2,
    )

    ax.set_title("Flow Latent Trajectory + KMeans Segmentation")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
        frameon=True,
    )
    ax.add_artist(legend)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_flow_latent_segmentation_with_arrows(
    latent_path=LATENT_PATH,
    flow_path=FLOW_MODEL_PATH,
    save_path=SAVE_ARROW_FIG,
    n_clusters=4,
    arrow_step=40,
):
    latent = _load_latent(latent_path)
    latent_dim = latent.shape[1]

    if not flow_path.exists():
        raise FileNotFoundError(f"Missing flow checkpoint: {flow_path}")

    flow = FlowModule(
        latent_dim, num_layers=FLOW_NUM_LAYERS, hidden_dim=FLOW_HIDDEN_DIM
    )
    flow.load_state_dict(torch.load(flow_path, map_location="cpu"))
    flow.eval()

    flow_latent = _flow_latent_trajectory(flow, latent)

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(flow_latent)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    labels = kmeans.fit_predict(flow_latent)

    time_index = np.arange(len(latent_2d))

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot(
        latent_2d[:, 0],
        latent_2d[:, 1],
        color="#cbd5e1",
        linewidth=1.0,
        alpha=0.65,
        zorder=1,
    )
    scatter = ax.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=time_index,
        cmap="viridis",
        s=10,
        alpha=0.95,
        zorder=2,
    )

    arrow_indices = np.arange(0, len(latent_2d) - 1, arrow_step)
    for start_idx in arrow_indices:
        end_idx = min(start_idx + arrow_step, len(latent_2d) - 1)
        ax.annotate(
            "",
            xy=(latent_2d[end_idx, 0], latent_2d[end_idx, 1]),
            xytext=(latent_2d[start_idx, 0], latent_2d[start_idx, 1]),
            arrowprops=dict(
                arrowstyle="->",
                color="#111827",
                lw=1.2,
                alpha=0.75,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )

    ax.set_title("Flow Latent Trajectory with Direction Arrows")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Time step")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PCA-projected flow latent trajectory colored by KMeans labels."
    )
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument(
        "--arrows",
        action="store_true",
        help="Also generate a direction-arrow version of the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_flow_latent_segmentation(n_clusters=args.clusters)
    if args.arrows:
        plot_flow_latent_segmentation_with_arrows(n_clusters=args.clusters)


if __name__ == "__main__":
    main()
