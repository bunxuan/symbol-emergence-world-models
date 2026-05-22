"""Compare MLP and flow latent trajectories with PCA and Procrustes alignment."""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

from manifold_plot import _load_latent
from world_model import FlowModule

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
LATENT_PATH = MODEL_DIR / "latent.npy"
FLOW_MODEL_PATH = MODEL_DIR / "flow.pth"
SAVE_FIG = ROOT / "analysis" / "plots" / "latent_structure_comparison.png"
FLOW_NUM_LAYERS = 8
FLOW_HIDDEN_DIM = 128


def _flow_latent_trajectory(flow, latent):
    flow.eval()
    with torch.no_grad():
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        flow_latent, _ = flow.flow(latent_tensor, reverse=False)
    return flow_latent.cpu().numpy().astype(np.float32)


def _pca_2d(points):
    return PCA(n_components=2).fit_transform(points)


def _format_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")


def plot_latent_structure_comparison(
    latent_path=LATENT_PATH,
    flow_path=FLOW_MODEL_PATH,
    save_path=SAVE_FIG,
):
    mlp_latent = _load_latent(latent_path)
    latent_dim = mlp_latent.shape[1]

    if not flow_path.exists():
        raise FileNotFoundError(f"Missing flow checkpoint: {flow_path}")

    flow = FlowModule(
        latent_dim, num_layers=FLOW_NUM_LAYERS, hidden_dim=FLOW_HIDDEN_DIM
    )
    flow.load_state_dict(torch.load(flow_path, map_location="cpu"))
    flow.eval()

    flow_latent = _flow_latent_trajectory(flow, mlp_latent)

    mlp_2d = _pca_2d(mlp_latent)
    flow_2d = _pca_2d(flow_latent)
    mlp_aligned, flow_aligned, disparity = procrustes(mlp_2d, flow_2d)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].plot(mlp_2d[:, 0], mlp_2d[:, 1], color="#94a3b8", linewidth=1.0, alpha=0.8)
    axes[0].scatter(mlp_2d[:, 0], mlp_2d[:, 1], s=10, color="#2563eb", alpha=0.9)
    _format_axes(axes[0], "MLP latent (PCA)")

    axes[1].plot(
        flow_aligned[:, 0],
        flow_aligned[:, 1],
        color="#94a3b8",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[1].scatter(
        flow_aligned[:, 0], flow_aligned[:, 1], s=10, color="#d97706", alpha=0.9
    )
    _format_axes(
        axes[1], f"Aligned flow latent (Procrustes)\nDisparity = {disparity:.4f}"
    )

    axes[2].plot(
        flow_2d[:, 0], flow_2d[:, 1], color="#94a3b8", linewidth=1.0, alpha=0.8
    )
    axes[2].scatter(flow_2d[:, 0], flow_2d[:, 1], s=10, color="#dc2626", alpha=0.9)
    _format_axes(axes[2], "Flow latent (PCA)")

    fig.suptitle("Flow and MLP Latent Structure", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MLP and flow latent trajectories using PCA and Procrustes alignment."
    )
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    plot_latent_structure_comparison()


if __name__ == "__main__":
    main()
