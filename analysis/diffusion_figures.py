"""Generate the diffusion figures used in the report."""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from manifold_plot import _load_latent, _project_pair
from world_model import DiffusionModule

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
REPORT_FIG_DIR = ROOT / "report" / "figures"
LATENT_PATH = MODEL_DIR / "latent.npy"
DIFFUSION_MODEL_PATH = MODEL_DIR / "diffusion.pth"
DIFFUSION_CHAIN_PATH = MODEL_DIR / "diffusion_chain.npy"
DIFFUSION_TIMESTEPS = 1000
DIFFUSION_HIDDEN_DIM = 256
DIFFUSION_TIME_EMBED_DIM = 128

FIG7A_PATH = REPORT_FIG_DIR / "fig7a_diffusion_reverse_chain.png"
FIG7B_PATH = REPORT_FIG_DIR / "fig7b_diffusion_segmentation.png"
FIG7C_PATH = REPORT_FIG_DIR / "fig7c_diffusion_vs_real_latent_comparison.png"


def _load_chain(
    chain_path=DIFFUSION_CHAIN_PATH,
    diffusion_path=DIFFUSION_MODEL_PATH,
    latent_path=LATENT_PATH,
):
    if chain_path.exists():
        return np.load(chain_path).astype(np.float32)

    latent = _load_latent(latent_path)
    latent_dim = latent.shape[1]
    if not diffusion_path.exists():
        raise FileNotFoundError(f"Missing diffusion checkpoint: {diffusion_path}")

    diffusion = DiffusionModule(
        latent_dim,
        timesteps=DIFFUSION_TIMESTEPS,
        hidden_dim=DIFFUSION_HIDDEN_DIM,
        time_embed_dim=DIFFUSION_TIME_EMBED_DIM,
    )
    diffusion.load_state_dict(torch.load(diffusion_path, map_location="cpu"))
    diffusion.eval()
    with torch.no_grad():
        _, chain_tensor = diffusion.sample(
            1, device=torch.device("cpu"), return_chain=True
        )
    chain = chain_tensor[:, 0, :].numpy().astype(np.float32)
    np.save(chain_path, chain)
    return chain


def _time_gradient_scatter(ax, points, title, cmap="plasma"):
    time_index = np.arange(len(points))
    ax.plot(points[:, 0], points[:, 1], color="#cbd5e1", linewidth=1.0, alpha=0.6)
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=time_index,
        cmap=cmap,
        s=9,
        alpha=0.95,
        zorder=2,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    return scatter


def plot_fig7a_reverse_chain(
    save_path=FIG7A_PATH,
    chain_path=DIFFUSION_CHAIN_PATH,
):
    chain = _load_chain(chain_path=chain_path)
    latent = _load_latent()
    latent_2d, chain_2d = _project_pair(latent, chain)

    fig, ax = plt.subplots(figsize=(5.1, 4.4))
    scatter = _time_gradient_scatter(ax, chain_2d, "Diffusion reverse chain")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="time step")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_fig7b_diffusion_segmentation(
    save_path=FIG7B_PATH,
    chain_path=DIFFUSION_CHAIN_PATH,
    n_clusters=4,
):
    chain = _load_chain(chain_path=chain_path)
    latent = _load_latent()
    latent_2d, chain_2d = _project_pair(latent, chain)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    labels = kmeans.fit_predict(chain)

    fig, ax = plt.subplots(figsize=(5.1, 4.4))
    ax.plot(chain_2d[:, 0], chain_2d[:, 1], color="#cbd5e1", linewidth=1.0, alpha=0.7)
    scatter = ax.scatter(
        chain_2d[:, 0],
        chain_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=9,
        alpha=0.95,
        zorder=2,
    )
    ax.set_title("Diffusion segmentation")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    legend = ax.legend(
        *scatter.legend_elements(), title="Cluster", loc="best", frameon=True
    )
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_fig7c_diffusion_vs_real_comparison(
    save_path=FIG7C_PATH,
    chain_path=DIFFUSION_CHAIN_PATH,
):
    latent = _load_latent()
    chain = _load_chain(chain_path=chain_path)
    latent_2d, chain_2d = _project_pair(latent, chain)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))

    axes[0].plot(latent_2d[:, 0], latent_2d[:, 1], color="#2563eb", linewidth=1.2)
    axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], s=8, alpha=0.6, color="#2563eb")
    axes[0].set_title("Real latent")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(chain_2d[:, 0], chain_2d[:, 1], color="#d97706", linewidth=1.2)
    axes[1].scatter(chain_2d[:, 0], chain_2d[:, 1], s=8, alpha=0.6, color="#d97706")
    axes[1].set_title("Diffusion reverse chain")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("Diffusion vs Real Latent", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def generate_all():
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig7a_reverse_chain()
    plot_fig7b_diffusion_segmentation()
    plot_fig7c_diffusion_vs_real_comparison()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate diffusion report figures.")
    parser.add_argument("--clusters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    generate_all()


if __name__ == "__main__":
    main()
