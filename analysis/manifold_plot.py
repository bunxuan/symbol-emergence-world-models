"""Manifold plotting utilities for flow and diffusion experiments."""

from pathlib import Path
import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
LATENT_PATH = MODEL_DIR / "latent.npy"
FLOW_MODEL_PATH = MODEL_DIR / "flow.pth"
FLOW_SAMPLES_PATH = MODEL_DIR / "flow_samples.npy"
DIFFUSION_MODEL_PATH = MODEL_DIR / "diffusion.pth"
DIFFUSION_CHAIN_PATH = MODEL_DIR / "diffusion_chain.npy"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from world_model import DiffusionModule, FlowModule


def _load_latent(latent_path=LATENT_PATH):
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_path}")
    return np.load(latent_path).astype(np.float32)


def _project_pair(real_points, generated_points):
    pca = PCA(n_components=2)
    pca.fit(np.vstack([real_points, generated_points]))
    return pca.transform(real_points), pca.transform(generated_points)


def plot_flow_comparison(
    latent_path=LATENT_PATH,
    flow_path=FLOW_MODEL_PATH,
    sample_path=FLOW_SAMPLES_PATH,
    save_path=RESULTS_DIR / "flow_vs_latent.png",
):
    latent = _load_latent(latent_path)
    latent_dim = latent.shape[1]

    if sample_path.exists():
        samples = np.load(sample_path).astype(np.float32)
    else:
        if not flow_path.exists():
            raise FileNotFoundError(f"Missing flow checkpoint: {flow_path}")

        flow = FlowModule(latent_dim)
        flow.load_state_dict(torch.load(flow_path, map_location="cpu"))
        flow.eval()
        with torch.no_grad():
            samples = flow.sample(len(latent), device=torch.device("cpu")).cpu().numpy()

    latent_2d, samples_2d = _project_pair(latent, samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], s=8, alpha=0.65, color="#2563eb")
    axes[0].set_title("Real latent distribution")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.2)

    density = axes[1].hexbin(
        samples_2d[:, 0],
        samples_2d[:, 1],
        gridsize=40,
        cmap="magma",
        mincnt=1,
    )
    axes[1].set_title("Flow fitted density")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.2)
    fig.colorbar(density, ax=axes[1], label="sample count")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_diffusion_comparison(
    latent_path=LATENT_PATH,
    diffusion_path=DIFFUSION_MODEL_PATH,
    chain_path=DIFFUSION_CHAIN_PATH,
    save_path=RESULTS_DIR / "diffusion_vs_real.png",
):
    latent = _load_latent(latent_path)
    latent_dim = latent.shape[1]

    if chain_path.exists():
        chain = np.load(chain_path).astype(np.float32)
    else:
        if not diffusion_path.exists():
            raise FileNotFoundError(f"Missing diffusion checkpoint: {diffusion_path}")

        diffusion = DiffusionModule(latent_dim)
        diffusion.load_state_dict(torch.load(diffusion_path, map_location="cpu"))
        diffusion.eval()
        with torch.no_grad():
            _, chain_tensor = diffusion.sample(
                1, device=torch.device("cpu"), return_chain=True
            )
        chain = chain_tensor[:, 0, :].numpy()

    latent_2d, chain_2d = _project_pair(latent, chain)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(latent_2d[:, 0], latent_2d[:, 1], color="#2563eb", linewidth=1.2)
    axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], s=8, alpha=0.55, color="#2563eb")
    axes[0].set_title("Real latent trajectory")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(chain_2d[:, 0], chain_2d[:, 1], color="#d97706", linewidth=1.2)
    axes[1].scatter(chain_2d[:, 0], chain_2d[:, 1], s=8, alpha=0.55, color="#d97706")
    axes[1].set_title("Diffusion reverse chain")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot latent manifolds for flow or diffusion."
    )
    parser.add_argument("--mode", choices=["flow", "diffusion"], default="flow")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "flow":
        plot_flow_comparison()
    else:
        plot_diffusion_comparison()


if __name__ == "__main__":
    main()
