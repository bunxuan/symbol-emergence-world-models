from pathlib import Path
import sys
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "model"))
if str(PROJECT_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from envs.collect_2d import collect_trajectory_2d
from envs.gridworld import GridWorld
from world_model import WorldModel
from mi import compute_symbol_mi
from spatial_maps import plot_spatial_map

MODEL_PATH = PROJECT_ROOT / "model" / "world_model.pth"
LATENT_DIM = 16
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures" / "gridworld"


def compute_jacobian_norm(model, states):
    states = torch.as_tensor(states, dtype=torch.float32)
    norms = []

    for idx in range(states.shape[0]):
        state = states[idx : idx + 1].clone().detach().requires_grad_(True)
        latent = model.encode(state)
        grads = []
        for latent_idx in range(latent.shape[1]):
            grad = torch.autograd.grad(latent[0, latent_idx], state, retain_graph=True)[
                0
            ]
            grads.append(grad.squeeze(0))
        jacobian = torch.stack(grads, dim=0)
        norms.append(torch.linalg.norm(jacobian).item())

    return np.asarray(norms, dtype=np.float32)


def compute_predictive_entropy(model, states):
    states = torch.as_tensor(states, dtype=torch.float32)
    with torch.no_grad():
        pred, _ = model(states[:-1])
        residual = pred - states[1:]

    variance = residual.pow(2).mean(dim=1).clamp_min(1e-8)
    entropy = 0.5 * torch.log(2.0 * math.pi * math.e * variance)
    return entropy.cpu().numpy()


def cluster_latent(z, n_clusters=6):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    return kmeans.fit_predict(z)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained world model: {MODEL_PATH}")

    env = GridWorld(size=10)
    states, actions = collect_trajectory_2d(env, steps=500, normalize=False)
    states_norm = states / float(env.size)

    model = WorldModel(state_dim=2, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    z = model.encode(torch.from_numpy(states_norm))
    jacobian_norm = compute_jacobian_norm(model, states_norm)
    entropy = compute_predictive_entropy(model, states_norm)
    clusters = cluster_latent(z.detach().cpu().numpy(), n_clusters=6)
    mi_matrix = compute_symbol_mi(clusters)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_spatial_map(
        states,
        jacobian_norm,
        title="Jacobian norm in 2D",
        save_path=OUTPUT_DIR / "fig10a_jacobian_norm_2d.png",
    )
    plot_spatial_map(
        states[1:],
        entropy,
        title="Predictive entropy in 2D",
        save_path=OUTPUT_DIR / "fig10b_predictive_entropy_2d.png",
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(states[:, 0], states[:, 1], c=clusters, cmap="tab10", s=25)
    plt.title("Symbolic clusters in 2D")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.savefig(
        OUTPUT_DIR / "fig10c_symbolic_clusters_2d.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(mi_matrix, cmap="magma")
    plt.colorbar()
    plt.title("Symbol MI matrix")
    plt.savefig(
        OUTPUT_DIR / "fig10d_symbol_mi_matrix_2d.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    np.save(PROJECT_ROOT / "analysis" / "jacobian_norm_2d.npy", jacobian_norm)
    np.save(PROJECT_ROOT / "analysis" / "predictive_entropy_2d.npy", entropy)
    np.save(PROJECT_ROOT / "analysis" / "latent_2d.npy", z.detach().cpu().numpy())
    np.save(PROJECT_ROOT / "analysis" / "clusters_2d.npy", clusters)
    np.save(PROJECT_ROOT / "analysis" / "symbol_mi_2d.npy", mi_matrix)


if __name__ == "__main__":
    main()
