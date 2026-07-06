from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "model"))
if str(PROJECT_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from world_model import WorldModel
from spatial_maps import plot_spatial_map

MODEL_PATH = PROJECT_ROOT / "model" / "world_model.pth"
STATES_PATH = PROJECT_ROOT / "data" / "trajectories_2d.npy"
LATENT_DIM = 16
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures" / "gridworld"


def cluster_latent(z, n_clusters=6):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    return kmeans.fit_predict(z)


def main():
    if not STATES_PATH.exists():
        raise FileNotFoundError(f"Missing 2D trajectory file: {STATES_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained world model: {MODEL_PATH}")

    states = np.load(STATES_PATH).astype(np.float32)
    model = WorldModel(state_dim=2, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    z = model.encode(torch.from_numpy(states)).detach().cpu().numpy()
    clusters = cluster_latent(z, n_clusters=6)

    plot_spatial_map(
        states,
        clusters,
        title="Symbolic clusters in 2D space",
        cmap="tab10",
        save_path=OUTPUT_DIR / "fig10c_symbolic_clusters_2d.png",
    )

    np.save(PROJECT_ROOT / "analysis" / "clusters_2d.npy", clusters)


if __name__ == "__main__":
    main()
