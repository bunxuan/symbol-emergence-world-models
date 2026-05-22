"""Flow Jacobian time-series analysis."""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from manifold_plot import _load_latent
from world_model import FlowModule

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
LATENT_PATH = MODEL_DIR / "latent.npy"
FLOW_MODEL_PATH = MODEL_DIR / "flow.pth"
TRAJ_PATH = ROOT / "data" / "trajectories.npy"
SAVE_FIG = ROOT / "analysis" / "plots" / "flow_jacobian_timeseries.png"
FLOW_NUM_LAYERS = 8
FLOW_HIDDEN_DIM = 128
SMOOTH_WINDOW = 21


def _find_collision_indices(traj):
    traj = np.asarray(traj).reshape(-1)
    if traj.size < 3:
        return np.array([], dtype=int)

    deltas = np.diff(traj)
    sign_changes = np.where(np.sign(deltas[:-1]) != np.sign(deltas[1:]))[0] + 1
    return sign_changes.astype(int)


def _flow_jacobian_norms(flow, latent, use_log=True):
    flow.eval()
    norms = []

    def flow_map(sample):
        z, _ = flow.flow(sample.unsqueeze(0), reverse=False)
        return z.squeeze(0)

    for sample in latent:
        sample = torch.as_tensor(sample, dtype=torch.float32)
        jacobian = torch.autograd.functional.jacobian(flow_map, sample, vectorize=True)
        norms.append(torch.linalg.norm(jacobian, ord="fro").item())

    norms = np.asarray(norms, dtype=np.float32)
    if use_log:
        return np.log(norms + 1e-12)
    return norms


def _smooth_series(values, window=SMOOTH_WINDOW):
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.size < window:
        return values

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def plot_flow_jacobian_timeseries(
    latent_path=LATENT_PATH,
    flow_path=FLOW_MODEL_PATH,
    traj_path=TRAJ_PATH,
    save_path=SAVE_FIG,
    use_log=True,
):
    latent = _load_latent(latent_path)
    latent_dim = latent.shape[1]

    if not flow_path.exists():
        raise FileNotFoundError(f"Missing flow checkpoint: {flow_path}")

    traj = np.load(traj_path).astype(np.float32).reshape(-1)
    collision_indices = _find_collision_indices(traj)

    flow = FlowModule(
        latent_dim, num_layers=FLOW_NUM_LAYERS, hidden_dim=FLOW_HIDDEN_DIM
    )
    flow.load_state_dict(torch.load(flow_path, map_location="cpu"))
    flow.eval()

    jacobian_values = _flow_jacobian_norms(flow, latent, use_log=False)
    smoothed_values = _smooth_series(jacobian_values)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        jacobian_values,
        color="#93c5fd",
        linewidth=1.0,
        alpha=0.55,
        label="raw ||J||",
    )
    ax.plot(
        smoothed_values,
        color="#2563eb",
        linewidth=2.0,
        label=f"smoothed ||J|| (window={SMOOTH_WINDOW})",
    )

    for idx, collision_index in enumerate(collision_indices):
        ax.axvline(
            collision_index,
            color="#dc2626",
            alpha=0.12,
            linewidth=1.0,
            label="collision" if idx == 0 else None,
        )

    if len(collision_indices) > 0:
        clipped_indices = np.clip(collision_indices, 0, len(jacobian_values) - 1)
        ax.scatter(
            clipped_indices,
            jacobian_values[clipped_indices],
            s=10,
            color="#dc2626",
            alpha=0.35,
            zorder=3,
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$\|J(x_t)\|_F$")
    ax.set_title("Flow Jacobian norm over time")
    ax.grid(True, alpha=0.2)
    if len(collision_indices) > 0:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot flow Jacobian over time.")
    parser.add_argument(
        "--mode",
        choices=["jacobian"],
        default="jacobian",
        help="Only one mode is currently available.",
    )
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    plot_flow_jacobian_timeseries()


if __name__ == "__main__":
    main()
