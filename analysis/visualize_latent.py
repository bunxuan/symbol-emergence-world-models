from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATES_PATH = PROJECT_ROOT / "data" / "trajectories_2d.npy"
DEFAULT_FIG_PATH = (
    PROJECT_ROOT / "report" / "figures" / "gridworld" / "fig10_gridworld_states.png"
)


def plot_2d_states(states, boundaries=None, save_path=None):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 2:
        raise ValueError(f"Expected states with shape (N, 2), got {states.shape}")

    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    time_colors = np.linspace(0.0, 1.0, len(states))
    ax.scatter(
        states[:, 0],
        states[:, 1],
        c=time_colors,
        s=10,
        cmap="viridis",
        alpha=0.9,
        linewidths=0,
    )

    if boundaries is not None:
        boundaries = np.asarray(boundaries, dtype=np.float32)
        if boundaries.size > 0:
            ax.scatter(
                boundaries[:, 0],
                boundaries[:, 1],
                c="#dc2626",
                s=28,
                label="boundary",
                edgecolors="white",
                linewidths=0.4,
            )

    ax.set_title("2D GridWorld Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    if boundaries is not None and len(boundaries) > 0:
        ax.legend(loc="best")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")

    return fig, ax


def _load_boundary_states(states):
    if states.ndim != 2 or states.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float32)

    eps = 1e-6
    edge_mask = np.any(
        np.isclose(states, 0.0, atol=eps)
        | np.isclose(states, states.max(axis=0, keepdims=True), atol=0.15),
        axis=1,
    )
    return states[edge_mask]


def main():
    if not DEFAULT_STATES_PATH.exists():
        print(f"Missing {DEFAULT_STATES_PATH}; run envs/collect_2d.py first.")
        return

    states = np.load(DEFAULT_STATES_PATH)
    boundaries = _load_boundary_states(states)
    plot_2d_states(states, boundaries=boundaries, save_path=DEFAULT_FIG_PATH)


if __name__ == "__main__":
    main()
