import matplotlib.pyplot as plt
import numpy as np


def plot_spatial_map(
    states, values, title="Spatial map", cmap="viridis", save_path=None
):
    """
    states: (N, 2) array of [x, y] positions
    values: (N,) array of scalar values (Jacobian norm, entropy, etc.)
    """
    states = np.asarray(states, dtype=np.float32)
    values = np.asarray(values)

    if states.ndim != 2 or states.shape[1] != 2:
        raise ValueError(f"Expected states with shape (N, 2), got {states.shape}")
    if values.ndim != 1 or values.shape[0] != states.shape[0]:
        raise ValueError(
            f"Expected values with shape (N,), got {values.shape} for {states.shape[0]} states"
        )

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(states[:, 0], states[:, 1], c=values, cmap=cmap, s=30)
    plt.colorbar(sc, label="value")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
