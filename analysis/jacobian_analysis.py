from pathlib import Path
import sys
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 让我们能 import model/world_model.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "model"))

from world_model import WorldModel

MODEL_PATH = ROOT / "model" / "world_model.pth"
TRAJ_PATH = ROOT / "data" / "trajectories.npy"
SAVE_JACOBIAN = ROOT / "analysis" / "jacobian.npy"
SAVE_FIG = ROOT / "analysis" / "plots" / "jacobian_heatmap.png"
SAVE_ENTROPY_FIG = (
    ROOT / "report" / "figures" / "entropy" / "fig9a_predictive_entropy_spike.png"
)
SAVE_ENTROPY = ROOT / "analysis" / "predictive_entropy.npy"
ENTROPY_WINDOW = 21


def _rolling_mean(values, window=ENTROPY_WINDOW):
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.size < window:
        return values

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        return values * 0.0
    return (values - mean) / std


def _predictive_entropy_proxy(model, x, y, window=ENTROPY_WINDOW):
    with torch.no_grad():
        pred, _ = model(x)

    residual = (pred - y).cpu().numpy()
    if residual.ndim == 1:
        residual = residual[:, None]

    residual_var = np.stack(
        [
            _rolling_mean(residual[:, dim] ** 2, window=window)
            for dim in range(residual.shape[1])
        ],
        axis=-1,
    )
    residual_var = np.clip(residual_var, 1e-8, None)

    entropy = 0.5 * np.log(2.0 * math.pi * math.e * residual_var).sum(axis=-1)
    return entropy, residual


def compute_jacobian(
    model_path=MODEL_PATH,
    traj_path=TRAJ_PATH,
    latent_dim=16,
    save_jacobian=SAVE_JACOBIAN,
    save_fig=SAVE_FIG,
):
    # 加载轨迹
    traj = np.load(traj_path)  # shape (T, 1)
    x = torch.tensor(traj[:-1], dtype=torch.float32)  # (T-1, 1)
    y = torch.tensor(traj[1:], dtype=torch.float32)  # (T-1, 1)

    # 加载模型
    input_dim = x.shape[1] if x.ndim > 1 else 1
    model = WorldModel(state_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    encoder = model.encoder  # 我们只关心 encoder 的 Jacobian

    jac_list = []

    for t in range(x.shape[0]):
        xt = x[t : t + 1].clone().detach().requires_grad_(True)  # shape (1, 1)
        ht = encoder(xt)  # shape (1, latent_dim)

        grads = []
        for i in range(latent_dim):
            encoder.zero_grad()
            if xt.grad is not None:
                xt.grad.zero_()  #

            hi = ht[0, i]
            grad = torch.autograd.grad(hi, xt, retain_graph=True)[0]
            grads.append(grad.squeeze(0).detach().cpu().numpy())

        jac_list.append(grads)

    jacobian = np.array(jac_list)  # shape (T-1, latent_dim, input_dim)
    np.save(save_jacobian, jacobian)
    print(f"Saved {save_jacobian}, shape = {jacobian.shape}")

    if jacobian.ndim == 3:
        jacobian_norm = np.linalg.norm(jacobian, axis=(1, 2))
        jacobian_heatmap = np.linalg.norm(jacobian, axis=-1)
    else:
        jacobian_norm = np.linalg.norm(jacobian, axis=1)
        jacobian_heatmap = jacobian

    predictive_entropy, _ = _predictive_entropy_proxy(model, x, y)
    np.save(SAVE_ENTROPY, predictive_entropy)
    print(f"Saved {SAVE_ENTROPY}, shape = {predictive_entropy.shape}")

    fig_entropy, ax_entropy = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax_entropy.plot(
        _zscore(jacobian_norm),
        color="#2563eb",
        linewidth=1.6,
        label="Jacobian norm (z-scored)",
    )
    ax_entropy.plot(
        _zscore(predictive_entropy),
        color="#d97706",
        linewidth=1.6,
        label="Predictive entropy proxy (z-scored)",
    )
    ax_entropy.set_xlabel("Time step")
    ax_entropy.set_ylabel("standardized score")
    ax_entropy.set_title("Jacobian norm and predictive entropy proxy")
    ax_entropy.grid(True, alpha=0.25)
    ax_entropy.legend(loc="best")

    try:
        traj = np.load(traj_path)
        if traj.ndim == 1 or traj.shape[1] == 1:
            traj_1d = traj.squeeze()
            vel = np.diff(traj_1d)
            coll_idx = np.where(np.sign(vel[:-1]) != np.sign(vel[1:]))[0] + 1
        else:
            coll_idx = (
                np.where(np.all(np.isclose(np.diff(traj, axis=0), 0.0), axis=1))[0] + 1
            )
        coll_idx = coll_idx[(coll_idx >= 0) & (coll_idx < jacobian.shape[0])]
        for ci in coll_idx:
            ax_entropy.axvline(
                ci, color="#111827", linestyle="--", linewidth=0.8, alpha=0.35
            )
    except Exception:
        pass

    fig_entropy.savefig(SAVE_ENTROPY_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig_entropy)
    print(f"Saved {SAVE_ENTROPY_FIG}")

    # 画热力图
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(
        jacobian_heatmap.T, aspect="auto", cmap="bwr", interpolation="nearest"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("d h_i / d x")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Latent dimension")
    ax.set_title("Jacobian of encoder(h) w.r.t input x")

    # detect collisions from trajectory and overlay vertical lines
    try:
        traj = np.load(traj_path)
        if traj.ndim == 1 or traj.shape[1] == 1:
            traj_1d = traj.squeeze()
            vel = np.diff(traj_1d)
            coll_idx = np.where(np.sign(vel[:-1]) != np.sign(vel[1:]))[0] + 1
        else:
            coll_idx = (
                np.where(np.all(np.isclose(np.diff(traj, axis=0), 0.0), axis=1))[0] + 1
            )
        coll_idx = coll_idx[(coll_idx >= 0) & (coll_idx < jacobian.shape[0])]
        for ci in coll_idx:
            ax.axvline(ci, color="k", linestyle="--", linewidth=0.8, alpha=0.8)
    except Exception:
        coll_idx = np.array([])

    # inset: summed absolute sensitivity around a representative collision
    if coll_idx.size > 0:
        first = coll_idx[0]
        lo = max(0, first - 20)
        hi = min(jacobian.shape[0], first + 21)
        ax_in = inset_axes(ax, width="30%", height="35%", loc="upper right")
        ax_in.plot(
            range(lo, hi),
            np.sum(np.abs(jacobian[lo:hi]), axis=tuple(range(1, jacobian.ndim))),
            "-o",
            markersize=3,
        )
        ax_in.set_title("Inset: |Jacobian| sum", fontsize=9)
        ax_in.set_xlabel("time", fontsize=8)
        ax_in.set_ylabel("sum |d h / d x|", fontsize=8)
        ax_in.tick_params(axis="both", labelsize=7)

    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    print(f"Saved {save_fig}")


if __name__ == "__main__":
    compute_jacobian()
