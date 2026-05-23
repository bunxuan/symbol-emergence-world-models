from pathlib import Path
import sys

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

    # 加载模型
    model = WorldModel(latent_dim=latent_dim)
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
                xt.grad.zero_()

            hi = ht[0, i]
            hi.backward(retain_graph=True)

            grads.append(xt.grad.item())  # 因为输入是 1D

        jac_list.append(grads)

    jacobian = np.array(jac_list)  # shape (T-1, latent_dim)
    np.save(save_jacobian, jacobian)
    print(f"Saved {save_jacobian}, shape = {jacobian.shape}")

    # 画热力图
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(jacobian.T, aspect="auto", cmap="bwr", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("d h_i / d x")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Latent dimension")
    ax.set_title("Jacobian of encoder(h) w.r.t input x")

    # detect collisions from trajectory and overlay vertical lines
    try:
        traj = np.load(traj_path).squeeze()
        vel = np.diff(traj)
        coll_idx = np.where(np.sign(vel[:-1]) != np.sign(vel[1:]))[0] + 1
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
        ax_in = inset_axes(ax, width="30%", height="35%", loc='upper right')
        ax_in.plot(range(lo, hi), np.sum(np.abs(jacobian[lo:hi]), axis=1), "-o", markersize=3)
        ax_in.set_title("Inset: |Jacobian| sum", fontsize=9)
        ax_in.set_xlabel("time", fontsize=8)
        ax_in.set_ylabel("sum |d h / d x|", fontsize=8)
        ax_in.tick_params(axis="both", labelsize=7)

    plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    print(f"Saved {save_fig}")


if __name__ == "__main__":
    compute_jacobian()
