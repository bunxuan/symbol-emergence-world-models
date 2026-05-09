from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# 让我们能 import model/world_model.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "model"))

from world_model import WorldModel

MODEL_PATH = ROOT / "model" / "world_model.pth"
TRAJ_PATH = ROOT / "data" / "trajectories.npy"
SAVE_JACOBIAN = ROOT / "analysis" / "jacobian.npy"
SAVE_FIG = ROOT / "results" / "jacobian_heatmap.png"


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
    plt.figure(figsize=(10, 4))
    plt.imshow(jacobian.T, aspect="auto", cmap="bwr", interpolation="nearest")
    plt.colorbar(label="d h_i / d x")
    plt.xlabel("Time step")
    plt.ylabel("Latent dimension")
    plt.title("Jacobian of encoder(h) w.r.t input x")

    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    print(f"Saved {save_fig}")


if __name__ == "__main__":
    compute_jacobian()
