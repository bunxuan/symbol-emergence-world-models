"""Training entry point."""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[0]
DATA_PATH = PROJECT_ROOT / "data" / "trajectories.npy"
LATENT_PATH = PROJECT_ROOT / "model" / "latent.npy"
MODEL_PATH = PROJECT_ROOT / "model" / "world_model.pth"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from world_model import WorldModel


def train_world_model(
    data_path=DATA_PATH, latent_dim=16, epochs=50, batch_size=32, lr=1e-3
):
    traj = np.load(data_path)  # shape (T, 1)
    x = traj[:-1]
    y = traj[1:]

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = WorldModel(latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            pred, _ = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)

    # 保存 latent（用于分析）
    with torch.no_grad():
        _, h = model(x)
        np.save(LATENT_PATH, h.numpy())

    print("Saved", MODEL_PATH, "and", LATENT_PATH)


if __name__ == "__main__":
    train_world_model()
