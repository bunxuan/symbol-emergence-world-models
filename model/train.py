"""Training entry point."""

from pathlib import Path
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[0]
DATA_PATH = PROJECT_ROOT / "data" / "trajectories.npy"
LATENT_PATH = MODEL_DIR / "latent.npy"
WORLD_MODEL_PATH = MODEL_DIR / "world_model.pth"
FLOW_MODEL_PATH = MODEL_DIR / "flow.pth"
DIFFUSION_MODEL_PATH = MODEL_DIR / "diffusion.pth"
FLOW_SAMPLES_PATH = MODEL_DIR / "flow_samples.npy"
DIFFUSION_SAMPLES_PATH = MODEL_DIR / "diffusion_samples.npy"
DIFFUSION_CHAIN_PATH = MODEL_DIR / "diffusion_chain.npy"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from world_model import DiffusionModule, FlowModule, WorldModel

DEFAULT_EPOCHS = {
    "world": 50,
    "flow": 300,
    "diffusion": 1000,
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_latent_array(latent_path=LATENT_PATH):
    if not latent_path.exists():
        raise FileNotFoundError(
            f"Missing latent file: {latent_path}. Run `python model/train.py --mode world` first."
        )
    return np.load(latent_path).astype(np.float32)


def train_world_model(
    data_path=DATA_PATH,
    latent_dim=16,
    epochs=50,
    batch_size=32,
    lr=1e-3,
):
    traj = np.load(data_path).astype(np.float32)  # shape (T, 1)
    x = torch.from_numpy(traj[:-1])
    y = torch.from_numpy(traj[1:])

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = _device()
    model = WorldModel(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred, _ = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Loss = {total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), WORLD_MODEL_PATH)

    model.eval()
    with torch.no_grad():
        _, h = model(x.to(device))
        latent = h.cpu().numpy()
        np.save(LATENT_PATH, latent)

    print(f"Saved {WORLD_MODEL_PATH} and {LATENT_PATH}")
    return model, latent


def train_flow(
    latent_path=LATENT_PATH,
    epochs=1000,
    batch_size=128,
    lr=5e-4,
    num_layers=8,
    hidden_dim=128,
):
    latent = _load_latent_array(latent_path)
    latent_dim = latent.shape[1]

    dataset = TensorDataset(torch.from_numpy(latent))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = _device()
    model = FlowModule(latent_dim, num_layers=num_layers, hidden_dim=hidden_dim).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()   
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch_h,) in loader:
            batch_h = batch_h.to(device)
            z, log_det = model(batch_h)
            base_log_prob = -0.5 * (z**2).sum(dim=-1) - 0.5 * latent_dim * math.log(
                2.0 * math.pi
            )
            loss = -(base_log_prob + log_det).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_h.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Flow NLL = {total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), FLOW_MODEL_PATH)

    model.eval()
    with torch.no_grad():
        samples = model.sample(len(latent), device=device).cpu().numpy()
        np.save(FLOW_SAMPLES_PATH, samples)

    print(f"Saved {FLOW_MODEL_PATH} and {FLOW_SAMPLES_PATH}")
    return model, samples


def train_diffusion(
    latent_path=LATENT_PATH,
    epochs=1000,
    batch_size=128,
    lr=5e-4,
    timesteps=1000,
    hidden_dim=256,
    time_embed_dim=128,
):
    latent = _load_latent_array(latent_path)
    latent_dim = latent.shape[1]
    latent_mean = latent.mean(axis=0, keepdims=True)
    latent_std = latent.std(axis=0, keepdims=True)
    latent_std = np.where(latent_std < 1e-6, 1.0, latent_std)
    latent_norm = (latent - latent_mean) / latent_std

    dataset = TensorDataset(torch.from_numpy(latent_norm))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = _device()
    model = DiffusionModule(
        latent_dim,
        timesteps=timesteps,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch_h,) in loader:
            batch_h = batch_h.to(device)
            loss = model.loss(batch_h)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_h.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs}, Diffusion MSE = {total_loss / len(dataset):.4f}"
        )

    torch.save(model.state_dict(), DIFFUSION_MODEL_PATH)

    model.eval()
    with torch.no_grad():
        samples = model.sample(len(latent), device=device).cpu().numpy()
        samples = samples * latent_std + latent_mean
        _, chain = model.sample(1, device=device, return_chain=True)
        chain = chain[:, 0, :].numpy()
        chain = chain * latent_std + latent_mean
        np.save(DIFFUSION_SAMPLES_PATH, samples)
        np.save(DIFFUSION_CHAIN_PATH, chain)

    print(
        f"Saved {DIFFUSION_MODEL_PATH}, {DIFFUSION_SAMPLES_PATH}, and {DIFFUSION_CHAIN_PATH}"
    )
    return model, samples, chain


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train world, flow, or diffusion models."
    )
    parser.add_argument(
        "--mode", choices=["world", "flow", "diffusion"], default="world"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "world":
        train_world_model(
            latent_dim=args.latent_dim,
            epochs=args.epochs or DEFAULT_EPOCHS["world"],
            batch_size=args.batch_size or 32,
            lr=args.lr,
        )
    elif args.mode == "flow":
        train_flow(
            epochs=args.epochs or DEFAULT_EPOCHS["flow"],
            batch_size=args.batch_size or 128,
            lr=args.lr,
        )
    elif args.mode == "diffusion":
        train_diffusion(
            epochs=args.epochs or DEFAULT_EPOCHS["diffusion"],
            batch_size=args.batch_size or 128,
            lr=args.lr,
            timesteps=args.timesteps,
        )


if __name__ == "__main__":
    main()
