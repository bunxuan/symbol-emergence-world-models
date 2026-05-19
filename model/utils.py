"""Shared model utilities."""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_latent(latent_path: Path | None = None):
    if latent_path is None:
        latent_path = MODEL_DIR / "latent.npy"
    return torch.from_numpy(__import__("numpy").load(latent_path).astype("float32"))


"""Model utilities."""
