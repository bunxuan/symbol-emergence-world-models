from pathlib import Path
import argparse
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "model"))
if str(PROJECT_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from data.generate_data import generate_1d_trajectory, load_config
from jacobian_analysis import _predictive_entropy_proxy
from mi import compute_symbol_mi
from train import train_world_model

DATA_ROOT = PROJECT_ROOT / "data" / "randomization"
ORIGINAL_DIR = DATA_ROOT / "dataset_original"
RANDOM_DIR = DATA_ROOT / "dataset_random"
RESULT_ROOT = PROJECT_ROOT / "results" / "randomization"
FIG_DIR = RESULT_ROOT / "figures"
COMPARISON_FIG = FIG_DIR / "original_vs_100_shuffled_pca.png"


def _parse_ratios(ratios_text):
    ratios = []
    for token in ratios_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value > 100.0:
            raise ValueError(f"Noise ratio must be in [0, 100], got {value}")
        ratios.append(value)
    if not ratios:
        raise ValueError("At least one noise ratio is required.")
    return sorted(set(ratios))


def _ratio_label(ratio):
    if abs(ratio - round(ratio)) < 1e-9:
        return str(int(round(ratio)))
    return str(ratio).replace(".", "p")


def _save_dataset(path, trajectory):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, trajectory.astype(np.float32))


def _shuffle_trajectory(trajectory, ratio, rng):
    shuffled = np.array(trajectory, copy=True)
    total = shuffled.shape[0]
    if ratio <= 0.0:
        return shuffled
    if ratio >= 100.0:
        indices = np.arange(total)
        rng.shuffle(indices)
        return shuffled[indices]

    count = max(2, int(round((ratio / 100.0) * total)))
    selected = rng.choice(total, size=count, replace=False)
    permuted = np.array(selected, copy=True)
    rng.shuffle(permuted)
    shuffled[selected] = shuffled[permuted]
    return shuffled


def _compute_jacobian_norm(model, x):
    x = torch.as_tensor(x, dtype=torch.float32)
    latent_dim = model.encode(x[:1]).shape[1]
    norms = []

    for idx in range(x.shape[0]):
        xt = x[idx : idx + 1].clone().detach().requires_grad_(True)
        ht = model.encode(xt)
        grads = []
        for latent_idx in range(latent_dim):
            grad = torch.autograd.grad(ht[0, latent_idx], xt, retain_graph=True)[0]
            grads.append(grad.squeeze(0).detach().cpu().numpy())
        jacobian = np.stack(grads, axis=0)
        norms.append(float(np.linalg.norm(jacobian)))

    return np.asarray(norms, dtype=np.float32)


def _pca_scatter(latent, ratio, output_path):
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    colors = np.linspace(0.0, 1.0, latent_2d.shape[0])
    sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, cmap="viridis", s=9)
    fig.colorbar(sc, ax=ax, label="time")
    ax.set_title(f"PCA under shuffle ratio {ratio:.0f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_original_vs_shuffled(original_latent, shuffled_latent, output_path):
    pca = PCA(n_components=2)
    combined = np.concatenate([original_latent, shuffled_latent], axis=0)
    pca.fit(combined)
    original_2d = pca.transform(original_latent)
    shuffled_2d = pca.transform(shuffled_latent)

    original_labels = KMeans(n_clusters=4, n_init=20, random_state=0).fit_predict(
        original_latent
    )
    shuffled_labels = KMeans(n_clusters=4, n_init=20, random_state=0).fit_predict(
        shuffled_latent
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    panels = [
        (axes[0], original_2d, original_labels, "Original trajectory"),
        (axes[1], shuffled_2d, shuffled_labels, "100% shuffled trajectory"),
    ]

    for ax, points, labels, title in panels:
        sc = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=9)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Original vs 100% shuffled trajectory in latent PCA space", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_trends(records, output_path):
    ratios = [item["ratio"] for item in records]
    mse_values = [item["prediction_mse"] for item in records]
    entropy_values = [item["mean_entropy"] for item in records]
    jac_values = [item["mean_jacobian_norm"] for item in records]
    mi_values = [item["adjacent_symbol_mi"] for item in records]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plots = [
        (axes[0, 0], mse_values, "Prediction MSE (lower is better)", "#1f77b4"),
        (axes[0, 1], entropy_values, "Predictive entropy", "#ff7f0e"),
        (axes[1, 0], jac_values, "Jacobian norm", "#2ca02c"),
        (axes[1, 1], mi_values, "Adjacent symbol MI", "#d62728"),
    ]

    for ax, values, title, color in plots:
        ax.plot(ratios, values, marker="o", linewidth=2.0, color=color)
        ax.set_title(title)
        ax.set_xlabel("Shuffle ratio (%)")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_randomization_study(
    ratios,
    seed=0,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    latent_dim=16,
):
    rng = np.random.default_rng(seed)

    config = load_config()
    base_trajectory = generate_1d_trajectory(
        length=config.get("trajectory_length", 1000),
        step_size=config.get("step_size", 0.03),
        x0=config.get("x0", 0.5),
        xmin=config.get("xmin", 0.0),
        xmax=config.get("xmax", 1.0),
    ).astype(np.float32)

    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    original_path = ORIGINAL_DIR / "trajectories.npy"
    _save_dataset(original_path, base_trajectory)

    records = []
    latent_cache = {}
    for ratio in ratios:
        if ratio <= 0.0:
            dataset_path = original_path
        else:
            shuffled = _shuffle_trajectory(base_trajectory, ratio, rng)
            dataset_path = (
                RANDOM_DIR / f"trajectories_shuffle_{_ratio_label(ratio)}.npy"
            )
            _save_dataset(dataset_path, shuffled)

        model, latent = train_world_model(
            data_path=dataset_path,
            state_dim=1,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        trajectory = np.load(dataset_path).astype(np.float32)
        x = torch.from_numpy(trajectory[:-1])
        y = torch.from_numpy(trajectory[1:])

        model.eval()
        with torch.no_grad():
            pred, _ = model(x)
            mse = torch.mean((pred - y) ** 2).item()

        entropy, _ = _predictive_entropy_proxy(model, x, y)
        jacobian_norm = _compute_jacobian_norm(model, x)

        labels = KMeans(n_clusters=4, n_init=20, random_state=0).fit_predict(latent)
        mi_matrix = compute_symbol_mi(labels)
        adjacent_mi = float(mutual_info_score(labels[:-1], labels[1:]))

        ratio_dir = RESULT_ROOT / f"ratio_{_ratio_label(ratio)}"
        ratio_dir.mkdir(parents=True, exist_ok=True)

        np.save(ratio_dir / "trajectory.npy", trajectory)
        np.save(ratio_dir / "latent.npy", latent)
        np.save(ratio_dir / "labels.npy", labels)
        np.save(ratio_dir / "predictive_entropy.npy", entropy)
        np.save(ratio_dir / "jacobian_norm.npy", jacobian_norm)
        np.save(ratio_dir / "symbol_mi_matrix.npy", mi_matrix)

        _pca_scatter(latent, ratio, FIG_DIR / f"pca_ratio_{_ratio_label(ratio)}.png")

        latent_cache[float(ratio)] = np.array(latent, copy=True)

        records.append(
            {
                "ratio": float(ratio),
                "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "prediction_mse": float(mse),
                "mean_entropy": float(np.mean(entropy)),
                "mean_jacobian_norm": float(np.mean(jacobian_norm)),
                "adjacent_symbol_mi": float(adjacent_mi),
            }
        )

    _plot_metric_trends(records, FIG_DIR / "randomization_metric_trends.png")

    if 0.0 in latent_cache and 100.0 in latent_cache:
        _plot_original_vs_shuffled(
            latent_cache[0.0], latent_cache[100.0], COMPARISON_FIG
        )

    summary_path = RESULT_ROOT / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print(f"Saved randomization summary to {summary_path}")
    print(f"Saved comparison figures to {FIG_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run symbol-emergence randomization controls with fixed model/analysis settings."
    )
    parser.add_argument("--ratios", type=str, default="0,20,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    ratios = _parse_ratios(args.ratios)
    run_randomization_study(
        ratios=ratios,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
    )


if __name__ == "__main__":
    main()
