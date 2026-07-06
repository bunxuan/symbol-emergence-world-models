# 2D GridWorld Pipeline Guide

This note explains the end-to-end path from raw 2D rollout data to the saved figures.

## 1. Raw Environment Data

The process starts in [envs/collect_2d.py](../envs/collect_2d.py), which instantiates [envs/gridworld.py](../envs/gridworld.py) and collects a random walk of `(x, y)` states plus actions.

Output files:

- `data/trajectories_2d.npy`
- `data/actions_2d.npy`

## 2. World-Model Training

The 2D trajectory is normalized by the grid size and passed into the shared world model through `model/train.py` with `--state-dim 2`.

What happens:

- `WorldModel(state_dim=2)` is created
- the encoder learns a 2D -> latent mapping
- the decoder reconstructs the next 2D state
- the trained weights are saved to `model/world_model.pth`
- the latent sequence is saved to `model/latent.npy`

## 3. Spatial Analysis

The main analysis entry point is [analysis/run_analysis.py](../analysis/run_analysis.py).

Inside that script:

1. `collect_trajectory_2d(...)` loads a fresh 2D rollout
2. `states_norm = states / env.size` prepares the model input
3. `z = model.encode(states_norm)` extracts latent states
4. `compute_jacobian_norm(model, states_norm)` measures encoder sensitivity
5. `compute_predictive_entropy(model, states_norm)` estimates uncertainty
6. `cluster_latent(z, n_clusters=6)` partitions latent states
7. `compute_symbol_mi(clusters)` builds the symbol MI matrix
8. the figures are saved under `report/figures/gridworld/`

Saved outputs:

- `fig10a_jacobian_norm_2d.png`
- `fig10b_predictive_entropy_2d.png`
- `fig10c_symbolic_clusters_2d.png`
- `fig10d_symbol_mi_matrix_2d.png`

## 4. Spatial Segmentation View

The companion script [analysis/segmentation.py](../analysis/segmentation.py) reuses the same latent states and clustering logic to draw the symbolic regions directly on the 2D plane.

This produces the same cluster map as the main analysis, but the script is kept separate so the segmentation view is easy to inspect on its own.

## 5. One-Key Execution

The whole workflow is wrapped by `run.py` and can be launched with:

```bat
run.bat gridworld
```

This command performs the full pipeline in order:

1. collect 2D rollout
2. train 2D world model
3. compute Jacobian / entropy / MI maps
4. render the symbolic clustering map

The figures end up in `report/figures/gridworld/`.