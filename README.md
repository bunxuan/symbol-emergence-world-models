# Symbol Emergence in a 1D World Model

A compact experiment showing how symbolic states emerge from continuous dynamics in a learned world model.

A small ball bounces between two boundaries in 1D, and a neural world model learns latent representations that can be clustered into symbolic states.

## Highlights

<table>
<tr>
<td colspan="2" align="center">
<img src="results/state_machine.png" alt="Symbolic State Machine" width="900">
<br>
<strong>1. Symbolic State Machine</strong><br>
Main result: emergent symbolic states and their transition counts.
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="results/jacobian_heatmap.png" alt="Jacobian Heatmap" width="100%">
<br>
<strong>2. Jacobian Heatmap</strong><br>
Encoder sensitivity and symbol boundaries.
</td>
<td align="center" width="50%">
<img src="results/symbol_clusters.png" alt="Symbol Clusters" width="100%">
<br>
<strong>3. Symbol Clusters</strong><br>
Latent-only clustering of emergent states.
</td>
</tr>
</table>

## Pipeline

- Generate trajectory: `data/generate_data.py`
- Train world model: `model/train.py`
- Analyze latent trajectory: `analysis/pca_analysis.py`
- Compute Jacobian: `analysis/jacobian_analysis.py`
- Cluster latent states: `analysis/symbol_clustering.py`
- Build state machine: `analysis/state_machine.py`

## How to Run

### Run everything at once

```bash
bash scripts/all.sh
```

## Note

Continuous dynamics -> latent encoding -> discrete symbolic states -> transition system.
