# Symbol Emergence in a 1D World Model

A minimal experiment showing how symbolic states can emerge from continuous dynamics in a learned world model.

## Overview

A small ball moves back and forth between two boundaries in a 1D world. A neural world model learns to predict the next position of the ball. After training, the model’s latent space is analyzed to see whether it forms stable symbolic states and transitions.

## Project Structure

```text
symbol-emergence-1d/
├── data/
│   └── generate_data.py
├── model/
│   ├── world_model.py
│   └── train.py
├── analysis/
│   ├── pca_analysis.py
│   ├── jacobian_analysis.py
│   ├── symbol_clustering.py
│   └── state_machine.py
├── results/
├── report/
├── scripts/
│   ├── run.sh
│   ├── analysis.sh
│   └── all.sh
└── README.md
```

## What Each Step Does

- `data/generate_data.py`: generates a 1D bouncing trajectory and saves `data/trajectories.npy`
- `model/train.py`: trains the world model and saves `model/world_model.pth` and `model/latent.npy`
- `analysis/pca_analysis.py`: plots the latent trajectory in 2D PCA space
- `analysis/jacobian_analysis.py`: computes the encoder Jacobian and saves a heatmap
- `analysis/symbol_clustering.py`: clusters latent states into symbolic labels
- `analysis/state_machine.py`: builds a symbolic transition graph

## Main Results

### 1. PCA of Latent Trajectory

The latent space forms a compact trajectory that reflects the underlying dynamics of the ball.

### 2. Jacobian Heatmap

The Jacobian `∂h/∂x` highlights where the encoder changes sharply and where stable regions appear.

### 3. Symbol Clustering

Latent vectors are clustered directly in latent space to obtain discrete symbolic states.

### 4. Symbolic State Machine

The final state machine visualizes transitions between emergent symbolic states.

![Symbolic State Machine](results/state_machine.png)

## How to Run

### Run everything at once

```bash
bash "c:/JyB/Symbol Emergence/symbol-emergence-1d/scripts/all.sh"
```

### Run in two stages

```bash
bash "c:/JyB/Symbol Emergence/symbol-emergence-1d/scripts/run.sh"
bash "c:/JyB/Symbol Emergence/symbol-emergence-1d/scripts/analysis.sh"
```

### Run step by step

```bash
python data/generate_data.py
python model/train.py
python analysis/pca_analysis.py
python analysis/jacobian_analysis.py
python analysis/symbol_clustering.py
python analysis/state_machine.py
```

## Interpretation

This project demonstrates a simple pipeline:

continuous dynamics -> latent encoding -> symbolic clustering -> transition system

It is a compact example of symbol emergence in a learned world model.

## Notes

This repository is part of ongoing research on Symbol Emergence and Cognitive Systems for graduate study in Japan.
