# Symbol Emergence in a 1D World Model

<p align="center">
  <img src="report/figures/fig1_pipeline.png" width="720">
</p>

This repository explores **how discrete symbolic structure can emerge from continuous predictive dynamics** in a simple 1D bouncing-ball environment.  
The project investigates the mechanisms by which **latent discontinuities, Jacobian changes, and transition structure** give rise to symbolic boundaries.

The experiments combine:
- A compact **world model** (encoder–transition–decoder)
- A **flow model** (RealNVP) on latent space
- A **diffusion model** (denoising latent dynamics)
- A full analysis pipeline for **latent geometry**, **Jacobian structure**, **symbol clustering**, and **state-transition graphs**

The goal is to provide a minimal but mechanistic demonstration of **individual symbol emergence**, forming a foundation for future work on **social symbol emergence**.

---

## 1. Overview

This project studies the hypothesis:

> **Symbolic boundaries arise when predictive dynamics undergo structural changes.**

Using a 1D environment with deterministic physics, we show that:
- Latent trajectories become **piecewise-linear**
- Encoder Jacobian exhibits **sharp discontinuities** at wall-bounce events
- These discontinuities align with **emergent symbolic states**
- Flow and diffusion models reveal **consistent symbolic geometry**
- A discrete **symbolic state machine** emerges from continuous dynamics

---

## 2. Repository Structure

```text
symbol-emergence-world-models/
│
├── model/                # World model, flow, diffusion
├── analysis/             # PCA, Jacobian, clustering, state machine
│   └── plots/            # Generated figures
├── results/              # Latent, samples, model weights
├── report/               # mini_report + final figures
└── README.md
```

---

## 3. Key Results

### **Latent PCA**
<p align="center">
  <img src="report/figures/fig2_latent_pca.png" width="480">
</p>

Latent trajectories form **segmented linear regions**, suggesting discrete regimes.

---

### **Jacobian Discontinuity**
<p align="center">
  <img src="report/figures/fig3_jacobian.png" width="480">
</p>

Encoder Jacobian shows **sharp structural changes** at bounce events —  
a mechanistic origin of symbolic boundaries.

---

### **Symbol Clusters & State Machine**
<p align="center">
  <img src="report/figures/fig4_symbol_clusters.png" width="420">
  <img src="report/figures/fig5_state_machine.png" width="420">
</p>

Discrete symbolic states emerge naturally, forming a **predictive state-transition graph**.

---

### **Flow & Diffusion Dynamics**
<p align="center">
  <img src="report/figures/fig6_flow_geometry.png" width="420">
  <img src="report/figures/fig7_diffusion_dynamics.png" width="420">
</p>

Both reversible (flow) and generative (diffusion) models preserve the same symbolic boundaries,  
indicating **model-independent symbolic structure**.

---

## 4. Methods

- **Environment:** 1D bouncing ball with deterministic physics  
- **World Model:** encoder → transition → decoder  
- **Flow Model:** RealNVP on latent space  
- **Diffusion Model:** denoising latent dynamics  
- **Analysis:** PCA, Jacobian, clustering, symbolic transition graph

All code is modular and reproducible.

---

## 5. Toward Social Symbol Emergence

This project focuses on **individual symbol emergence** —  
how a single agent’s predictive dynamics produce symbolic boundaries.

In future work, this framework can extend to **multi-agent systems**, where:
- Agents share or negotiate symbolic categories  
- Communication induces alignment  
- Symbol systems emerge at the **societal level**

This connects directly to ongoing research in **Symbol Emergence Systems**.

---

## 6. Citation

A preprint is in preparation.  
If you find this project useful, please cite or star the repository.

---

## 7. Contact

**Xu Wenxuan**  
Ningbo University of Technology  
GitHub: https://github.com/bunxuan  
Email: jyosa@nbut.edu.cn