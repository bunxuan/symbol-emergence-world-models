# Symbol Emergence in a 1D World Model

This repository explores how **discrete symbolic structure** can emerge from **continuous predictive dynamics** in a minimal one-dimensional environment.  
The project provides a mechanistic analysis of how **latent geometry**, **Jacobian discontinuities**, and **transition structure** give rise to symbolic boundaries.

---

## Full Report (Mini Preprint)

The full mechanistic analysis—including methods, results, figures, and references—is available here:

<p align="center">
  <a href="report/mini_report.md" style="display:inline-block;padding:10px 18px;border-radius:999px;background:#111827;color:#ffffff;text-decoration:none;font-weight:700;letter-spacing:0.2px;">
    Open Mini Report / Preprint Draft
  </a>
  <a href="report/mini_report.pdf" style="display:inline-block;padding:10px 18px;border-radius:999px;background:#0f766e;color:#ffffff;text-decoration:none;font-weight:700;letter-spacing:0.2px;margin-left:8px;">
    Open mini_report.pdf
  </a>
</p>

The README provides a concise overview; the report contains the full technical details.

---

## Key Mechanistic Insights

This project suggests that **symbolic boundaries** can emerge when a predictive model must represent **structurally different predictive regimes**.

### • Latent Manifold  
PCA shows that the latent trajectory lies on a **smooth, low‑dimensional manifold** rather than a scattered cloud.  
The world model compresses continuous dynamics into a structured trajectory.

### • Jacobian Discontinuities  
Because the encoder is piecewise linear, changes in ReLU activation patterns produce **sharp Jacobian jumps**.  
These jumps align with physical event boundaries (wall collisions), providing a mechanistic marker of symbolic segmentation.

### • Symbol Clustering  
K‑means discretizes the latent manifold into **phase‑like segments**, each corresponding to a distinct predictive regime.

### • Symbolic State Machine  
The sequence of cluster assignments forms a **discrete symbolic state sequence**, which induces a compact **state‑transition graph** summarizing predictive dynamics.

### • Model Robustness (Flow & Diffusion)  
- **Flow models** preserve segmentation‑like boundaries because invertible maps cannot create or destroy topological structure.  
- **Diffusion reverse chains** reproduce **similar large‑scale geometric bends**, though with stochastic dispersion.  
Together, these results suggest that symbolic boundaries arise primarily from the **environment’s predictive structure**, not from a specific architecture.

<p align="center">
  <img src="report/figures/fig1_pipeline.png" width="720">
</p>

---

## 1. Overview

We investigate the hypothesis:

> **Symbolic boundaries arise when predictive dynamics undergo structural changes.**

Using a deterministic 1D bouncing‑ball environment, we observe:

- Latent trajectories become **piecewise‑linear**  
- Encoder Jacobian exhibits **sharp discontinuities** at collisions  
- These discontinuities align with **emergent symbolic states**  
- Flow and diffusion models reveal **consistent geometric trends**  
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

### Latent PCA
<p align="center">
<img src="report/figures/fig2_latent_pca.png" width="480">
</p>

Latent trajectories form segmented linear regions, indicating distinct predictive regimes.

### Jacobian Discontinuity
<p align="center">
<img src="report/figures/fig3_jacobian.png" width="480">
</p>

Encoder Jacobian shows sharp structural changes at bounce events —
a mechanistic origin of symbolic boundaries.

### Symbol Clusters & State Machine
<p align="center">
<img src="report/figures/fig4_symbol_clusters.png" width="420">
<img src="report/figures/fig5_state_machine.png" width="420">
</p>

Discrete symbolic states emerge naturally, forming a predictive state‑transition graph.

### Flow & Diffusion Dynamics
<p align="center">
<img src="report/figures/fig6a_flow_latent_segmentation.png" width="420">
<img src="report/figures/fig7_diffusion_vs_real.png" width="420">
</p>

Flow models preserve segmentation‑like boundaries via invertible reparameterizations.
Diffusion reverse chains reproduce similar geometric bends with stochastic dispersion.

## 4. Methods (Brief)
Environment: 1D bouncing ball (deterministic)

World Model: encoder → transition → decoder

Flow Model: RealNVP on latent space

Diffusion Model: denoising latent dynamics

Analysis: PCA, Jacobian, clustering, symbolic transition graph

Full details are in the report.

## 5. Run
Windows:

```bat
run.bat
run.bat all
scripts\\run_all.bat
```

Or run individual stages:

```bat
run.bat world
run.bat flow
run.bat diffusion
```

## 6. Toward Social Symbol Emergence
This project focuses on individual symbol emergence.
Future work will extend this framework to multi‑agent systems, where:

Agents align or negotiate symbolic categories

Communication stabilizes shared symbols

Symbol systems emerge at the societal level

## 7. Citation
A preprint is in preparation.
If you find this project useful, please star the repository or cite the report.

## 8. Contact
Xu Wenxuan  
Ningbo University of Technology
GitHub: https://github.com/bunxuan  
Email: jyosa@nbut.edu.cn