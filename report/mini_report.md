# Symbol Emergence from Predictive Dynamics in a 1D World Model
_A mechanistic study of how discrete symbolic structure arises from continuous latent dynamics_

---

## 1. Introduction

### 1.1 Background
- Symbol Emergence: how discrete symbols arise from continuous sensory-motor experience.
- Predictive processing and world models.
- Prior work: SERKET, multimodal categorization, developmental robotics.

### 1.2 Problem Statement
- How do symbolic boundaries emerge inside a single agent?
- Can we observe the mechanism directly in latent dynamics?

### 1.3 Key Idea
- Symbolic boundaries correspond to **structural changes in predictive dynamics**.
- These changes can be detected through:
  - latent geometry,
  - encoder Jacobian discontinuities,
  - clustering of latent regimes,
  - symbolic transition structure.

### 1.4 Contributions
1. A minimal world-model experiment showing **piecewise-linear latent structure**.
2. Evidence that **Jacobian discontinuities** align with symbolic boundaries.
3. A symbolic **state-transition graph** emerging from continuous dynamics.
4. Flow and diffusion models showing **model-independent symbolic geometry**.

---

## 2. Methods

### 2.1 Environment: 1D Bouncing Ball
- Deterministic physics.
- State representation.
- Trajectory generation.

### 2.2 World Model Architecture
- Encoder → Transition → Decoder.
- Training objective.
- Latent extraction.

### 2.3 Flow Model (RealNVP)
- Motivation: reversible mapping of latent space.
- Architecture and training.
- Sampling and geometry analysis.

### 2.4 Diffusion Model
- Denoising latent dynamics.
- Reverse chain visualization.
- Comparison with real latent trajectories.

### 2.5 Analysis Pipeline
- PCA of latent trajectories.
- Encoder Jacobian computation.
- Clustering of latent regimes.
- Symbolic state-transition graph construction.

---

## 3. Results

### 3.1 Latent Geometry
- PCA reveals segmented linear regions.
- Boundary alignment with bounce events.

**Figure 2. Latent PCA**

### 3.2 Jacobian Discontinuity
- Sharp structural changes in encoder Jacobian.
- Mechanistic explanation of symbolic boundaries.

**Figure 3. Jacobian heatmap**

### 3.3 Symbolic Clusters
- Clustering reveals discrete symbolic states.
- Interpretation of each symbolic region.

**Figure 4. Symbol clusters**

### 3.4 Symbolic State Machine
- Transition graph extracted from latent dynamics.
- Predictive structure becomes discrete.

**Figure 5. State-transition graph**

### 3.5 Flow Geometry
- Flow model preserves symbolic boundaries.
- Reversible mapping shows consistent segmentation.

**Figure 6. Flow vs latent geometry**

### 3.6 Diffusion Dynamics
- Denoising process reconstructs symbolic structure.
- Reverse chain aligns with real latent trajectories.

**Figure 7. Diffusion vs real dynamics**

---

## 4. Discussion

### 4.1 Mechanistic Origin of Symbolic Boundaries
- Why Jacobian discontinuities appear.
- Relationship between prediction error and segmentation.
- Interpretation as “event boundaries”.

### 4.2 Model-Independence of Symbol Emergence
- Flow and diffusion models show consistent structure.
- Suggests symbolic boundaries are **environment-driven**, not model-specific.

### 4.3 Limitations
- Single-agent setting.
- Simple 1D environment.
- No communication or shared symbols.

### 4.4 Toward Social Symbol Emergence
- This work studies **individual symbol emergence**.
- Next step: multi-agent systems where:
  - agents negotiate shared categories,
  - communication aligns symbolic systems,
  - SERKET modules can be combined.

This connects directly to the direction described in  
*Symbol Emergence Systems: Cognition, Language and Society (2026)*.

---

## 5. Conclusion

- Demonstrated how symbolic boundaries arise from predictive dynamics.
- Identified Jacobian discontinuity as a mechanistic marker.
- Showed consistent symbolic geometry across world model, flow, and diffusion.
- Provided a minimal foundation for future work on **social symbol emergence**.

---

## Appendix (optional)
- Hyperparameters
- Training curves
- Additional visualizations
