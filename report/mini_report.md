A preprint based on this repository is currently in preparation.
This work investigates how symbolic structure emerges from world dynamics.

<p align="center">
	<a href="../README.md" style="display:inline-block;padding:10px 18px;border-radius:999px;background:#111827;color:#ffffff;text-decoration:none;font-weight:700;letter-spacing:0.2px;">
		Back to Project README
	</a>
</p>


# Symbol Emergence from Predictive Dynamics in a 1D World Model
_A mechanistic study of how discrete symbolic structure arises from continuous latent dynamics_

---
## Abstract

This report investigates how discrete symbolic structure can emerge from the predictive geometry of a learned world model.  
In a 1D bouncing-ball environment, the latent space organizes into piecewise-smooth regions corresponding to distinct predictive regimes, and Jacobian discontinuities mark transitions between these regimes.  
Clustering these regions yields a symbolic segmentation whose transition graph reflects the underlying dynamics.  
The mechanism is model-independent: MLP, flow, and diffusion models all reproduce the same geometric boundaries despite architectural differences.

We extend the analysis to a 2D GridWorld and show that the same predictive-geometry mechanism scales to spatial environments.  
Walls, corners, and doorways induce geometric bends and sensitivity spikes, producing symbolic regions aligned with rooms and corridors.  
Mutual information between symbolic states recovers the adjacency structure of the environment, demonstrating that symbolic boundaries arise from changes in predictive organization rather than external labels.

Together, these results provide a unified account of how discrete symbols can emerge from continuous predictive experience, offering a foundation for future work on multi-agent symbol negotiation and shared semantic structure.


## 1. Introduction
Understanding how discrete symbolic structure can emerge from continuous experience is a central question in cognitive science and machine learning.

Predictive world models provide a minimal setting in which such structure may arise: to forecast future states, the model must compress continuous dynamics into an internal representation that supports accurate prediction. This raises a fundamental question: what structure does a predictive model impose on its latent space, and how does this structure relate to the environment's dynamics?

To investigate this question, we analyze the latent representations of a world model trained on a simple one-dimensional bouncing-ball environment. This environment offers a controlled testbed in which continuous motion is punctuated by discrete events, allowing us to isolate how predictive models respond to changes in dynamical regimes. Despite its simplicity, the setting captures the essential challenge of symbol emergence: continuous trajectories must be organized into discrete, behaviorally meaningful units.

Our analysis combines geometric, differential, and clustering-based methods. PCA reveals the global organization of the latent trajectory, while the encoder Jacobian exposes local changes in sensitivity that provide a differential signature of regime transitions. Clustering the latent states further reveals discrete segments corresponding to qualitatively distinct phases of the environment's dynamics. The repository-consistent experimental settings are default latent_dim=16, KMeans k=4, and analysis sample size N=999 latent states (see Appendix for implementation details).

Across multiple architectures—including multi-layer perceptrons (MLPs), invertible flow models, and diffusion models—we observe a consistent pattern: the latent trajectory forms a smooth manifold segmented by sharp transitions aligned with event boundaries. These segments can be interpreted as symbolic states, and their transitions form a compact state-transition graph summarizing the predictive structure of the environment. The cross-model consistency suggests that the observed symbolic boundaries arise from the environment's predictive dynamics rather than architecture-specific inductive biases.

This minimal framework allows us to examine how symbolic structure may emerge from prediction alone, without supervision or predefined categories, and provides a mechanistic foundation for extending symbol-emergence studies to richer environments and, ultimately, to multi-agent systems.


## 2. Related Work
Research on symbol emergence has examined how discrete categories can arise from continuous sensorimotor experience. Prior work describes symbol emergence as a multi-layered process involving individual representation learning, multimodal categorization, and eventually the formation of shared symbol systems within society. Other studies highlight how agents acquire internal categories through interaction and prediction. Unlike these approaches, our analysis focuses on the mechanistic origin of symbolic boundaries within predictive latent dynamics, even without multimodality or social interaction.

Predictive world models provide a minimal setting for studying such emergence. Earlier studies on predictive coding and learned dynamics have shown that latent spaces often reflect environmental structure, but these works typically emphasize performance or compression rather than the geometry of the latent trajectory. In contrast, we analyze how predictive regimes shape the latent manifold itself, and how discrete symbolic states may arise from transitions between these regimes.

Our approach is also related to research on representation geometry and mechanistic interpretability, which uses tools such as PCA, SVD, and Jacobian analysis to study the structure of learned representations. Prior work has examined curvature, linear regions, and activation patterns in neural networks, but has not connected these geometric features to the emergence of symbolic structure. Jacobian discontinuities provide a direct mechanistic link between predictive dynamics and symbolic segmentation.

Finally, our analysis draws on flow-based and diffusion-based generative models. Flow models provide invertible mappings that preserve structural boundaries, while diffusion models reconstruct trajectories through iterative denoising. Although typically used for density estimation or generation, we show that both models preserve large-scale geometric trends relevant to segmentation. This supports the view that observed boundaries are primarily driven by the environment's predictive structure rather than architectural artifacts.

---

## 3. Methods

### 3.1 Training Procedure

The world model is trained by gradient-based optimization to minimize prediction loss.

### 3.2 Environment

We use a deterministic one-dimensional bouncing-ball environment in which the agent observes the ball position over time and the only discrete events are left-wall and right-wall collisions. This setting isolates how symbolic boundaries arise from predictive dynamics.

### 3.3 World Model

The world model combines an encoder and decoder in a low-dimensional latent space, and it is trained to reconstruct observations while forecasting the next step.

### 3.4 Flow Model

A RealNVP flow model is trained on the latent trajectories to probe the reversible geometry of the representation. Because the flow is invertible, it provides a direct test of whether symbolic boundaries persist under a bijective reparameterization.

### 3.5 Diffusion Model

A denoising diffusion model is trained on latent trajectories to study the generative dynamics. Its reverse chain is compared with the real latent trajectory to test whether the same segmentation reappears under iterative denoising.

### 3.6 Analysis Pipeline

We analyze latent geometry with PCA, detect local structural changes with encoder Jacobians, cluster latent states into symbolic categories, and construct the induced state-transition graph.

## 4. Results

### 4.1 Latent Geometry and Symbol Segmentation

The latent geometry reveals how predictive regimes shape the internal representation.

![PCA projection of the latent trajectory](figures/latent/fig2_latent_pca.png)
**Figure 2.** PCA projection of the latent trajectory, showing a low-dimensional manifold with collision-marked breakpoints (N=999, seed=42; PCA shown on `model/latent.npy`).

The encoder Jacobian changes sharply at the same event locations, indicating nonuniform local sensitivity along the trajectory. 

![Encoder Jacobian over time](figures/jacobian/fig3_jacobian.png)
**Figure 3.** Encoder Jacobian over time, highlighting sharp sensitivity changes aligned with collisions (N=999, seed=42; Jacobian computed via autograd on the trained `WorldModel`).

Clustering the latent states produces a small number of compact groups that partition the trajectory into discrete regions of the predictive latent space.

![Symbol segmentation in latent space](figures/clustering/fig4_symbol_clusters.png)
**Figure 4.** Symbol segmentation in latent space, where clustering partitions the trajectory into symbolic regions (KMeans, k=4, n_init=20, random_state=0; N=999).

The resulting state-transition graph summarizes the repeated switching pattern observed along the trajectory.

![Symbolic state-transition graph](figures/clustering/fig5_state_machine.png)
**Figure 5.** Symbolic state-transition graph summarizing transitions among the inferred states (derived from clustered latent sequence; k=4).

### 4.2 Flow Model Analysis (Model Independence)

Flow models preserve segmentation because invertible maps cannot merge or create boundaries. Because flows are invertible, they provide a natural test of whether the observed segmentation persists under a bijective transformation of the latent space.

![Flow latent segmentation](figures/flow/fig6a_flow_latent_segmentation.png)
**Figure 6a.** Flow latent segmentation.

![Flow Jacobian time series](figures/jacobian/fig6b_flow_jacobian_timeseries.png)
**Figure 6b.** Flow Jacobian time series.

![Latent structure comparison](figures/latent/fig6c_latent_structure_comparison.png)
**Figure 6c.** Latent structure comparison. Flow-model comparison, showing that the segmentation survives an invertible reparameterization (flow samples Procrustes-aligned to `model/latent.npy`; N=999; Procrustes with scaling).

### 4.3 Diffusion Model Analysis

Diffusion models reproduce large-scale bends because denoising follows the manifold geometry. Diffusion models reconstruct data through iterative denoising, so they provide a complementary test of whether the same segmentation structure reappears in the generative setting.

![Diffusion vs. real latent trajectory](figures/diffusion/fig7_diffusion_vs_real.png)
**Figure 7.** Diffusion-model comparison, showing similar geometric transitions in the sampled trajectory, although with stochastic dispersion due to the diffusion reverse process (N=999, seed=42; reverse chain is stochastic).

### 4.4 Summary of Model-Independent Structure

Together, these results show that segmentation is preserved under both invertible and stochastic generative processes. Despite their architectural differences, the MLP and flow models exhibit closely aligned segmentation patterns, while diffusion models reproduce similar large-scale geometric trends with greater dispersion. These observations suggest that the observed boundaries reflect environment-induced structure rather than model-specific inductive bias. The summary figure below combines the most informative views of this shared structure.

![Cross-model segmentation summary](figures/jacobian/pipeline/fig8_segmentation_overlay.png)
**Figure 8.** Cross-model segmentation summary, highlighting the shared segmentation pattern across models (flow samples Procrustes-aligned; clustering k=4; N=999).

## 5. Information-Theoretic Analysis

The geometric results above can also be phrased in terms of uncertainty, compression, and predictive dependence. Because the world model is deterministic, the predictive entropy curve is estimated with a local Gaussian approximation to the one-step residuals: $\sigma_t^2 \approx \mathrm{Var}_w(x_{t+1} - \hat{x}_{t+1})$ and $H_t \approx \frac12 \log(2\pi e\sigma_t^2)$.

### 5.1 Predictive Entropy Spike

The predictive entropy proxy spikes near the same collision boundaries that produce the strongest Jacobian changes. In the one-dimensional bouncing-ball environment, these peaks are expected because the next observation becomes locally harder to predict when the trajectory switches from free motion to post-impact motion. The entropy curve therefore provides a complementary uncertainty view of the same regime boundaries identified by the Jacobian.

![Predictive entropy proxy and Jacobian norm](figures/entropy/fig9a_predictive_entropy_spike.png)
**Figure 9a.** Predictive entropy proxy and Jacobian norm over time. Both curves are standardized for comparison, and their peaks tend to align near collision-induced regime changes.

### 5.2 Segment Entropy

Segment entropy measures how concentrated each symbolic cluster is in latent space. We estimate $H(Z \mid S = k)$ for every cluster with a diagonal Gaussian approximation on the latent vectors assigned to that cluster, then average them to obtain $H(Z \mid S)$. Lower within-cluster entropy means each symbolic state occupies a tighter region of the latent manifold, while the gap $H(Z) - H(Z \mid S)$ measures how much structure the segmentation explains.

![Segment internal entropy across symbolic clusters](figures/entropy/fig9b_segment_entropy.png)
**Figure 9b.** Segment internal entropy in the clustered latent space. The global entropy line sits above the weighted conditional entropy when the partition captures meaningful latent structure.

### 5.3 Mutual Information of Symbols

The state machine exposes temporal dependence between symbolic states. We measure discrete mutual information between consecutive symbols, $I(S_t; S_{t+1})$, and also report normalized mutual information for scale comparison. Higher values indicate that the symbolic sequence retains predictive memory rather than behaving like a memoryless partition.

![Mutual information of consecutive symbols](figures/entropy/fig9c_symbol_mutual_information.png)
**Figure 9c.** Mutual information summary for the symbolic state sequence. The discrete symbols retain nontrivial predictive dependence across adjacent time steps.

### 5.4 Theoretical Lower Bound

The symbolic metrics above are conservative because clustering is a coarse-graining map, $S = q(Z)$. By the data processing inequality, the measured symbolic mutual information satisfies $I(S_t; S_{t+1}) \le I(Z_t; Z_{t+1})$, so the discrete FSM gives a lower bound on the predictive information present in the continuous latent trajectory. Equivalently, the entropy gap $H(Z) - H(Z \mid S)$ is the amount of latent structure captured by the symbol partition, while the predictive entropy proxy marks where that structure is hardest to forecast.

Taken together, the entropy spike, the segment-entropy gap, and the symbolic mutual information provide a consistent lower-bound view of the same regime transitions that appear geometrically in the latent manifold.

---

## 6. 2D Experiments

### 6.1 GridWorld Setup
To test whether the information-theoretic structure observed in 1D generalizes to spatial environments, we extend the world model to a 2D GridWorld.  
The agent performs a random walk on a $10 \times 10$ grid with four actions (left, right, up, down).  
States are normalized before training, so the same encoder, decoder, and analysis pipeline can be reused without modification.

![2D GridWorld Trajectory](figures/gridworld/fig10_gridworld_states.png)
Figure 10 shows a typical rollout, with boundary-contact states highlighted in red. (If a dedicated rollout figure exists, insert it here; otherwise, the following subfigures constitute the 2D analysis.)

---

### 6.2 Jacobian and Predictive Entropy in 2D
**Jacobian Norm (Fig. 10a).**  
The Jacobian norm of the encoder with respect to the 2D input is computed to quantify local sensitivity.  
The resulting spatial map shows strong Jacobian spikes near walls, corners, and doorways, indicating sharp changes in local dynamics.  
These spikes correspond to dynamical boundaries in the environment.

![Jacobian norm map in the 2D GridWorld](figures/gridworld/fig10a_jacobian_norm_2d.png)
**Figure 10a.** Jacobian norm map in the 2D GridWorld. Spikes appear near walls, corners, and doorways, marking dynamical boundaries.

**Predictive Entropy (Fig. 10b).**  
Predictive entropy is computed using a residual-based Gaussian approximation.  
Entropy is low inside rooms and corridors, where transitions are stable, but high near boundaries, where motion becomes constrained.  
The alignment between Jacobian spikes and entropy spikes demonstrates that dynamical boundaries coincide with information boundaries.

![Predictive entropy map in the 2D GridWorld](figures/gridworld/fig10b_predictive_entropy_2d.png)
**Figure 10b.** Predictive entropy map in the 2D GridWorld. Entropy is low inside rooms and corridors, high near boundaries.

---

### 6.3 Symbolic Clusters in 2D
**Latent Clustering (Fig. 10c).**  
Clustering the latent states yields distinct spatial regions corresponding to rooms, corridors, and corner zones.  
These clusters represent minimal-uncertainty symbolic regions, extending the 1D notion of segments into 2D spatial geometry.

![Symbolic clusters in the 2D latent space](figures/gridworld/fig10c_symbolic_clusters_2d.png)
**Figure 10c.** Symbolic clusters in the 2D latent space. Each cluster corresponds to a coherent spatial region (room interior, corridor, corner zone).

---

### 6.4 Symbol Mutual Information
**Symbol MI Matrix (Fig. 10d).**  
Mutual information between successive symbolic states is computed to reveal predictive dependencies.  
Adjacent spatial regions exhibit high MI, while distant or disconnected regions show low MI.  
This produces a symbolic adjacency structure that mirrors the topology of the environment.

![Symbol mutual information matrix](figures/gridworld/fig10d_symbol_mi_matrix_2d.png)
**Figure 10d.** Symbol mutual information matrix. High mutual information between adjacent symbolic regions mirrors the spatial topology of the environment.

---

### 6.5 Summary
Across Jacobian, entropy, clustering, and MI analyses, the 2D results show a consistent pattern:

1. **Boundaries** (walls, corners, doorways) → Jacobian spikes + entropy spikes  
2. **Regions** (rooms, corridors) → stable low‑entropy clusters  
3. **Adjacency** → high MI between neighboring symbolic regions

Together, these results demonstrate that symbolic boundaries in world models emerge from changes in predictive structure, and that this mechanism generalizes naturally from 1D to 2D environments.  
These findings provide a spatial extension of the predictive‑geometry mechanism discussed in Section 8.

---

## 7. Randomization Control

To test whether symbol emergence depends on meaningful trajectory structure, we keep the model, loss, PCA, entropy, and clustering settings fixed and only shuffle the 1D trajectory order.

### 7.1 Original vs 100% Shuffled Trajectory        

The original trajectory preserves collision-driven temporal structure, so the latent PCA often separates into stable cluster-like regions. When the same states are fully shuffled, the temporal ordering is destroyed and the latent organization becomes much less coherent. The comparison below shows the two cases side by side.

![Original trajectory versus 100% shuffled trajectory in latent PCA space](figures/randomization/original_vs_100_shuffled_pca.png)
**Figure 11.** Original trajectory versus 100% shuffled trajectory in latent PCA space. The original case shows clearer cluster structure, while the shuffled case becomes more mixed and fragmented.

### 7.2 Shuffle Ratio Trend

To quantify the effect of progressive temporal degradation, we sweep shuffle ratios of 0%, 20%, and 100% and record how prediction error, entropy, Jacobian norm, and symbol mutual information change as trajectory order is degraded. If symbol emergence depends on meaningful temporal structure, the metrics should deteriorate as the shuffle ratio increases.

| Metric | ratio = 0% | ratio = 20% | ratio = 100% | Trend |
| --- | --- | --- | --- | --- |
| Prediction MSE | 0.0135 | 0.0498 | 0.0845 | ↑ |
| Mean Predictive Entropy | -0.7916 | -0.1859 | +0.1657 | ↑ |
| Mean Jacobian Norm | 1.8455 | 1.7872 | 1.0155 | ↓ |
| Adjacent Symbol MI | 1.0466 | 0.4177 | 0.0027 | ↓ |

![Randomization metric trends](figures/randomization/randomization_metric_trends.png)
**Figure 12.** Metric trends across shuffle ratios. As trajectory order is increasingly randomized, prediction MSE and entropy rise monotonically, while Jacobian norm and adjacent symbol mutual information decline.

### 7.3 Interpretation

As the shuffle ratio increases, predictive performance deteriorates and mutual information between adjacent symbols decreases. This suggests that the emergent symbols are not merely memorized statistical patterns, but are intrinsically related to temporal organization in the observed trajectories. Without temporal coherence, the symbolic state machine structure disintegrates entirely, confirming that symbol emergence in world models is driven by the predictability of the environment's dynamics rather than static correlations.


---
## 8. Discussion

### 8.1 Predictive Latent Geometry

The PCA results suggest that the learned latent trajectory does not form an unstructured cloud, but lies on a low‑dimensional manifold with piecewise‑smooth geometry. In the bouncing‑ball environment, free motion, boundary approach, and post‑collision motion correspond to different predictive situations. The world model compresses these situations into a representation that supports forecasting, and that compression organizes the latent space into connected regions with directional changes rather than a single globally smooth curve. This geometry suggests that predictive dynamics not only produce a compact representation, but also induce structure in that representation. A piecewise‑smooth manifold thus becomes a natural substrate for symbol emergence: discrete symbolic categories can be defined as stable partitions of a continuous predictive manifold.

### 8.2 Mechanism of Symbolic Boundaries

The local mechanism behind these boundaries is the switching of ReLU activation patterns in the encoder. Because the encoder is piecewise linear, each change in activation pattern induces a new local linearization, which appears in the Jacobian as a jump or sharp change. The Jacobian therefore marks transitions between predictive regimes. When the system approaches a collision, the model must switch prediction strategies, and that regime switching is reflected as a discontinuity in sensitivity to the input. At the mechanistic level, a predictive regime is a mode of representation in the model. The correspondence between regime switching and Jacobian discontinuity provides evidence that symbolic boundaries may arise from internal predictive dynamics rather than being externally imposed. The boundary is “symbolic” because it separates distinct predictive organizations, and it is mechanistic because the separation is realized by a change in the encoder’s local linear geometry.

### 8.3 Temporal Structure as a Requirement for Emergence

A plausible alternative explanation is that the observed boundaries are merely artifacts of the ReLU network’s piecewise‑linear nature, independent of the environment’s temporal structure. The randomization control experiment (Section 7) tests this directly: if boundaries were purely architectural, shuffling the temporal order of the trajectory should leave them intact. Instead, as temporal coherence is progressively destroyed, the predictive error and entropy rise while the Jacobian norm and the mutual information between symbolic states drop. The symbolic transition structure disintegrates. This counterfactual outcome indicates that the geometric segmentation is not a fixed property of the network, but relies on the presence of a learnable predictive structure in the environment. It strengthens the claim that the boundaries observed in Section 8.2 arise from the interplay of temporal dynamics and predictive learning, not from network architecture alone.

### 8.4 Interpretation of Predictive Entropy and Its Alignment with the Jacobian

The predictive entropy was computed by first estimating the variance of the one‑step residual (predicted minus actual next state) within a sliding window. Because only second‑order statistics are reliably available from finite samples, we adopted a maximum‑entropy principle and modeled the residual as Gaussian. Under this assumption, the differential entropy per dimension reduces to $\frac{1}{2}\log(2\pi e\,\sigma^2)$, providing a direct summary of the model’s local predictive uncertainty. The Jacobian norm, in turn, measures how sharply the latent representation changes with respect to the input; to make it comparable across different spatial locations, we expressed it as a z‑score relative to its own mean and standard deviation across the environment. When both quantities are overlaid, the model exhibits a natural coupling: regions where the representation is highly sensitive to the input (large Jacobian z‑score) are precisely the regions where the model acknowledges higher uncertainty (elevated entropy). This synchronous variation is not engineered—it emerges from the same underlying predictive geometry, showing that the model implicitly encodes where its predictions are unreliable and allocates representation sensitivity accordingly. The alignment therefore reinforces the interpretation that symbolic boundaries correspond to transitions between predictive regimes, where both sensitivity and uncertainty peak.

### 8.5 From Manifold to Discrete Symbols

Once the latent geometry is segmented, k‑means discretizes the continuous manifold into a finite set of clusters. Each cluster can be interpreted as a symbolic state—not a ground‑truth label, but a stable region of the predictive latent space. The transition structure among these clusters yields a state‑transition graph that captures how the representation moves from one predictive regime to another over time. This construction shows how symbol‑like categories can emerge from geometry without explicit supervision. The clustering step turns a smooth predictive manifold into a finite symbolic alphabet, and the state‑transition graph turns that alphabet into structured dynamics while preserving temporal order.

### 8.6 Model‑Independent Structure

The comparison across MLP, flow, and diffusion models tests whether the observed segmentation is architecture‑specific or environment‑driven. The flow model, being invertible, can reshape the manifold but cannot merge distinct regions or create new boundaries; topology is preserved. Consequently, flow latent segmentation aligns with the MLP segmentation, and Procrustes comparison confirms near‑identical geometry. The flow Jacobian time series further shows that the same event‑related changes survive the invertible mapping.

Diffusion provides complementary evidence. Although the reverse process is stochastic, the denoising trajectory follows the geometry of the learned data manifold. Because that manifold is piecewise smooth, the reverse chain reproduces the same large‑scale bends and directional discontinuities seen in the predictive latent trajectory, even though individual paths are more dispersed. Thus, diffusion is stochastic at the sampling level but geometry‑dominated at the representation level.

Because both flow and diffusion models reproduce similar geometric bends, the segmentation appears driven by the environment’s predictive structure rather than by architectural details. Taken together, these results support a unified explanation: predictive regimes impose latent geometry, geometry induces segmentation, and segmentation gives rise to symbolic boundaries. This chain does not depend on a particular architecture—it appears across deterministic, invertible, and stochastic models because all must represent the same environment‑induced structure. The architectural differences change how the structure is expressed, but not whether it appears.

### 8.7 Toward Social Symbol Emergence

While this study focuses on a single predictive agent, the resulting structure suggests a natural path toward multi‑agent symbol systems. If symbolic boundaries reflect predictive regimes, then agents in the same environment may develop partially aligned segmentation patterns that can be negotiated or stabilized through communication. A multi‑agent extension would allow us to examine how individually formed symbolic states become coordinated through interaction and how shared categories emerge from the need to predict each other’s behavior. This perspective is consistent with the broader account of symbol emergence in which social interaction transforms individual predictive structures into communal symbol systems.


## Appendix

### PCA as a Geometric Operator

PCA provides a global view of the latent manifold, complementing the local sensitivity captured by the Jacobian. (See Section 4.1 for the main results.)

PCA can be interpreted geometrically as fitting an ellipsoid to the centered data cloud. Centering is 
necessary because PCA assumes that the linear transformation acts around the origin; without 
centering, the mean shift would be interpreted as variance.

Given a centered data matrix X, PCA seeks directions v that maximize the projected variance:

    maximize   vᵀ (Xᵀ X) v
    subject to ||v|| = 1

The solution is given by the eigenvectors of XᵀX, with the largest eigenvalue corresponding to the 
first principal component.

SVD provides an equivalent and more geometric interpretation. Any linear transformation maps the 
unit circle into an ellipse whose axes correspond to the singular vectors. The first right singular 
vector identifies the direction of maximal stretching, which is identical to the first principal 
component. Thus, PCA can be computed directly via the SVD of the centered data matrix:

    X = U Σ Vᵀ

where the columns of V are the principal directions and the squared singular values Σ² correspond to 
the explained variances.

### Implementation Notes

All experiments were run with fixed seeds for reproducibility.

The repository contains the code used to generate the figures and the corresponding training scripts. The default settings are summarized below for readers who want to reproduce or extend the experiments.

| Item | Value (repository) | Notes |
|---|---:|---|
| Environment | 1D bouncing-ball (deterministic) | `data/generate_data.py` |
| Latent dimension | 16 | Default CLI value used by `model/train.py` |
| Encoder | Linear(1,32) -> ReLU -> Linear(32, latent_dim) | `model/world_model.py::WorldModel` |
| Decoder | Linear(latent_dim,32) -> ReLU -> Linear(32,1) | `model/world_model.py::WorldModel` |
| Optimizer | Adam | Used across trainers |
| Learning rate | 1e-3 | Default training rate in `model/train.py` |
| Batch size | 32 (world), 128 (flow/diffusion) | Dispatcher defaults |
| Epochs | 50 (world), 300 (flow), 1000 (diffusion) | Default epoch schedule |
| Flow model | RealNVP, num_layers=8, hidden_dim=128 | Invertible baseline |
| Diffusion model | timesteps=1000, hidden_dim=256, time_embed_dim=128 | Generative baseline |
| PCA | `sklearn.decomposition.PCA(n_components=2)` | Latent-geometry summary |
| Clustering | `KMeans(n_clusters=4, n_init=20, random_state=0)` | Symbolic segmentation |
| Jacobian computation | autograd | See the subsection above |

Example commands:

```bash
python -m pip install -r requirements.txt
python model/train.py --mode world --latent-dim 8 --lr 1e-3 --batch-size 32 --epochs 50
python model/train.py --mode flow --lr 5e-4 --batch-size 128 --epochs 300
python model/train.py --mode diffusion --timesteps 1000 --lr 5e-4 --batch-size 128 --epochs 1000
python model/eval.py --checkpoint model/checkpoint.pth --plot_dir figures/
```

### 2D GridWorld Reproduction

The 2D extension uses the same model architecture with `state_dim=2` and a normalized GridWorld rollout. Run it with:

```bash
python envs/collect_2d.py
python model/train.py --mode world --data-path data/trajectories_2d.npy --state-dim 2 --epochs 20 --batch-size 32 --latent-dim 16
python analysis/run_analysis.py
python analysis/segmentation.py
```

This reproduces the 2D Jacobian norm map, predictive entropy map, symbolic cluster map, and symbol MI matrix saved under `report/figures/gridworld/`.

Record the exact git commit hash and configuration file for each experiment. Where possible, save model checkpoints, training curves, and generated figures alongside the run metadata.

## References

- Taniguchi, T., et al. (2026). *Symbol Emergence Systems: An Interdisciplinary Discussion About Cognition, Language and Society*. Springer.
- Taniguchi, T., et al. (2016). *Symbol emergence in cognitive developmental systems: A survey*. IEEE Transactions on Cognitive and Developmental Systems.
- Nakamura, T., et al. (2014). *Multimodal categorization and symbol emergence*. Advanced Robotics.
- Ha, D., & Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122.
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). *Density estimation using Real NVP*. ICLR.
- Arora, R., et al. (2018). *Understanding deep neural networks with rectified linear units*. ICLR.
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Berkeley Symposium.