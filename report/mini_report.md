# Mini Report

## Objective
Study whether discrete structure emerges in the 1D world-model latent space, then fit flow and diffusion models on the learned latents.

## Methods
1. Train the world model on the 1D bouncing trajectory and save `latent.npy`.
2. Train a RealNVP-style flow on the latent vectors and compare the fitted density to the latent distribution.
3. Train a denoising diffusion model on the same latent vectors and compare the reverse chain to the latent trajectory.
4. Visualize the latent manifold with PCA, Jacobian heatmaps, symbol clusters, and manifold comparison plots.

## Results
Record:
- world-model reconstruction or prediction loss
- flow negative log-likelihood
- diffusion denoising loss
- whether the PCA / manifold plots preserve the latent trajectory shape
- whether the symbol clusters and state machine remain stable after adding the new latent models

## Conclusion
Summarize which latent model better matches the observed manifold and what structure is still missing.
