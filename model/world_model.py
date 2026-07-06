"""World model and latent-space modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    input_dim,
    output_dim,
    hidden_dim=64,
    num_hidden_layers=2,
    activation=nn.ReLU,
):
    layers = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    def __init__(self, state_dim=1, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        """
        x: shape (batch, state_dim)
        returns:
            pred: predicted next x
            h: latent representation
        """
        h = self.encode(x)
        pred = self.decode(h)
        return pred, h


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dim=64):
        super().__init__()

        mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.ndim != 1 or mask.numel() != dim:
            raise ValueError("mask must be a 1D tensor with length equal to dim")

        self.dim = dim
        self.register_buffer("mask", mask)

        transformed_dim = int(mask.sum().item())
        conditioner_dim = dim - transformed_dim
        if transformed_dim == 0 or conditioner_dim == 0:
            raise ValueError(
                "Affine coupling needs both transformed and conditioning dimensions"
            )

        self.net = build_mlp(
            conditioner_dim,
            transformed_dim * 2,
            hidden_dim=hidden_dim,
            num_hidden_layers=2,
            activation=nn.ReLU,
        )

    def forward(self, x, reverse=False):
        transformed = x[:, self.mask]
        conditioned = x[:, ~self.mask]

        params = self.net(conditioned)
        scale, shift = params.chunk(2, dim=-1)
        scale = torch.tanh(scale)

        if not reverse:
            transformed = transformed * torch.exp(scale) + shift
            log_det = scale.sum(dim=-1)
        else:
            transformed = (transformed - shift) * torch.exp(-scale)
            log_det = -scale.sum(dim=-1)

        y = x.clone()
        y[:, self.mask] = transformed
        return y, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, num_layers=4, hidden_dim=64):
        super().__init__()

        if dim < 2:
            raise ValueError("RealNVP needs at least 2 dimensions")

        self.dim = dim
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            mask = torch.zeros(dim, dtype=torch.bool)
            mask[layer_idx % 2 :: 2] = True
            self.layers.append(AffineCoupling(dim, mask, hidden_dim=hidden_dim))

    def forward(self, x, reverse=False):
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            x, log_det = layer(x, reverse=reverse)
            log_det_total = log_det_total + log_det

        return x, log_det_total

    @torch.no_grad()
    def sample(self, num_samples, device=None):
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.forward(z, reverse=True)
        return x


class FlowModule(nn.Module):
    def __init__(self, dim, num_layers=4, hidden_dim=64):
        super().__init__()
        self.flow = RealNVP(dim, num_layers=num_layers, hidden_dim=hidden_dim)

    def forward(self, z):
        return self.flow(z)

    def sample(self, num_samples, device=None):
        return self.flow.sample(num_samples, device=device)


class EpsilonPredictor(nn.Module):
    def __init__(self, dim, timesteps, hidden_dim=128, time_embed_dim=64):
        super().__init__()

        self.timesteps = timesteps
        self.time_embed = nn.Embedding(timesteps, time_embed_dim)
        self.net = build_mlp(
            dim + time_embed_dim,
            dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=2,
            activation=nn.SiLU,
        )

    def forward(self, z_t, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)

        t = t.long().view(-1).clamp(0, self.timesteps - 1)
        if t.numel() == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])

        time_features = self.time_embed(t)
        return self.net(torch.cat([z_t, time_features], dim=-1))


class DiffusionModule(nn.Module):
    def __init__(self, dim, timesteps=1000, hidden_dim=128, time_embed_dim=64):
        super().__init__()

        self.dim = dim
        self.timesteps = timesteps
        self.eps_model = EpsilonPredictor(
            dim,
            timesteps,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
        )

        betas = torch.linspace(1e-4, 2e-2, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)

    def _extract(self, buffer, t, x_shape):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.long().view(-1)
        if t.numel() == 1 and x_shape[0] > 1:
            t = t.expand(x_shape[0])

        values = buffer.gather(0, t)
        return values.view(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)

        sqrt_alpha_bars = self._extract(self.sqrt_alpha_bars, t, z0.shape)
        sqrt_one_minus_alpha_bars = self._extract(
            self.sqrt_one_minus_alpha_bars, t, z0.shape
        )
        return sqrt_alpha_bars * z0 + sqrt_one_minus_alpha_bars * noise

    def loss(self, z0):
        batch_size = z0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise=noise)
        pred_noise = self(z_t, t)
        return F.mse_loss(pred_noise, noise)

    def forward(self, z_t, t):
        return self.eps_model(z_t, t)

    def p_sample(self, z_t, t):
        pred_noise = self(z_t, t)
        betas_t = self._extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alpha_bars_t = self._extract(
            self.sqrt_one_minus_alpha_bars, t, z_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, z_t.shape)
        posterior_variance_t = self._extract(self.posterior_variance, t, z_t.shape)

        model_mean = sqrt_recip_alphas_t * (
            z_t - betas_t * pred_noise / sqrt_one_minus_alpha_bars_t
        )

        noise = torch.randn_like(z_t)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (z_t.dim() - 1)))
        return (
            model_mean
            + nonzero_mask
            * torch.sqrt(torch.clamp(posterior_variance_t, min=1e-20))
            * noise
        )

    @torch.no_grad()
    def sample(self, num_samples, device=None, return_chain=False):
        if device is None:
            device = next(self.parameters()).device

        z_t = torch.randn(num_samples, self.dim, device=device)
        chain = [z_t.detach().cpu()] if return_chain else None

        for step in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            z_t = self.p_sample(z_t, t)
            if return_chain:
                chain.append(z_t.detach().cpu())

        if return_chain:
            return z_t, torch.stack(chain, dim=0)
        return z_t
