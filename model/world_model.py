"""World model definition."""
import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: shape (batch, 1)
        returns:
            pred: predicted next x
            h: latent representation
        """
        h = self.encoder(x)
        pred = self.decoder(h)
        return pred, h

