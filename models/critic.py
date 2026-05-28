import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        input_dim = cfg.obs_dim * cfg.n_agents

        self.input_proj = nn.Linear(input_dim, cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            batch_first=True,
            dropout=0.0,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)

        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1)
        )

    def forward(self, state_seq):
        """
        state_seq: [B, N, H, obs_dim]
        """

        B, N, H, D = state_seq.shape

        # Concatenate all agents as global state
        x = state_seq.reshape(B, H, N * D)

        x = self.input_proj(x)

        x = self.encoder(x)

        # Use mean pooling over time for a more stable aggregated representation
        z = x.mean(dim=1)

        value = self.value_head(z)

        return value.squeeze(-1)