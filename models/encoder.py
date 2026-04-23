# transformer and MLP

import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__ (self, cfg):
        super().__init__()

        self.input_proj = nn.Linear(cfg.obs_dim, cfg.hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)

        self.agent_embed = nn.Embedding(cfg.n_agents, cfg.hidden_dim)
    
    def forward(self, x, agent_id):
        # x: (B, H, obs_dim)
        h = self.input_proj(x)

        agent_emb = self.agent_embed(agent_id).unsqueeze(1)
        h = h + agent_emb 
        h = self.encoder(h)
        return h[:, -1]


class MLPSequenceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.horizon * cfg.obs_dim
        self.agent_embed = nn.Embedding(cfg.n_agents, cfg.hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, agent_id):
        # x: (B, H, obs_dim)
        x_flat = x.reshape(x.shape[0], -1)
        agent_emb = self.agent_embed(agent_id)
        return self.net(torch.cat([x_flat, agent_emb], dim=-1))