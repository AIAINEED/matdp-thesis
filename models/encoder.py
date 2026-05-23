# transformer and MLP

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, H, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class SequenceEncoder(nn.Module):
    def __init__ (self, cfg):
        super().__init__()

        self.input_proj = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.pos_encoder = PositionalEncoding(
            d_model=cfg.hidden_dim,
            max_len=cfg.history_len
        )

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)

        self.agent_embed = nn.Embedding(cfg.n_agents, cfg.hidden_dim)
    
    def forward(self, x, agent_id):
        # x: (B, H, obs_dim)
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        agent_emb = self.agent_embed(agent_id).unsqueeze(1)
        h = h + agent_emb 
        h = self.encoder(h)
        return h[:, -1]


class MLPSequenceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.history_len * cfg.obs_dim
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