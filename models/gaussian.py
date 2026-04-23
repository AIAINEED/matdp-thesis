import math
import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.action_dim = cfg.action_dim
        self.mean_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

    def sample(self, h):
        mean = self.mean_head(h)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        eps = torch.randn_like(mean)
        return mean + std * eps

    def loss(self, a0, h):
        # Negative log-prob per sample, shape [B]
        mean = self.mean_head(h)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)

        z = (a0 - mean) / (std + 1e-8)
        # Diagonal Gaussian NLL: 0.5 * (z^2 + 2logstd + log(2pi))
        nll = 0.5 * (z.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        return nll.sum(dim=-1)
