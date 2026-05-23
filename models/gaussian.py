import math
import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.action_dim = cfg.action_dim
        action_low = getattr(cfg, "action_low", -1.0)
        action_high = getattr(cfg, "action_high", 1.0)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.action_eps = float(getattr(cfg, "diffusion_action_eps", 1e-4))
        self.mean_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

    def _expand_bounds(self, ref):
        action_low = self.action_low.to(device=ref.device, dtype=ref.dtype)
        action_high = self.action_high.to(device=ref.device, dtype=ref.dtype)
        while action_low.ndim < ref.ndim:
            action_low = action_low.unsqueeze(0)
            action_high = action_high.unsqueeze(0)
        scale = (action_high - action_low).clamp(min=1e-6) * 0.5
        center = (action_high + action_low) * 0.5
        return center, scale

    def _squash(self, raw_action):
        action_low = self.action_low.to(device=raw_action.device, dtype=raw_action.dtype)
        action_high = self.action_high.to(device=raw_action.device, dtype=raw_action.dtype)
        while action_low.ndim < raw_action.ndim:
            action_low = action_low.unsqueeze(0)
            action_high = action_high.unsqueeze(0)
        center, scale = self._expand_bounds(raw_action)
        action = center + scale * torch.tanh(raw_action)
        margin = (action_high - action_low).clamp(min=1e-6) * self.action_eps
        return torch.clamp(action, action_low + margin, action_high - margin)

    def _unsquash(self, action):
        center, scale = self._expand_bounds(action)
        limit = 1.0 - self.action_eps
        normalized = torch.clamp((action - center) / scale, -limit, limit)
        return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    def _squash_log_jacobian(self, action):
        center, scale = self._expand_bounds(action)
        limit = 1.0 - self.action_eps
        normalized = torch.clamp((action - center) / scale, -limit, limit)
        jacobian = scale * (1.0 - normalized.pow(2)) + 1e-6
        return torch.log(jacobian).sum(dim=-1)

    def _distribution(self, h):
        mean = self.mean_head(h)
        log_std = self.log_std.clamp(-5.0, 2.0).unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        return mean, log_std, std

    def sample(self, h):
        mean = self.mean_head(h)
        log_std = self.log_std.clamp(-5.0, 2.0).unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        return self._squash(mean + std * eps)

    def loss(self, a0, h):
        return -self.get_log_prob(a0, h).squeeze(-1)

    def get_log_prob(self, a, h, k=None, noise=None):
        """
        Compute log probability under Gaussian policy.
        Parameters k and noise are ignored (for diffusion compatibility).

        Args:
            a: action [B, action_dim]
            h: hidden state [B, hidden_dim]
            k: (unused, for API compatibility)
            noise: (unused, for API compatibility)

        Returns:
            log_prob [B, 1]
        """
        raw_a = self._unsquash(a)
        mean, log_std, std = self._distribution(h)

        z = (raw_a - mean) / (std + 1e-8)
        log_prob = -0.5 * (z.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        log_prob = log_prob.sum(dim=-1) - self._squash_log_jacobian(a)
        return log_prob.unsqueeze(-1)
