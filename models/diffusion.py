import torch
import torch.nn as nn
from torch.distributions import Normal

class DiffusionPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.K = cfg.diffusion_steps
        self.action_dim = cfg.action_dim

        '''
        betas: noise variance
        alphas: signal retention rate
        alpha_bars: cumulative product of alphas, total signal retention after t steps
        '''
        betas = torch.linspace(1e-4, 0.02, self.K)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.t_embed = nn.Embedding(self.K, cfg.hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.action_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim)
        )
    
    # Diffusion forward process: add noise to action a0 at step t
    def q_sample(self, a0, t, noise):
        if not torch.is_tensor(t):
            t = torch.full((a0.shape[0],), int(t), device=a0.device, dtype=torch.long)
        t = t.to(device=a0.device, dtype=torch.long)
        alpha_bar = self.alpha_bars.index_select(0, t).unsqueeze(-1)
        return torch.sqrt(alpha_bar) * a0 + torch.sqrt(1 - alpha_bar) * noise
    
    # Diffusion reverse process: predict noise at step t to denoise a_t
    def pred_noise(self, a_t, h, t):
        if not torch.is_tensor(t):
            t = torch.full((a_t.shape[0],), int(t), device=a_t.device, dtype=torch.long)
        t = t.to(device=a_t.device, dtype=torch.long)
        t_emb = self.t_embed(t)
        x = torch.cat([a_t, h, t_emb], dim=-1)
        return self.net(x)

    def predict_x0(self, a_t, h, t):
        if not torch.is_tensor(t):
            t = torch.full((a_t.shape[0],), int(t), device=a_t.device, dtype=torch.long)
        t = t.to(device=a_t.device, dtype=torch.long)
        alpha_bar = self.alpha_bars.index_select(0, t).unsqueeze(-1)
        eps = torch.nan_to_num(self.pred_noise(a_t, h, t), nan=0.0, posinf=0.0, neginf=0.0)
        x0_pred = (a_t - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar.clamp(min=1e-8))
        return torch.nan_to_num(x0_pred, nan=0.0, posinf=1.0, neginf=-1.0)

    def p_mean_variance(self, a_t, x0_pred, t):
        if not torch.is_tensor(t):
            t = torch.full((a_t.shape[0],), int(t), device=a_t.device, dtype=torch.long)
        t = t.to(device=a_t.device, dtype=torch.long)

        betas = self.betas.index_select(0, t).unsqueeze(-1)
        alphas = self.alphas.index_select(0, t).unsqueeze(-1)
        alpha_bar = self.alpha_bars.index_select(0, t).unsqueeze(-1)
        prev_t = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.alpha_bars.index_select(0, prev_t).unsqueeze(-1)
        alpha_bar_prev = torch.where((t == 0).unsqueeze(-1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)

        denom = (1.0 - alpha_bar).clamp(min=1e-8)
        coef_x0 = betas * torch.sqrt(alpha_bar_prev) / denom
        coef_xt = torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / denom
        mean = coef_x0 * x0_pred + coef_xt * a_t
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        var = betas * (1.0 - alpha_bar_prev) / denom
        var = torch.nan_to_num(var, nan=1e-4, posinf=1.0, neginf=1e-4)
        var = var.clamp(min=1e-4)
        return mean, var
    
    # Diffusion loss: MSE between predicted noise and true noise, averaged over action dimensions
    def loss(self, a0, h):
        B = a0.shape[0]
        t = torch.randint(0, self.K, (B,), device=a0.device)
        noise = torch.randn_like(a0)
        a_t = self.q_sample(a0, t, noise)
        pred = self.pred_noise(a_t, h, t)
        # Return per-sample loss [B] to enable fine-grained advantage weighting
        return ((pred - noise) ** 2).mean(dim=-1)  # [B]

    def get_log_prob(self, a0, h, k, noise=None):
        if noise is None:
            noise = torch.randn_like(a0)

        if not torch.is_tensor(k):
            k = torch.full((a0.shape[0],), int(k), device=a0.device, dtype=torch.long)
        k = k.to(device=a0.device, dtype=torch.long).clamp(min=1, max=self.K - 1)

        a_k = self.q_sample(a0, k, noise)
        a_km1 = self.q_sample(a0, k - 1, noise)
        x0_pred = self.predict_x0(a_k, h, k)
        mean, var = self.p_mean_variance(a_k, x0_pred, k)

        dist = Normal(mean, torch.sqrt(var).clamp(min=1e-2))
        return dist.log_prob(a_km1).sum(dim=-1, keepdim=True)
    
    # Diffusion backward sampling: start from pure noise and iteratively denoise to get action sample
    def sample(self, h):
        B = h.shape[0]
        a = torch.randn(B, self.action_dim, device=h.device)

        for t in reversed(range(self.K)):
            t_tensor = torch.full((B,), t, device=h.device, dtype=torch.long)
            x0_pred = self.predict_x0(a, h, t_tensor)
            mean, var = self.p_mean_variance(a, x0_pred, t_tensor)

            if t > 0:
                noise = torch.randn_like(a)
            else:
                noise = torch.zeros_like(a)

            a = mean + torch.sqrt(var) * noise
 
        return a.clamp(-1.0, 1.0)