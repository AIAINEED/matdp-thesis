import torch
import torch.nn as nn

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
        self.register_buffer("alpha_bars", alpha_bars)
        self.t_embed = nn.Embedding(self.K, cfg.hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.action_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim)
        )
    
    # Diffusion forward process: add noise to action a0 at step t
    def q_sample(self, a0, t, noise):
        alpha_bar = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(alpha_bar) * a0 + torch.sqrt(1 - alpha_bar) * noise
    
    # Diffusion reverse process: predict noise at step t to denoise a_t
    def pred_noise(self, a_t, h, t):
        t_emb = self.t_embed(t)
        x = torch.cat([a_t, h, t_emb], dim=-1)
        return self.net(x)
    
    # Diffusion loss: MSE between predicted noise and true noise, averaged over action dimensions
    def loss(self, a0, h):
        B = a0.shape[0]
        t = torch.randint(0, self.K, (B,), device=a0.device)
        noise = torch.randn_like(a0)
        a_t = self.q_sample(a0, t, noise)
        pred = self.pred_noise(a_t, h, t)
        # Return per-sample loss [B] to enable fine-grained advantage weighting
        return ((pred - noise) ** 2).mean(dim=-1)  # [B]
    
    # Diffusion backward sampling: start from pure noise and iteratively denoise to get action sample
    def sample(self, h):
        B = h.shape[0]
        a = torch.randn(B, self.action_dim, device=h.device)

        for t in reversed(range(self.K)):
            t_tensor = torch.full((B,), t, device=h.device, dtype=torch.long)
            eps = self.pred_noise(a, h, t_tensor)
            alpha_bar = self.alpha_bars[t]
            a = (a - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
 
        return a