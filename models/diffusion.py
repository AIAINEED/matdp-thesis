import torch
import torch.nn as nn
from torch.distributions import Normal

class DiffusionPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.K = cfg.diffusion_steps
        self.action_dim = cfg.action_dim
        action_low = getattr(cfg, "action_low", -1.0)
        action_high = getattr(cfg, "action_high", 1.0)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.action_eps = float(getattr(cfg, "diffusion_action_eps", 1e-4))
        self.min_std = float(getattr(cfg, "diffusion_min_std", 0.05))

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

    def _expand_bounds(self, ref):
        action_low = self.action_low.to(device=ref.device, dtype=ref.dtype)
        action_high = self.action_high.to(device=ref.device, dtype=ref.dtype)
        while action_low.ndim < ref.ndim:
            action_low = action_low.unsqueeze(0)
            action_high = action_high.unsqueeze(0)
        scale = (action_high - action_low).clamp(min=1e-6) * 0.5
        center = (action_high + action_low) * 0.5
        return action_low, action_high, center, scale

    def _squash(self, raw_action):
        action_low, action_high, center, scale = self._expand_bounds(raw_action)
        action = center + scale * torch.tanh(raw_action)
        margin = (action_high - action_low).clamp(min=1e-6) * self.action_eps
        return torch.clamp(action, action_low + margin, action_high - margin)

    def _unsquash(self, action):
        _, _, center, scale = self._expand_bounds(action)
        limit = 1.0 - self.action_eps
        normalized = torch.clamp((action - center) / scale, -limit, limit)
        return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    def _squash_log_jacobian(self, action):
        _, _, center, scale = self._expand_bounds(action)
        limit = 1.0 - self.action_eps
        normalized = torch.clamp((action - center) / scale, -limit, limit)
        jacobian = scale * (1.0 - normalized.pow(2)) + 1e-6
        return torch.log(jacobian).sum(dim=-1)
    
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
        x = torch.cat([self._unsquash(a_t), h, t_emb], dim=-1)
        return self.net(x)

    def predict_x0(self, a_t, h, t):
        if not torch.is_tensor(t):
            t = torch.full((a_t.shape[0],), int(t), device=a_t.device, dtype=torch.long)
        t = t.to(device=a_t.device, dtype=torch.long)
        alpha_bar = self.alpha_bars.index_select(0, t).unsqueeze(-1)
        eps = torch.nan_to_num(self.pred_noise(a_t, h, t), nan=0.0, posinf=0.0, neginf=0.0)
        a_t_raw = self._unsquash(a_t)
        x0_pred = (a_t_raw - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar.clamp(min=1e-8))
        # Keep x0_pred in raw (un-squashed) space and avoid squashing here.
        # Relax extreme clamp to avoid creating derivatives that blow up.
        x0_pred = torch.nan_to_num(x0_pred, nan=0.0, posinf=10.0, neginf=-10.0)
        return x0_pred

    def p_mean_variance(self, a_t, x0_pred, t):
        if not torch.is_tensor(t):
            t = torch.full((a_t.shape[0],), int(t), device=a_t.device, dtype=torch.long)
        t = t.to(device=a_t.device, dtype=torch.long)

        a_t_raw = self._unsquash(a_t)
        betas = self.betas.index_select(0, t).unsqueeze(-1)
        alphas = self.alphas.index_select(0, t).unsqueeze(-1)
        alpha_bar = self.alpha_bars.index_select(0, t).unsqueeze(-1)
        prev_t = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.alpha_bars.index_select(0, prev_t).unsqueeze(-1)
        alpha_bar_prev = torch.where((t == 0).unsqueeze(-1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)

        denom = (1.0 - alpha_bar).clamp(min=1e-8)
        coef_x0 = betas * torch.sqrt(alpha_bar_prev) / denom
        coef_xt = torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / denom
        # x0_pred is already in raw space (returned by predict_x0),
        # do NOT call _unsquash on it. Combine using raw a_t.
        mean = coef_x0 * x0_pred + coef_xt * a_t_raw
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)

        var = betas * (1.0 - alpha_bar_prev) / denom
        var = torch.nan_to_num(var, nan=1e-4, posinf=1.0, neginf=1e-4)
        var = var.clamp(min=max(1e-4, self.min_std ** 2))
        return mean, var

    def get_step_distribution(self, obs, a_t, t, h=None):
        if h is None:
            raise ValueError("h is required for step-wise diffusion distributions")
        x0_pred = self.predict_x0(a_t, h, t)
        mean, var = self.p_mean_variance(a_t, x0_pred, t)
        std = torch.sqrt(var).clamp(min=self.min_std)
        return Normal(mean, std)

    def get_step_log_prob(self, obs, a_t, a_prev, t, h=None):
        dist = self.get_step_distribution(obs, a_t, t, h=h)
        a_prev_raw = self._unsquash(a_prev)
        log_prob = dist.log_prob(a_prev_raw).sum(dim=-1)
        log_prob = log_prob - self._squash_log_jacobian(a_prev)
        return torch.nan_to_num(log_prob, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Diffusion loss: MSE between predicted noise and true noise, averaged :ver action dimens,ons
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

        dist = Normal(mean, torch.sqrt(var).clamp(min=self.min_std))
        log_prob = dist.log_prob(a_km1).sum(dim=-1, keepdim=True)
        return torch.nan_to_num(log_prob, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Diffusion backward sampling: start from pure noise and iteratively denoise to get action sample
    def sample(self, h):
        B = h.shape[0]
        a = torch.randn(B, self.action_dim, device=h.device)

        for t in reversed(range(self.K)):
            t_tensor = torch.full((B,), t, device=h.device, dtype=torch.long)
            a_squashed = self._squash(a)
            x0_pred = self.predict_x0(a_squashed, h, t_tensor)
            mean, var = self.p_mean_variance(a_squashed, x0_pred, t_tensor)

            if t > 0:
                noise = torch.randn_like(a)
            else:
                noise = torch.zeros_like(a)

            std = torch.sqrt(var).clamp(min=self.min_std)
            a = mean + std * noise
 
        return self._squash(a)

    def sample_with_chain(self, h):
        B = h.shape[0]
        a = torch.randn(B, self.action_dim, device=h.device)
        chain = [self._squash(a)]
        step_log_probs = [None for _ in range(self.K)]

        for step in reversed(range(self.K)):
            t_tensor = torch.full((B,), step, device=h.device, dtype=torch.long)
            a_t = chain[-1]
            x0_pred = self.predict_x0(a_t, h, t_tensor)
            mean, var = self.p_mean_variance(a_t, x0_pred, t_tensor)

            if step > 0:
                noise = torch.randn_like(a)
            else:
                noise = torch.zeros_like(a)

            std = torch.sqrt(var).clamp(min=self.min_std)
            a = mean + std * noise
            a_prev = self._squash(a)
            step_log_probs[step] = self.get_step_log_prob(None, a_t, a_prev, t_tensor, h=h)
            chain.append(a_prev)

        return chain[-1], torch.stack(chain, dim=1), torch.stack(step_log_probs, dim=1)


class JointDiffusionPolicy(DiffusionPolicy):
    """
    Joint diffusion over concatenated actions for all agents.
    Action dimension internally is cfg.action_dim * cfg.n_agents.
    Joint hidden dim is cfg.hidden_dim * cfg.n_agents (from stacked per-agent h).
    """
    def __init__(self, cfg):
        # create a copy of cfg with adjusted action and hidden dims
        super().__init__(cfg)
        self.n_agents = int(getattr(cfg, 'n_agents', 1))
        self.joint_action_dim = int(cfg.action_dim * self.n_agents)
        self.joint_hidden_dim = int(cfg.hidden_dim * self.n_agents)

        # rebuild t_embed and net to match joint dims
        self.t_embed = nn.Embedding(self.K, self.joint_hidden_dim)
        # override net to accept joint inputs: unsquashed a_t (joint_action_dim) + joint h + t_emb
        self.net = nn.Sequential(
            nn.Linear(self.joint_hidden_dim + self.joint_action_dim + self.joint_hidden_dim, self.joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.joint_hidden_dim, self.joint_action_dim)
        )

    def _expand_bounds(self, ref):
        # ref expected to have joint_action_dim in last dim
        action_low = self.action_low.to(device=ref.device, dtype=ref.dtype)
        action_high = self.action_high.to(device=ref.device, dtype=ref.dtype)
        # expand bounds to joint dims by tiling
        if action_low.ndim == 0 or action_low.numel() == 1:
            action_low = action_low.repeat(self.joint_action_dim)
            action_high = action_high.repeat(self.joint_action_dim)
        elif action_low.numel() * self.n_agents == self.joint_action_dim:
            action_low = action_low.repeat(self.n_agents)
            action_high = action_high.repeat(self.n_agents)
        elif action_low.numel() == self.joint_action_dim:
            # already joint-sized
            pass
        else:
            # fallback: repeat to joint action dim
            action_low = action_low.repeat(self.joint_action_dim // max(1, action_low.numel()))
            action_high = action_high.repeat(self.joint_action_dim // max(1, action_high.numel()))
        while action_low.ndim < ref.ndim:
            action_low = action_low.unsqueeze(0)
            action_high = action_high.unsqueeze(0)
        scale = (action_high - action_low).clamp(min=1e-6) * 0.5
        center = (action_high + action_low) * 0.5
        return action_low, action_high, center, scale

    def _squash(self, raw_action):
        # raw_action: [B, joint_action_dim]
        action_low, action_high, center, scale = self._expand_bounds(raw_action)
        action = center + scale * torch.tanh(raw_action)
        margin = (action_high - action_low).clamp(min=1e-6) * self.action_eps
        return torch.clamp(action, action_low + margin, action_high - margin)

    def _unsquash(self, action):
        # reverse of _squash for joint actions
        _, _, center, scale = self._expand_bounds(action)
        limit = 1.0 - self.action_eps
        normalized = torch.clamp((action - center) / scale, -limit, limit)
        return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    def predict_x0(self, a_t, h, t):
        # h expected to be joint_flat of size [B, joint_hidden_dim]
        if not torch.is_tensor(t):
            t = torch.full((a_t.shape[0],), int(t), device=a_t.device, dtype=torch.long)
        t = t.to(device=a_t.device, dtype=torch.long)
        alpha_bar = self.alpha_bars.index_select(0, t).unsqueeze(-1)
        eps = torch.nan_to_num(self.pred_noise(a_t, h, t), nan=0.0, posinf=0.0, neginf=0.0)
        a_t_raw = self._unsquash(a_t)
        x0_pred = (a_t_raw - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar.clamp(min=1e-8))
        x0_pred = torch.nan_to_num(x0_pred, nan=0.0, posinf=10.0, neginf=-10.0)
        return x0_pred

    def pred_noise(self, a_t, h, t):
        if not torch.is_tensor(t):
            t = torch.full((a_t.shape[0],), int(t), device=a_t.device, dtype=torch.long)
        t = t.to(device=a_t.device, dtype=torch.long)
        t_emb = self.t_embed(t)
        x = torch.cat([self._unsquash(a_t), h, t_emb], dim=-1)
        return self.net(x)

    def sample_with_chain(self, h):
        # h: [B, joint_hidden_dim]
        B = h.shape[0]
        a = torch.randn(B, self.joint_action_dim, device=h.device)
        chain = [self._squash(a)]
        step_log_probs = [None for _ in range(self.K)]

        for step in reversed(range(self.K)):
            t_tensor = torch.full((B,), step, device=h.device, dtype=torch.long)
            a_t = chain[-1]
            x0_pred = self.predict_x0(a_t, h, t_tensor)
            mean, var = self.p_mean_variance(a_t, x0_pred, t_tensor)

            if step > 0:
                noise = torch.randn_like(a)
            else:
                noise = torch.zeros_like(a)

            std = torch.sqrt(var).clamp(min=self.min_std)
            a = mean + std * noise
            a_prev = self._squash(a)
            step_log_probs[step] = self.get_step_log_prob(None, a_t, a_prev, t_tensor, h=h)
            chain.append(a_prev)

        # chain: list len K+1 of [B, joint_action_dim] -> stack to [B, K+1, joint_action_dim]
        return chain[-1], torch.stack(chain, dim=1), torch.stack(step_log_probs, dim=1)

    def get_step_log_prob(self, obs, a_t, a_prev, t, h=None):
        dist = self.get_step_distribution(obs, a_t, t, h=h)
        a_prev_raw = self._unsquash(a_prev)
        log_prob = dist.log_prob(a_prev_raw).sum(dim=-1)
        # log_prob = (log_prob - self._squash_log_jacobian(a_prev)) / scale
        log_prob = (log_prob - self._squash_log_jacobian(a_prev))
        return torch.nan_to_num(log_prob, nan=-1e6, posinf=1e6, neginf=-1e6)

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

        dist = Normal(mean, torch.sqrt(var).clamp(min=self.min_std))
        # scale = max(float(self.joint_action_dim) ** 0.5, 1.0)
        log_prob = dist.log_prob(a_km1).sum(dim=-1, keepdim=True)
        # log_prob = (log_prob - self._squash_log_jacobian(a_km1).unsqueeze(-1)) / scale
        log_prob = (log_prob - self._squash_log_jacobian(a_km1).unsqueeze(-1))
        return torch.nan_to_num(log_prob, nan=-1e6, posinf=1e6, neginf=-1e6)
