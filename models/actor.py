import torch
import torch.nn as nn
from .encoder import MLPSequenceEncoder, SequenceEncoder
from .diffusion import DiffusionPolicy
from .gaussian import GaussianPolicy


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.n_agents = cfg.n_agents

        if cfg.encoder_type == "mlp":
            self.encoder = MLPSequenceEncoder(cfg)
        else:
            self.encoder = SequenceEncoder(cfg)

        if cfg.policy_type == "gaussian":
            self.policy = GaussianPolicy(cfg)
        else:
            self.policy = DiffusionPolicy(cfg)

    def forward(self, obs_seq):
        """
        obs_seq: [B, N, H, obs_dim]

        return:
            actions: [B, N, action_dim]
            h: list of length N, each [B, hidden_dim]
        """

        B, N, H, D = obs_seq.shape
        device = obs_seq.device

        h_list = []
        actions = []

        for i in range(N):
            agent_id = torch.full((B,), i, device=device, dtype=torch.long)

            h = self.encoder(obs_seq[:, i], agent_id)        # [B, hidden]
            a = self.policy.sample(h)                        # [B, action_dim]

            h_list.append(h)
            actions.append(a)

        actions = torch.stack(actions, dim=1)  # [B, N, action_dim]

        return actions, h_list

    def sample_with_chain(self, obs_seq):
        """
        Sample diffusion actions together with the full denoising chain and per-step log-probs.

        Returns:
            actions: [B, N, action_dim]
            chains: [B, N, K+1, action_dim]
            step_log_probs: [B, N, K]
        """
        B, N, H, D = obs_seq.shape
        device = obs_seq.device

        actions = []
        chains = []
        step_log_probs = []

        for i in range(N):
            agent_id = torch.full((B,), i, device=device, dtype=torch.long)
            h = self.encoder(obs_seq[:, i], agent_id)
            if not hasattr(self.policy, "sample_with_chain"):
                raise RuntimeError("sample_with_chain requires the diffusion policy")
            action_i, chain_i, step_log_prob_i = self.policy.sample_with_chain(h)
            actions.append(action_i)
            chains.append(chain_i)
            step_log_probs.append(step_log_prob_i)

        actions = torch.stack(actions, dim=1)
        chains = torch.stack(chains, dim=1)
        step_log_probs = torch.stack(step_log_probs, dim=1)
        return actions, chains, step_log_probs

    def encode(self, obs_seq):
        """
        return:
            h: [B, N, hidden_dim]
        """
        B, N, H, D = obs_seq.shape
        device = obs_seq.device

        h_all = []

        for i in range(N):
            agent_id = torch.full((B,), i, device=device, dtype=torch.long)
            h = self.encoder(obs_seq[:, i], agent_id)
            h_all.append(h)

        return torch.stack(h_all, dim=1)  # [B, N, hidden]