import torch
import torch.nn as nn
from .encoder import MLPSequenceEncoder, SequenceEncoder
from .diffusion import DiffusionPolicy, JointDiffusionPolicy
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
        # team-level encoder to produce cross-agent latent for joint-action modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=getattr(cfg, 'n_heads', 4),
            batch_first=True,
            dropout=0.0,
        )
        # reuse cfg.n_layers for team encoder depth
        self.team_encoder = nn.TransformerEncoder(encoder_layer, getattr(cfg, 'n_layers', 1))
        # optionally initialize joint policy
        if getattr(cfg, 'enable_joint_policy', False):
            self.joint_policy = JointDiffusionPolicy(cfg)

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

    def team_encode(self, obs_seq):
        """
        Produce a team-level cross-agent latent from per-agent encodings.

        Returns:
            joint_h: [B, N, hidden_dim] (encoded agent sequence)
            joint_flat: [B, N*hidden_dim] (flattened joint latent)
        """
        # first get per-agent hidden states
        h_all = self.encode(obs_seq)  # [B, N, hidden]
        # apply cross-agent transformer
        joint_h = self.team_encoder(h_all)
        B, N, H = joint_h.shape
        joint_flat = joint_h.reshape(B, N * H)
        return joint_h, joint_flat

    def sample_with_chain_independent(self, obs_seq):
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

    def sample_with_chain(self, obs_seq, joint=False):
        """
        If joint is False (default), behaves as before and samples per-agent chains.
        If joint is True and a joint_policy exists, samples a joint action chain and reshapes outputs.
        """
        if not joint:
            return self.sample_with_chain_independent(obs_seq)

        if not hasattr(self, 'joint_policy'):
            raise RuntimeError('Joint policy not initialized on Actor')

        # produce joint latent
        joint_h, joint_flat = self.team_encode(obs_seq)  # joint_flat: [B, N*hidden]
        # joint_policy.sample_with_chain expects joint_flat as h
        action_flat, chain_flat, step_log_prob_flat = self.joint_policy.sample_with_chain(joint_flat)

        # reshape flat action to per-agent actions
        B = action_flat.shape[0]
        joint_action_dim = self.joint_policy.joint_action_dim
        single_action_dim = int(joint_action_dim / self.n_agents)
        actions = action_flat.reshape(B, self.n_agents, single_action_dim)

        # chain_flat: [B, K+1, joint_action_dim] -> reshape to [B, N, K+1, A]
        Kp1 = chain_flat.shape[1]
        chains = chain_flat.reshape(B, Kp1, self.n_agents, single_action_dim).permute(0,2,1,3)

        # keep the joint diffusion trajectory as a single scalar chain per sample
        # step_log_prob_flat already has shape [B, K]
        step_log_probs = step_log_prob_flat

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