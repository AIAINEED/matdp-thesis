import torch
import math


class ReplayBuffer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        # running stats for returns (persist across episodes)
        self.ret_count = 0
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.step_log_probs = []
        self.chains = []
        self.ks = []
        self.noises = []

    def add(
        self,
        obs,
        action,
        reward,
        done,
        value,
        log_prob=None,
        step_log_prob=None,
        chain=None,
        k=None,
        noise=None,
    ):
        """
        obs: [1, N, H, obs_dim]
        action: [1, N, action_dim]
        """
        self.obs.append(obs.detach())
        self.actions.append(action.detach())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        if log_prob is not None:
            self.log_probs.append(log_prob.detach())
        if step_log_prob is not None:
            self.step_log_probs.append(step_log_prob.detach())
        if chain is not None:
            self.chains.append(chain.detach())
        if k is not None:
            self.ks.append(k.detach())
        if noise is not None:
            self.noises.append(noise.detach())

    def compute_returns_advantages(self, last_value):
        rewards = self.rewards
        values = self.values + [float(last_value)]
        dones = self.dones

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]

        # update running mean/var for returns (Welford / batch merge)
        batch = torch.tensor(returns, dtype=torch.float32).cpu().numpy()
        if batch.size > 0:
            batch_mean = float(batch.mean())
            batch_var = float(batch.var())
            batch_count = batch.size

            # combine existing running stats with batch stats
            if self.ret_count == 0:
                self.ret_mean = batch_mean
                self.ret_var = batch_var
                self.ret_count = batch_count
            else:
                new_count = self.ret_count + batch_count
                delta = batch_mean - self.ret_mean
                new_mean = self.ret_mean + delta * (batch_count / new_count)

                m_a = self.ret_var * self.ret_count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta * delta * (self.ret_count * batch_count / new_count)
                new_var = M2 / new_count if new_count > 0 else 1.0

                self.ret_mean = new_mean
                self.ret_var = new_var
                self.ret_count = new_count

        # store advantages and normalized returns for stability
        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # DO NOT normalize returns here — store raw returns so critic
        # predicts values on the same scale as rewards. Advantage
        # normalization is handled elsewhere before actor update.
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

    def get(self):
        obs = torch.cat(self.obs, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)
        log_probs = torch.cat(self.log_probs, dim=0).to(self.device) if self.log_probs else None
        step_log_probs = (
            torch.cat(self.step_log_probs, dim=0).to(self.device)
            if self.step_log_probs
            else None
        )
        chains = torch.cat(self.chains, dim=0).to(self.device) if self.chains else None
        ks = torch.cat(self.ks, dim=0).to(self.device) if self.ks else None
        noises = torch.cat(self.noises, dim=0).to(self.device) if self.noises else None

        adv = self.advantages
        ret = self.returns

        return obs, actions, adv, ret, step_log_probs, chains, ks, noises