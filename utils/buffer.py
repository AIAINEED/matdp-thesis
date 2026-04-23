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

    def add(self, obs, action, reward, done, value):
        """
        obs: [1, N, H, obs_dim]
        action: [1, N, action_dim]
        """
        self.obs.append(obs.detach())
        self.actions.append(action.detach())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

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

        # normalize returns using running stats
        eps = 1e-8
        denom = math.sqrt(self.ret_var + eps)
        normalized = [(r - self.ret_mean) / denom for r in returns]
        self.returns = torch.tensor(normalized, dtype=torch.float32, device=self.device)

    def get(self):
        obs = torch.cat(self.obs, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)

        adv = self.advantages
        ret = self.returns

        # normalize advantage（必须）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return obs, actions, adv, ret