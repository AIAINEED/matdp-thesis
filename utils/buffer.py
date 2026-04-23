import torch

class ReplayBuffer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
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

        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

    def get(self):
        obs = torch.cat(self.obs, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)

        adv = self.advantages
        ret = self.returns

        # normalize advantage（必须）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return obs, actions, adv, ret