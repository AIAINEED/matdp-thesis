import torch

def compute_gae(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0.0

    values = values + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]

    return (
        torch.tensor(advantages, dtype=torch.float32),
        torch.tensor(returns, dtype=torch.float32)
    )