import numpy as np
from pettingzoo.mpe import simple_spread_v3

class SimpleMultiAgentEnv:
    def __init__(self, n_agents, obs_dim=None, action_dim=None, max_cycles=50, seed=None):
        self.seed = seed
        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=True,
            render_mode=None,
        )

        self.agent_ids = list(self.env.possible_agents)
        self.n_agents = len(self.agent_ids)

        sample_obs_space = self.env.observation_space(self.agent_ids[0])
        sample_act_space = self.env.action_space(self.agent_ids[0])

        self.obs_dim = int(np.prod(sample_obs_space.shape))
        self.action_dim = int(np.prod(sample_act_space.shape))

        self._act_low = sample_act_space.low.astype(np.float32)
        self._act_high = sample_act_space.high.astype(np.float32)
        self._act_shape = sample_act_space.shape

    def _stack_obs(self, obs_dict):
        if not obs_dict:
            return np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)

        obs_list = []
        for agent in self.agent_ids:
            obs = obs_dict.get(agent)
            if obs is None:
                obs = np.zeros((self.obs_dim,), dtype=np.float32)
            obs_list.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        return np.stack(obs_list, axis=0)

    def reset(self):
        obs_dict, _ = self.env.reset(seed=self.seed)
        self.seed = None
        return self._stack_obs(obs_dict)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, self._act_low, self._act_high)

        action_dict = {}
        for i, agent in enumerate(self.agent_ids):
            action_dict[agent] = actions[i].reshape(self._act_shape)

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
        next_obs = self._stack_obs(obs_dict)

        team_reward = float(sum(reward_dict.values()))
        done = all(term_dict.get(a, False) or trunc_dict.get(a, False) for a in self.agent_ids)
        return next_obs, team_reward, done, info_dict

    def close(self):
        self.env.close() 