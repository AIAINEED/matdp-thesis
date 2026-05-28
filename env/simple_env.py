import numpy as np
try:
    from pettingzoo.mpe import simple_spread_v3
except ModuleNotFoundError:
    from mpe2 import simple_spread_v3


class SimpleMultiAgentEnv:
    def __init__(
        self,
        n_agents,
        obs_dim=None,
        action_dim=None,
        max_cycles=50,
        seed=None,
        gamma=0.99,
        pbrs_on=False,
        potential_scale=5.0,
        coverage_bonus=5.0,
        fov_mask_on=False,
        fov_radius=0.5,
        fov_visibility_on=True,
        terminate_on_success=False,
    ):
        self.seed = seed
        self.gamma = float(gamma)
        self.pbrs_on = bool(pbrs_on)
        self.potential_scale = float(potential_scale)
        self.coverage_bonus = float(coverage_bonus)
        self.prev_potential = 0.0
        self.last_env_reward = 0.0
        self.last_shaping_reward = 0.0
        self.last_team_reward = 0.0
        self.last_success = False
        self.last_landmark_coverage = 0.0
        self.last_min_landmark_distance = 0.0
        self.fov_mask_on = bool(fov_mask_on)
        self.fov_radius = float(fov_radius)
        self.fov_visibility_on = bool(fov_visibility_on)
        self.terminate_on_success = bool(terminate_on_success)

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

        self.base_obs_dim = int(np.prod(sample_obs_space.shape))
        self.visibility_dim = 0
        if self.fov_mask_on and self.fov_visibility_on:
            self.visibility_dim = self.n_agents + max(self.n_agents - 1, 0)
        self.obs_dim = self.base_obs_dim + self.visibility_dim
        self.action_dim = int(np.prod(sample_act_space.shape))

        self._act_low = sample_act_space.low.astype(np.float32)
        self._act_high = sample_act_space.high.astype(np.float32)
        self._act_shape = sample_act_space.shape

    def _apply_fov_mask(self, stacked_obs):
        if (not self.fov_mask_on) or stacked_obs.size == 0:
            return stacked_obs

        masked_obs = stacked_obs.copy()
        n_agents = masked_obs.shape[0]
        landmark_visible = np.ones((n_agents, self.n_agents), dtype=np.float32)
        agent_visible = np.ones((n_agents, max(self.n_agents - 1, 0)), dtype=np.float32)

        # simple_spread observation layout (flattened):
        # [self_vel(2), self_pos(2), landmarks(2*N), other_agents(2*(N-1)), comm(...)]
        landmark_start = 4
        landmark_end = landmark_start + 2 * self.n_agents
        other_start = landmark_end
        other_end = other_start + 2 * max(self.n_agents - 1, 0)

        if masked_obs.shape[1] < other_end:
            if self.fov_visibility_on:
                visibility = np.concatenate([landmark_visible, agent_visible], axis=1)
                return np.concatenate([masked_obs, visibility], axis=1).astype(np.float32, copy=False)
            return masked_obs.astype(np.float32, copy=False)

        for i in range(n_agents):
            landmarks_rel = masked_obs[i, landmark_start:landmark_end].reshape(self.n_agents, 2)
            l_dists = np.linalg.norm(landmarks_rel, axis=1)
            landmark_visible[i] = (l_dists <= self.fov_radius).astype(np.float32)
            landmarks_rel[landmark_visible[i] == 0.0] = 0.0
            masked_obs[i, landmark_start:landmark_end] = landmarks_rel.reshape(-1)

            if self.n_agents > 1:
                agents_rel = masked_obs[i, other_start:other_end].reshape(self.n_agents - 1, 2)
                a_dists = np.linalg.norm(agents_rel, axis=1)
                agent_visible[i] = (a_dists <= self.fov_radius).astype(np.float32)
                agents_rel[agent_visible[i] == 0.0] = 0.0
                masked_obs[i, other_start:other_end] = agents_rel.reshape(-1)

        if self.fov_visibility_on:
            visibility = np.concatenate([landmark_visible, agent_visible], axis=1)
            masked_obs = np.concatenate([masked_obs, visibility], axis=1)

        return masked_obs.astype(np.float32, copy=False)

    def _stack_obs(self, obs_dict):
        if not obs_dict:
            return np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)

        obs_list = []
        for agent in self.agent_ids:
            obs = obs_dict.get(agent)
            if obs is None:
                obs = np.zeros((self.base_obs_dim,), dtype=np.float32)
            obs_list.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        stacked = np.stack(obs_list, axis=0)
        return self._apply_fov_mask(stacked)

    def _get_potential(self, obs_dict):
        if not obs_dict:
            return 0.0

        min_dists_to_landmarks = self._get_min_dists_to_landmarks(obs_dict)
        if min_dists_to_landmarks is None:
            return 0.0
        return -float(np.sum(min_dists_to_landmarks))

    def _get_min_dists_to_landmarks(self, obs_dict):
        if not obs_dict:
            return None

        all_landmark_rel_pos = []
        for agent in self.agent_ids:
            obs = np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1)
            landmark_end = 4 + 2 * self.n_agents
            if obs.shape[0] < landmark_end:
                return None
            landmark_rel_pos = obs[4:landmark_end].reshape(self.n_agents, 2)
            all_landmark_rel_pos.append(landmark_rel_pos)

        all_landmark_rel_pos = np.asarray(all_landmark_rel_pos, dtype=np.float32)
        distances = np.linalg.norm(all_landmark_rel_pos, axis=-1)
        return np.min(distances, axis=0)

    def _get_landmark_coverage(self, obs_dict, threshold=0.1):
        min_dists_to_landmarks = self._get_min_dists_to_landmarks(obs_dict)
        if min_dists_to_landmarks is None:
            return 0.0
        return float(np.mean(min_dists_to_landmarks <= threshold))

    def _get_success(self, obs_dict):
        min_dists_to_landmarks = self._get_min_dists_to_landmarks(obs_dict)
        if min_dists_to_landmarks is None:
            return False
        return bool(np.all(min_dists_to_landmarks <= 0.2))

    def reset(self):
        obs_dict, _ = self.env.reset(seed=self.seed)
        self.seed = None
        self.prev_potential = self._get_potential(obs_dict)
        self.last_env_reward = 0.0
        self.last_shaping_reward = 0.0
        self.last_team_reward = 0.0
        self.last_success = False
        self.last_landmark_coverage = 0.0
        self.last_min_landmark_distance = 0.0
        return self._stack_obs(obs_dict)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, self._act_low, self._act_high)

        action_dict = {}
        for i, agent in enumerate(self.agent_ids):
            action_dict[agent] = actions[i].reshape(self._act_shape)

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
        next_obs = self._stack_obs(obs_dict)

        env_team_reward = float(sum(reward_dict.values()))
        shaping_reward = 0.0
        if self.pbrs_on:
            current_potential = self._get_potential(obs_dict)
            shaping_reward = (self.gamma * current_potential - self.prev_potential) * self.potential_scale
            self.prev_potential = current_potential

        # Gravity bonus 已回退：保留为注释，便于后续对照实验恢复
        # try:
        #     # 计算每个 agent 到最近地标的距离
        #     landmark_end = 4 + 2 * self.n_agents
        #     gravity_bonus_sum = 0.0
        #     for i, agent in enumerate(self.agent_ids):
        #         obs = np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1)
        #         if obs.shape[0] < landmark_end:
        #             continue
        #         landmark_rel_pos = obs[4:landmark_end].reshape(self.n_agents, 2)
        #         dists = np.linalg.norm(landmark_rel_pos, axis=1)
        #         # agent 最短距离到任一地标
        #         agent_min = float(np.min(dists))
        #         gravity_bonus = float(np.exp(-5.0 * agent_min))
        #         gravity_bonus_sum += gravity_bonus
        #     # 把 gravity bonus 作为 shaping reward 的一部分
        #     shaping_reward += float(gravity_bonus_sum)
        # except Exception:
        #     # 如果 obs_dict 格式不符合预期，则跳过 gravity bonus
        #     pass

        coverage_bonus_reward = self.coverage_bonus if self._get_success(obs_dict) else 0.0
        team_reward = env_team_reward + shaping_reward + coverage_bonus_reward
        self.last_env_reward = env_team_reward
        self.last_shaping_reward = float(shaping_reward + coverage_bonus_reward)
        self.last_team_reward = float(team_reward)
        self.last_success = self._get_success(obs_dict)
        min_dists_to_landmarks = self._get_min_dists_to_landmarks(obs_dict)
        if min_dists_to_landmarks is None:
            self.last_landmark_coverage = 0.0
            self.last_min_landmark_distance = 0.0
        else:
            self.last_landmark_coverage = self._get_landmark_coverage(obs_dict)
            self.last_min_landmark_distance = float(np.min(min_dists_to_landmarks))

        done = all(term_dict.get(a, False) or trunc_dict.get(a, False) for a in self.agent_ids)
        if self.terminate_on_success and self.last_success:
            done = True
        return next_obs, team_reward, done, info_dict

    def close(self):
        self.env.close()
