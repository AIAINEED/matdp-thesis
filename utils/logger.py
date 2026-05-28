import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir='logs', plot_meta=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.plot_meta = plot_meta or {}
        
        # Generate unique run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        self.episodes = []
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        # diagnostics
        self.adv_means = []
        self.adv_stds = []
        self.approx_kls = []
        # new ratio diagnostics
        self.clip_fracs = []
        self.ratio_means = []
        self.ratio_stds = []
        self.raw_log_ratio_means = []
        self.raw_log_ratio_stds = []
        self.mse_means = []
        self.actor_grad_norms = []
        self.critic_grad_norms = []
        self.nan_skips = []
        self.valid_updates = []
        self.update_failed = []
        self.skip_critic_loss = []
        self.skip_critic_grad = []
        self.skip_actor_log_prob = []
        self.skip_actor_loss = []
        self.skip_actor_grad = []
        self.actor_grad_sanitized = []
        self.did_updates = []
        self.rollout_steps = []
        self.reward_scales = []
        self.env_rewards = []
        self.shaping_rewards = []
        self.effective_shaping_rewards = []
        self.shaping_weights = []
        self.team_rewards = []
        self.successes = []
        self.success_rates = []
        self.episode_steps = []
        self.landmark_coverages = []
        self.min_landmark_distances = []
        
        print(f"[Logger] Run ID: {self.run_id}")
        print(f"[Logger] Results will be saved to: {self.run_dir}")
    
    def record(self, episode, reward, actor_loss, critic_loss, diag=None):
        """Record metrics from one episode."""
        self.episodes.append(episode)
        self.rewards.append(float(reward) if hasattr(reward, 'item') else float(reward))
        self.actor_losses.append(float(actor_loss) if hasattr(actor_loss, 'item') else float(actor_loss))
        self.critic_losses.append(float(critic_loss) if hasattr(critic_loss, 'item') else float(critic_loss))
        if diag is not None:
            self.adv_means.append(float(diag.get('adv_mean', 0.0)))
            self.adv_stds.append(float(diag.get('adv_std', 0.0)))
            self.approx_kls.append(float(diag.get('approx_kl', 0.0)))
            self.clip_fracs.append(float(diag.get('clip_frac', 0.0)))
            self.ratio_means.append(float(diag.get('ratio_mean', 0.0)))
            self.ratio_stds.append(float(diag.get('ratio_std', 0.0)))
            self.raw_log_ratio_means.append(float(diag.get('raw_log_ratio_mean', 0.0)))
            self.raw_log_ratio_stds.append(float(diag.get('raw_log_ratio_std', 0.0)))
            self.mse_means.append(float(diag.get('mse_mean', 0.0)))
            self.actor_grad_norms.append(float(diag.get('actor_grad_norm', 0.0)))
            self.critic_grad_norms.append(float(diag.get('critic_grad_norm', 0.0)))
            self.nan_skips.append(int(diag.get('nan_skips', 0)))
            self.valid_updates.append(int(diag.get('valid_updates', 0)))
            self.update_failed.append(int(diag.get('update_failed', 0)))
            self.skip_critic_loss.append(int(diag.get('skip_critic_loss', 0)))
            self.skip_critic_grad.append(int(diag.get('skip_critic_grad', 0)))
            self.skip_actor_log_prob.append(int(diag.get('skip_actor_log_prob', 0)))
            self.skip_actor_loss.append(int(diag.get('skip_actor_loss', 0)))
            self.skip_actor_grad.append(int(diag.get('skip_actor_grad', 0)))
            self.actor_grad_sanitized.append(int(diag.get('actor_grad_sanitized', 0)))
            self.did_updates.append(int(diag.get('did_update', 0)))
            self.rollout_steps.append(int(diag.get('rollout_steps', 0)))
            self.reward_scales.append(float(diag.get('reward_scale', 1.0)))
            self.env_rewards.append(float(diag.get('env_reward', 0.0)))
            self.shaping_rewards.append(float(diag.get('shaping_reward', 0.0)))
            self.effective_shaping_rewards.append(float(diag.get('effective_shaping_reward', 0.0)))
            self.shaping_weights.append(float(diag.get('shaping_weight', 1.0)))
            self.team_rewards.append(float(diag.get('team_reward', reward if hasattr(reward, 'item') else reward)))
            success_value = float(diag.get('success', diag.get('success_rate', 0.0)))
            self.successes.append(success_value)
            self.success_rates.append(success_value)
            self.episode_steps.append(int(diag.get('episode_steps', 0)))
            self.landmark_coverages.append(float(diag.get('landmark_coverage', 0.0)))
            self.min_landmark_distances.append(float(diag.get('min_landmark_distance', 0.0)))
        else:
            # keep lengths consistent
            self.adv_means.append(0.0)
            self.adv_stds.append(0.0)
            self.approx_kls.append(0.0)
            self.clip_fracs.append(0.0)
            self.ratio_means.append(0.0)
            self.ratio_stds.append(0.0)
            self.raw_log_ratio_means.append(0.0)
            self.raw_log_ratio_stds.append(0.0)
            self.mse_means.append(0.0)
            self.actor_grad_norms.append(0.0)
            self.critic_grad_norms.append(0.0)
            self.nan_skips.append(0)
            self.valid_updates.append(0)
            self.update_failed.append(0)
            self.skip_critic_loss.append(0)
            self.skip_critic_grad.append(0)
            self.skip_actor_log_prob.append(0)
            self.skip_actor_loss.append(0)
            self.skip_actor_grad.append(0)
            self.actor_grad_sanitized.append(0)
            self.did_updates.append(0)
            self.rollout_steps.append(0)
            self.reward_scales.append(1.0)
            self.env_rewards.append(0.0)
            self.shaping_rewards.append(0.0)
            self.effective_shaping_rewards.append(0.0)
            self.shaping_weights.append(1.0)
            self.team_rewards.append(float(reward) if hasattr(reward, 'item') else float(reward))
            self.successes.append(0.0)
            self.success_rates.append(0.0)
            self.episode_steps.append(0)
            self.landmark_coverages.append(0.0)
            self.min_landmark_distances.append(0.0)
    def _moving_average(self, values, window=50):
        """Compute moving average."""
        if len(values) < window:
            return values
        ma = np.convolve(values, np.ones(window) / window, mode='valid')
        return np.concatenate([values[:window-1], ma])
    
    def plot(self, save=True):
        """Plot and save training curves."""
        if len(self.episodes) == 0:
            return
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 11))
        axes = axes.reshape(-1)
        fig.subplots_adjust(bottom=0.12, top=0.93, hspace=0.42, wspace=0.28)

        footer_items = []
        for key in ("max_steps", "episodes", "actor_lr", "critic_lr", "reward_scale", "method","n_agents"):
            if key in self.plot_meta and self.plot_meta[key] is not None:
                footer_items.append(f"{key}={self.plot_meta[key]}")
        footer_text = " | ".join(footer_items)
        
        # Reward plot
        ax = axes[0]
        ax.plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Episode Reward')
        ma_rewards = self._moving_average(self.rewards, window=50)
        ax.plot(self.episodes[:len(ma_rewards)], ma_rewards, 'b-', linewidth=1, label='MA-50 Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Actor Loss plot
        ax = axes[1]
        ax.plot(self.episodes, self.actor_losses, 'r-', alpha=0.3, label='Episode ActorLoss')
        ma_actor = self._moving_average(self.actor_losses, window=50)
        ax.plot(self.episodes[:len(ma_actor)], ma_actor, 'r-', linewidth=1, label='MA-50 ActorLoss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Critic Loss plot
        ax = axes[2]
        ax.plot(self.episodes, self.critic_losses, 'g-', alpha=0.3, label='Episode CriticLoss')
        ma_critic = self._moving_average(self.critic_losses, window=50)
        ax.plot(self.episodes[:len(ma_critic)], ma_critic, 'g-', linewidth=1, label='MA-50 CriticLoss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Approx KL plot
        ax = axes[3]
        ax.plot(self.episodes, self.approx_kls, 'm-', alpha=0.3, label='Episode Approx KL')
        ma_kl = self._moving_average(self.approx_kls, window=50)
        ax.plot(self.episodes[:len(ma_kl)], ma_kl, 'm-', linewidth=1, label='MA-50 Approx KL')
        ax.axhline(0.01, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(0.05, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Approx KL')
        ax.set_title('PPO Approx KL')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gradient norms plot
        ax = axes[4]
        ax.plot(self.episodes, self.actor_grad_norms, color='tab:orange', alpha=0.35, label='Actor Grad Norm')
        ma_actor_grad = self._moving_average(self.actor_grad_norms, window=50)
        ax.plot(self.episodes[:len(ma_actor_grad)], ma_actor_grad, color='tab:orange', linewidth=1, label='MA-50 Actor Grad Norm')
        ax.plot(self.episodes, self.critic_grad_norms, color='tab:green', alpha=0.35, label='Critic Grad Norm')
        ma_critic_grad = self._moving_average(self.critic_grad_norms, window=50)
        ax.plot(self.episodes[:len(ma_critic_grad)], ma_critic_grad, color='tab:green', linewidth=1, label='MA-50 Critic Grad Norm')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Advantage stats plot
        ax = axes[5]
        ax.plot(self.episodes, self.adv_means, color='tab:blue', alpha=0.35, label='Adv Mean')
        ma_adv_mean = self._moving_average(self.adv_means, window=50)
        ax.plot(self.episodes[:len(ma_adv_mean)], ma_adv_mean, color='tab:blue', linewidth=1, label='MA-50 Adv Mean')
        ax.plot(self.episodes, self.adv_stds, color='tab:red', alpha=0.35, label='Adv Std')
        ma_adv_std = self._moving_average(self.adv_stds, window=50)
        ax.plot(self.episodes[:len(ma_adv_std)], ma_adv_std, color='tab:red', linewidth=1, label='MA-50 Adv Std')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Advantage')
        ax.set_title('Advantage Stats')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Success rate plot
        ax = axes[6]
        ax.plot(self.episodes, self.success_rates, color='tab:green', alpha=0.35, label='Episode Success')
        ma_success = self._moving_average(self.success_rates, window=50)
        ax.plot(self.episodes[:len(ma_success)], ma_success, color='tab:green', linewidth=1, label='MA-50 Success Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success')
        ax.set_title('Success Rate')
        ax.set_yticks([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Episode length plot
        ax = axes[7]
        ax.plot(self.episodes, self.episode_steps, color='tab:purple', alpha=0.35, label='Episode Steps')
        ma_steps = self._moving_average(self.episode_steps, window=50)
        ax.plot(self.episodes[:len(ma_steps)], ma_steps, color='tab:purple', linewidth=1, label='MA-50 Episode Steps')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Landmark coverage plot
        ax = axes[8]
        ax.plot(self.episodes, self.landmark_coverages, color='tab:olive', alpha=0.35, label='Episode Coverage')
        ma_coverage = self._moving_average(self.landmark_coverages, window=50)
        ax.plot(self.episodes[:len(ma_coverage)], ma_coverage, color='tab:olive', linewidth=1, label='MA-50 Coverage')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Coverage')
        ax.set_title('Landmark Coverage Ratio')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Minimum landmark distance plot
        ax = axes[9]
        ax.plot(self.episodes, self.min_landmark_distances, color='tab:brown', alpha=0.35, label='Episode Min Dist')
        ma_min_dist = self._moving_average(self.min_landmark_distances, window=50)
        ax.plot(self.episodes[:len(ma_min_dist)], ma_min_dist, color='tab:brown', linewidth=1, label='MA-50 Min Dist')
        ax.axhline(0.1, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Success Threshold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Distance')
        ax.set_title('Minimum Landmark Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if footer_text:
            fig.text(0.5, 0.04, footer_text, ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        if footer_text:
            plt.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            plt.tight_layout()
        
        if save:
            plot_file = self.run_dir / 'training_curves.png'
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            print(f"[Log] Saved plot to {plot_file}")
            
            # Update "latest" symlink for quick access
            latest_link = self.log_dir / 'latest'
            try:
                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(self.run_dir.resolve())
                print(f"[Log] Updated latest symlink: {latest_link} -> {self.run_dir}")
            except Exception as e:
                print(f"[Log] Warning: Could not create symlink: {e}")

        plt.close(fig)
    
    def save_metrics(self, extra=None):
        """Save metrics to JSON for later analysis."""
        metrics = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'episodes': self.episodes,
            'rewards': self.rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'adv_means': self.adv_means,
            'adv_stds': self.adv_stds,
            'approx_kls': self.approx_kls,
            'clip_fracs': self.clip_fracs,
            'ratio_means': self.ratio_means,
            'ratio_stds': self.ratio_stds,
            'raw_log_ratio_means': self.raw_log_ratio_means,
            'raw_log_ratio_stds': self.raw_log_ratio_stds,
            'mse_means': self.mse_means,
            'actor_grad_norms': self.actor_grad_norms,
            'critic_grad_norms': self.critic_grad_norms,
            'nan_skips': self.nan_skips,
            'valid_updates': self.valid_updates,
            'update_failed': self.update_failed,
            'skip_critic_loss': self.skip_critic_loss,
            'skip_critic_grad': self.skip_critic_grad,
            'skip_actor_log_prob': self.skip_actor_log_prob,
            'skip_actor_loss': self.skip_actor_loss,
            'skip_actor_grad': self.skip_actor_grad,
            'actor_grad_sanitized': self.actor_grad_sanitized,
            'did_updates': self.did_updates,
            'rollout_steps': self.rollout_steps,
            'reward_scales': self.reward_scales,
            'env_rewards': self.env_rewards,
            'shaping_rewards': self.shaping_rewards,
            'effective_shaping_rewards': self.effective_shaping_rewards,
            'shaping_weights': self.shaping_weights,
            'team_rewards': self.team_rewards,
            'successes': self.successes,
            'success_rate': self.success_rates,
            'success_rates': self.success_rates,
            'episode_steps': self.episode_steps,
            'landmark_coverages': self.landmark_coverages,
            'min_landmark_distances': self.min_landmark_distances,
            'summary': {
                'final_reward': float(self.rewards[-1]) if self.rewards else 0,
                'avg_reward_last_10': float(np.mean(self.rewards[-10:])) if len(self.rewards) >= 10 else float(np.mean(self.rewards)),
                'avg_reward': float(np.mean(self.rewards)) if self.rewards else 0,
                'avg_actor_loss': float(np.mean(self.actor_losses)) if self.actor_losses else 0,
                'avg_critic_loss': float(np.mean(self.critic_losses)) if self.critic_losses else 0,
                'avg_adv_mean': float(np.mean(self.adv_means)) if self.adv_means else 0,
                'avg_adv_std': float(np.mean(self.adv_stds)) if self.adv_stds else 0,
                'avg_approx_kl': float(np.mean(self.approx_kls)) if self.approx_kls else 0,
                'avg_mse_mean': float(np.mean(self.mse_means)) if self.mse_means else 0,
                'avg_clip_frac': float(np.mean(self.clip_fracs)) if self.clip_fracs else 0,
                'avg_ratio_mean': float(np.mean(self.ratio_means)) if self.ratio_means else 0,
                'avg_ratio_std': float(np.mean(self.ratio_stds)) if self.ratio_stds else 0,
                'avg_raw_log_ratio_mean': float(np.mean(self.raw_log_ratio_means)) if self.raw_log_ratio_means else 0,
                'avg_raw_log_ratio_std': float(np.mean(self.raw_log_ratio_stds)) if self.raw_log_ratio_stds else 0,
                'avg_actor_grad_norm': float(np.mean(self.actor_grad_norms)) if self.actor_grad_norms else 0,
                'avg_critic_grad_norm': float(np.mean(self.critic_grad_norms)) if self.critic_grad_norms else 0,
                'avg_nan_skips': float(np.mean(self.nan_skips)) if self.nan_skips else 0,
                'avg_valid_updates': float(np.mean(self.valid_updates)) if self.valid_updates else 0,
                'total_update_failed': int(np.sum(self.update_failed)) if self.update_failed else 0,
                'avg_actor_grad_sanitized': float(np.mean(self.actor_grad_sanitized)) if self.actor_grad_sanitized else 0,
                'total_policy_updates': int(np.sum(self.did_updates)) if self.did_updates else 0,
                'avg_rollout_steps': float(np.mean(self.rollout_steps)) if self.rollout_steps else 0,
                'avg_reward_scale': float(np.mean(self.reward_scales)) if self.reward_scales else 0,
                'avg_env_reward': float(np.mean(self.env_rewards)) if self.env_rewards else 0,
                'avg_shaping_reward': float(np.mean(self.shaping_rewards)) if self.shaping_rewards else 0,
                'avg_effective_shaping_reward': float(np.mean(self.effective_shaping_rewards)) if self.effective_shaping_rewards else 0,
                'avg_shaping_weight': float(np.mean(self.shaping_weights)) if self.shaping_weights else 0,
                'avg_team_reward': float(np.mean(self.team_rewards)) if self.team_rewards else 0,
                'avg_success': float(np.mean(self.successes)) if self.successes else 0,
                'avg_success_rate': float(np.mean(self.success_rates)) if self.success_rates else 0,
                'avg_episode_steps': float(np.mean(self.episode_steps)) if self.episode_steps else 0,
                'avg_landmark_coverage': float(np.mean(self.landmark_coverages)) if self.landmark_coverages else 0,
                'avg_min_landmark_distance': float(np.mean(self.min_landmark_distances)) if self.min_landmark_distances else 0,
            }
        }

        if extra is not None:
            metrics['extra'] = extra
        
        metrics_file = self.run_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[Log] Saved metrics to {metrics_file}")
    
    def get_stats(self):
        """Get current statistics."""
        if len(self.rewards) == 0:
            return {}
        
        success_window = self.success_rates[-10:] if self.success_rates else []
        success_alias_window = self.successes[-10:] if self.successes else []
        step_window = self.episode_steps[-10:] if self.episode_steps else []
        coverage_window = self.landmark_coverages[-10:] if self.landmark_coverages else []
        min_dist_window = self.min_landmark_distances[-10:] if self.min_landmark_distances else []
        return {
            'episode': self.episodes[-1] if self.episodes else 0,
            'avg_reward': np.mean(self.rewards[-10:]),
            'avg_actor_loss': np.mean(self.actor_losses[-10:]),
            'avg_critic_loss': np.mean(self.critic_losses[-10:]),
            'avg_success': float(np.mean(success_alias_window)) if success_alias_window else 0.0,
            'avg_success_rate': float(np.mean(success_window)) if success_window else 0.0,
            'avg_episode_steps': float(np.mean(step_window)) if step_window else 0.0,
            'avg_landmark_coverage': float(np.mean(coverage_window)) if coverage_window else 0.0,
            'avg_min_landmark_distance': float(np.mean(min_dist_window)) if min_dist_window else 0.0,
            'final_reward': float(self.rewards[-1]),
        }
