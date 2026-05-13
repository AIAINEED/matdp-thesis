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
        self.mse_means = []
        self.actor_grad_norms = []
        self.critic_grad_norms = []
        self.nan_skips = []
        
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
            self.mse_means.append(float(diag.get('mse_mean', 0.0)))
            self.actor_grad_norms.append(float(diag.get('actor_grad_norm', 0.0)))
            self.critic_grad_norms.append(float(diag.get('critic_grad_norm', 0.0)))
            self.nan_skips.append(int(diag.get('nan_skips', 0)))
        else:
            # keep lengths consistent
            self.adv_means.append(0.0)
            self.adv_stds.append(0.0)
            self.approx_kls.append(0.0)
            self.mse_means.append(0.0)
            self.actor_grad_norms.append(0.0)
            self.critic_grad_norms.append(0.0)
            self.nan_skips.append(0)
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
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.reshape(-1)
        fig.subplots_adjust(bottom=0.14, top=0.92, hspace=0.35, wspace=0.28)

        footer_items = []
        for key in ("max_steps", "episodes", "actor_lr", "critic_lr", "method","n_agents"):
            if key in self.plot_meta and self.plot_meta[key] is not None:
                footer_items.append(f"{key}={self.plot_meta[key]}")
        footer_text = " | ".join(footer_items)
        
        # Reward plot
        ax = axes[0]
        ax.plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Episode Reward')
        ma_rewards = self._moving_average(self.rewards, window=50)
        ax.plot(self.episodes[:len(ma_rewards)], ma_rewards, 'b-', linewidth=2, label='MA-50 Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Actor Loss plot
        ax = axes[1]
        ax.plot(self.episodes, self.actor_losses, 'r-', alpha=0.3, label='Episode ActorLoss')
        ma_actor = self._moving_average(self.actor_losses, window=50)
        ax.plot(self.episodes[:len(ma_actor)], ma_actor, 'r-', linewidth=2, label='MA-50 ActorLoss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Critic Loss plot
        ax = axes[2]
        ax.plot(self.episodes, self.critic_losses, 'g-', alpha=0.3, label='Episode CriticLoss')
        ma_critic = self._moving_average(self.critic_losses, window=50)
        ax.plot(self.episodes[:len(ma_critic)], ma_critic, 'g-', linewidth=2, label='MA-50 CriticLoss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Approx KL plot
        ax = axes[3]
        ax.plot(self.episodes, self.approx_kls, 'm-', alpha=0.3, label='Episode Approx KL')
        ma_kl = self._moving_average(self.approx_kls, window=50)
        ax.plot(self.episodes[:len(ma_kl)], ma_kl, 'm-', linewidth=2, label='MA-50 Approx KL')
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
        ax.plot(self.episodes[:len(ma_actor_grad)], ma_actor_grad, color='tab:orange', linewidth=2, label='MA-50 Actor Grad Norm')
        ax.plot(self.episodes, self.critic_grad_norms, color='tab:green', alpha=0.35, label='Critic Grad Norm')
        ma_critic_grad = self._moving_average(self.critic_grad_norms, window=50)
        ax.plot(self.episodes[:len(ma_critic_grad)], ma_critic_grad, color='tab:green', linewidth=2, label='MA-50 Critic Grad Norm')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Advantage stats plot
        ax = axes[5]
        ax.plot(self.episodes, self.adv_means, color='tab:blue', alpha=0.35, label='Adv Mean')
        ma_adv_mean = self._moving_average(self.adv_means, window=50)
        ax.plot(self.episodes[:len(ma_adv_mean)], ma_adv_mean, color='tab:blue', linewidth=2, label='MA-50 Adv Mean')
        ax.plot(self.episodes, self.adv_stds, color='tab:red', alpha=0.35, label='Adv Std')
        ma_adv_std = self._moving_average(self.adv_stds, window=50)
        ax.plot(self.episodes[:len(ma_adv_std)], ma_adv_std, color='tab:red', linewidth=2, label='MA-50 Adv Std')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Advantage')
        ax.set_title('Advantage Stats')
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
            'mse_means': self.mse_means,
            'actor_grad_norms': self.actor_grad_norms,
            'critic_grad_norms': self.critic_grad_norms,
            'nan_skips': self.nan_skips,
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
                'avg_actor_grad_norm': float(np.mean(self.actor_grad_norms)) if self.actor_grad_norms else 0,
                'avg_critic_grad_norm': float(np.mean(self.critic_grad_norms)) if self.critic_grad_norms else 0,
                'avg_nan_skips': float(np.mean(self.nan_skips)) if self.nan_skips else 0,
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
        
        return {
            'episode': self.episodes[-1] if self.episodes else 0,
            'avg_reward': np.mean(self.rewards[-10:]),
            'avg_actor_loss': np.mean(self.actor_losses[-10:]),
            'avg_critic_loss': np.mean(self.critic_losses[-10:]),
        }
