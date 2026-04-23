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
        self.mse_means = []
        self.actor_grad_norms = []
        self.critic_grad_norms = []
        
        print(f"[Logger] Run ID: {self.run_id}")
        print(f"[Logger] Results will be saved to: {self.run_dir}")
    
    def record(self, episode, reward, actor_loss, critic_loss, diag=None):
        """Record metrics from one episode."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        if diag is not None:
            self.adv_means.append(float(diag.get('adv_mean', 0.0)))
            self.adv_stds.append(float(diag.get('adv_std', 0.0)))
            self.mse_means.append(float(diag.get('mse_mean', 0.0)))
            self.actor_grad_norms.append(float(diag.get('actor_grad_norm', 0.0)))
            self.critic_grad_norms.append(float(diag.get('critic_grad_norm', 0.0)))
        else:
            # keep lengths consistent
            self.adv_means.append(0.0)
            self.adv_stds.append(0.0)
            self.mse_means.append(0.0)
            self.actor_grad_norms.append(0.0)
            self.critic_grad_norms.append(0.0)
    def _moving_average(self, values, window=10):
        """Compute moving average."""
        if len(values) < window:
            return values
        ma = np.convolve(values, np.ones(window) / window, mode='valid')
        return np.concatenate([values[:window-1], ma])
    
    def plot(self, save=True):
        """Plot and save training curves."""
        if len(self.episodes) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.28)

        footer_items = []
        for key in ("max_steps", "episodes", "actor_lr", "critic_lr", "method"):
            if key in self.plot_meta and self.plot_meta[key] is not None:
                footer_items.append(f"{key}={self.plot_meta[key]}")
        footer_text = " | ".join(footer_items)
        
        # Reward plot
        ax = axes[0]
        ax.plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Episode Reward')
        ma_rewards = self._moving_average(self.rewards, window=10)
        ax.plot(self.episodes[:len(ma_rewards)], ma_rewards, 'b-', linewidth=2, label='MA-10 Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Actor Loss plot
        ax = axes[1]
        ax.plot(self.episodes, self.actor_losses, 'r-', alpha=0.3, label='Episode ActorLoss')
        ma_actor = self._moving_average(self.actor_losses, window=10)
        ax.plot(self.episodes[:len(ma_actor)], ma_actor, 'r-', linewidth=2, label='MA-10 ActorLoss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Critic Loss plot
        ax = axes[2]
        ax.plot(self.episodes, self.critic_losses, 'g-', alpha=0.3, label='Episode CriticLoss')
        ma_critic = self._moving_average(self.critic_losses, window=10)
        ax.plot(self.episodes[:len(ma_critic)], ma_critic, 'g-', linewidth=2, label='MA-10 CriticLoss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
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
            'mse_means': self.mse_means,
            'actor_grad_norms': self.actor_grad_norms,
            'critic_grad_norms': self.critic_grad_norms,
            'summary': {
                'final_reward': float(self.rewards[-1]) if self.rewards else 0,
                'avg_reward_last_10': float(np.mean(self.rewards[-10:])) if len(self.rewards) >= 10 else float(np.mean(self.rewards)),
                'avg_reward': float(np.mean(self.rewards)) if self.rewards else 0,
                'avg_actor_loss': float(np.mean(self.actor_losses)) if self.actor_losses else 0,
                'avg_critic_loss': float(np.mean(self.critic_losses)) if self.critic_losses else 0,
                'avg_adv_mean': float(np.mean(self.adv_means)) if self.adv_means else 0,
                'avg_mse_mean': float(np.mean(self.mse_means)) if self.mse_means else 0,
                'avg_actor_grad_norm': float(np.mean(self.actor_grad_norms)) if self.actor_grad_norms else 0,
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
