import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate unique run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        self.episodes = []
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        print(f"[Logger] Run ID: {self.run_id}")
        print(f"[Logger] Results will be saved to: {self.run_dir}")
    
    def record(self, episode, reward, actor_loss, critic_loss):
        """Record metrics from one episode."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
    
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
            'summary': {
                'final_reward': float(self.rewards[-1]) if self.rewards else 0,
                'avg_reward_last_10': float(np.mean(self.rewards[-10:])) if len(self.rewards) >= 10 else float(np.mean(self.rewards)),
                'avg_reward': float(np.mean(self.rewards)) if self.rewards else 0,
                'avg_actor_loss': float(np.mean(self.actor_losses)) if self.actor_losses else 0,
                'avg_critic_loss': float(np.mean(self.critic_losses)) if self.critic_losses else 0,
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
