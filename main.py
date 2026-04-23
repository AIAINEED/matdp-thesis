import argparse
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
from config import Config
from models.actor import Actor
from models.critic import Critic
from env.simple_env import SimpleMultiAgentEnv
from utils.buffer import ReplayBuffer
from utils.logger import TrainingLogger


def parse_args(cfg):
    parser = argparse.ArgumentParser(description="Train MA-DPPO-Seq on PettingZoo MPE simple_spread")
    parser.add_argument("--method", type=str, default=cfg.method, choices=["full", "no_transformer", "no_diffusion"])
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--episodes", type=int, default=cfg.episodes)
    parser.add_argument("--max-steps", type=int, default=cfg.max_steps)
    parser.add_argument("--max-cycles", type=int, default=cfg.max_cycles)
    parser.add_argument("--plot-every", type=int, default=cfg.plot_every)
    parser.add_argument("--log-dir", type=str, default=cfg.log_dir)
    parser.add_argument("--save-checkpoints", type=int, choices=[0, 1], default=int(cfg.save_checkpoints))
    parser.add_argument("--checkpoint-every", type=int, default=cfg.checkpoint_every)
    parser.add_argument("--resume", type=str, default=cfg.resume_path)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--bc-warmup-episodes", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def save_checkpoint(run_dir, tag, episode, actor, critic, opt_actor, opt_critic):
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pt"
    torch.save(
        {
            "episode": episode,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "opt_actor": opt_actor.state_dict(),
            "opt_critic": opt_critic.state_dict(),
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved to {ckpt_path}")


def load_checkpoint(ckpt_path, device, actor, critic, opt_actor, opt_critic):
    ckpt = torch.load(ckpt_path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    opt_actor.load_state_dict(ckpt["opt_actor"])
    opt_critic.load_state_dict(ckpt["opt_critic"])
    last_episode = int(ckpt.get("episode", -1))
    print(f"[Checkpoint] Loaded from {ckpt_path} (last_episode={last_episode})")
    return last_episode


# ================== UPDATE ==================
def update(actor, critic, buffer, cfg, opt_actor, opt_critic, beta_override=None):
    states, actions, advantages, returns = buffer.get()

    assert states.ndim == 4, f"states should be [T,N,H,D], got {states.shape}"
    assert actions.ndim == 3, f"actions should be [T,N,A], got {actions.shape}"
    assert advantages.ndim == 1, f"advantages should be [T], got {advantages.shape}"
    assert returns.ndim == 1, f"returns should be [T], got {returns.shape}"
    assert states.shape[0] == actions.shape[0] == advantages.shape[0] == returns.shape[0], (
        f"time dimension mismatch: states={states.shape[0]}, actions={actions.shape[0]}, "
        f"adv={advantages.shape[0]}, returns={returns.shape[0]}"
    )

    # 【修复 1：标准化 Advantage】
    # 将 Advantage 标准化为均值为 0，方差为 1 的分布，这能极大稳定网络更新
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    norm_advantages = (advantages - adv_mean) / adv_std
    norm_advantages = norm_advantages.detach()  # 必须切断梯度，防止 Actor 更新 干扰 Critic

    # 【修复 2：替代 PPO Clip 的安全机制】
    # 将异常大的 Advantage 截断，防止单步梯度爆炸，保护脆弱的 Diffusion 模型
    norm_advantages = torch.clamp(norm_advantages, min=-2.0, max=2.0)

    for _ in range(cfg.epochs):

        # ===== Critic =====
        values = critic(states)
        critic_loss = ((values - returns) ** 2).mean()
        if torch.isnan(critic_loss):
            raise RuntimeError("critic_loss is NaN")

        opt_critic.zero_grad()
        critic_loss.backward()
        # compute critic grad norm before clipping
        critic_grad_norm = 0.0
        total = 0.0
        for p in critic.parameters():
            if p.grad is not None:
                total += float((p.grad.data ** 2).sum().item())
        critic_grad_norm = float(total ** 0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        opt_critic.step()

        # ===== Actor =====
        h_all = actor.encode(states)

        actor_loss = 0.0
        mse_sum = 0.0
        mse_count = 0

        for i in range(cfg.n_agents):
            h = h_all[:, i]
            a = actions[:, i]

            # [B] 获取扩散模型去噪步骤的均方误差
            mse_loss = actor.policy.loss(a, h)  
            # accumulate mse stats
            try:
                mse_sum += float(mse_loss.mean().item())
                mse_count += 1
            except Exception:
                pass
            
            # 【终极修复：使用指数优势加权 (AWAC/DPPO 标配)】
            # beta 是温度系数，通常设为 1.0 或 2.0。它控制着对"好动作"的偏好程度。
            beta = cfg.beta if beta_override is None else float(beta_override)
            weights = torch.exp(beta * norm_advantages)
            
            # 加上一个 clamp 防止某些极其惊艳的动作导致权重单步过大 (比如超过 10 倍)
            weights = torch.clamp(weights, max=10.0)
            
            # 现在的 loss 永远在被最小化，坏动作只是 weight 接近 0 而已被忽略
            loss_i = (mse_loss * weights).mean()
            
            actor_loss += loss_i

        actor_loss /= cfg.n_agents
        if torch.isnan(actor_loss):
            raise RuntimeError("actor_loss is NaN")

        opt_actor.zero_grad()
        actor_loss.backward()
        # compute actor grad norm before clipping
        actor_grad_norm = 0.0
        total_a = 0.0
        for p in actor.parameters():
            if p.grad is not None:
                total_a += float((p.grad.data ** 2).sum().item())
        actor_grad_norm = float(total_a ** 0.5)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        opt_actor.step()

    # compute mse mean across agents/time
    try:
        mse_mean = float(mse_sum / mse_count)
    except Exception:
        mse_mean = 0.0

    diag = {
        'adv_mean': float(adv_mean.item()),
        'adv_std': float(adv_std),
        'mse_mean': mse_mean,
        'actor_grad_norm': float(actor_grad_norm) if 'actor_grad_norm' in locals() else 0.0,
        'critic_grad_norm': float(critic_grad_norm) if 'critic_grad_norm' in locals() else 0.0,
    }

    return actor_loss.item(), critic_loss.item(), diag


def train(args):
    cfg = Config()
    set_seed(args.seed)

    if args.actor_lr is not None:
        cfg.actor_lr = args.actor_lr
    if args.critic_lr is not None:
        cfg.critic_lr = args.critic_lr
    if args.beta is not None:
        cfg.beta = args.beta
    if args.horizon is not None:
        cfg.horizon = args.horizon
    if args.diffusion_steps is not None:
        cfg.diffusion_steps = args.diffusion_steps

    if args.method == "no_transformer":
        cfg.encoder_type = "mlp"
        cfg.policy_type = "diffusion"
    elif args.method == "no_diffusion":
        cfg.encoder_type = "transformer"
        cfg.policy_type = "gaussian"
    else:
        cfg.encoder_type = "transformer"
        cfg.policy_type = "diffusion"

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    env = SimpleMultiAgentEnv(cfg.n_agents, max_cycles=args.max_cycles, seed=args.seed)

    cfg.n_agents = env.n_agents
    cfg.obs_dim = env.obs_dim
    cfg.action_dim = env.action_dim

    logger = TrainingLogger(
        log_dir=args.log_dir,
        plot_meta={
            "max_steps": args.max_steps,
            "episodes": args.episodes,
            "actor_lr": cfg.actor_lr,
            "critic_lr": cfg.critic_lr,
            "method": args.method,
        },
    )
    actor = Actor(cfg).to(device)
    critic = Critic(cfg).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    buffer = ReplayBuffer(cfg, device)
    start_episode = 0
    # Track best episode reward for checkpointing (save only best)
    best_reward = float("-inf")

    if args.resume:
        start_episode = load_checkpoint(
            ckpt_path=args.resume,
            device=device,
            actor=actor,
            critic=critic,
            opt_actor=opt_actor,
            opt_critic=opt_critic,
        ) + 1

    print(
        f"[Run Config] method={args.method} seed={args.seed} episodes={args.episodes} max_steps={args.max_steps} "
        f"max_cycles={args.max_cycles} n_agents={cfg.n_agents} obs_dim={cfg.obs_dim} action_dim={cfg.action_dim} "
        f"encoder={cfg.encoder_type} policy={cfg.policy_type} "
        f"actor_lr={cfg.actor_lr} critic_lr={cfg.critic_lr} horizon={cfg.horizon} diffusion_steps={cfg.diffusion_steps} "
        f"beta={cfg.beta} "
        f"save_checkpoints={bool(args.save_checkpoints)} checkpoint_every={args.checkpoint_every} "
        f"resume={'none' if not args.resume else args.resume} start_episode={start_episode}"
    )

    try:
        episode_iter = range(start_episode, args.episodes)
        use_tqdm = tqdm is not None
        if use_tqdm:
            episode_iter = tqdm(
                episode_iter,
                total=max(args.episodes - start_episode, 0),
                desc=f"Train {args.method} seed={args.seed}",
                dynamic_ncols=True,
            )

        for episode in episode_iter:
            obs = env.reset()  # [N, obs_dim]
            obs_t = torch.tensor(obs.tolist(), dtype=torch.float32, device=device)

            # init obs_seq with the first obs
            obs_seq = obs_t.unsqueeze(1).repeat(1, cfg.horizon, 1).unsqueeze(0)  # [1, N, H, D]

            total_reward = 0.0

            for _ in range(args.max_steps):
                with torch.no_grad():
                    actions, _ = actor(obs_seq)
                    value = critic(obs_seq).item()

                actions_env = actions[0].detach().cpu().numpy()
                
                # 这里强制截断，无论网络输出多离谱，传给环境的绝对不会超标
                actions_env = np.clip(actions_env, 0.0, 1.0)
                
                # 再转成 list 传给环境
                next_obs, reward, done, _ = env.step(actions_env.tolist())
                
                # scale reward and account for team-sum aggregation
                # divide also by number of agents to match per-agent magnitude
                scaled_reward = reward / (10.0 * cfg.n_agents)

                buffer.add(obs_seq.clone(), actions, scaled_reward, done, value)
                total_reward += reward

                next_obs_t = torch.tensor(next_obs.tolist(), dtype=torch.float32, device=device)
                next_obs_t = next_obs_t.unsqueeze(0).unsqueeze(2)  # [1, N, 1, D]
                obs_seq = torch.cat([obs_seq[:, :, 1:], next_obs_t], dim=2)

                if done:
                    break

            with torch.no_grad():
                last_value = critic(obs_seq).item()

            buffer.compute_returns_advantages(last_value)
            beta_now = 0.0 if episode < args.bc_warmup_episodes else cfg.beta
            actor_loss, critic_loss, diag = update(
                actor,
                critic,
                buffer,
                cfg,
                opt_actor,
                opt_critic,
                beta_override=beta_now,
            )
            buffer.clear()

            diag["beta_used"] = float(beta_now)
            logger.record(episode, total_reward, actor_loss, critic_loss, diag=diag)

            # 线性学习率衰减 (Linear LR Decay)
            # 1. 计算剩余比例 (从 1.0 匀速降到 0.0)
            fraction = 1.0 - (episode / args.episodes)
            
            # 2. 计算当前学习率 (兜底 1e-6)
            lr_actor_now = max(cfg.actor_lr * fraction, 1e-6)
            lr_critic_now = max(cfg.critic_lr * fraction, 1e-6)
            
            # 3. 注入优化器
            for param_group in opt_actor.param_groups:
                param_group['lr'] = lr_actor_now
            for param_group in opt_critic.param_groups:
                param_group['lr'] = lr_critic_now

            if use_tqdm:
                episode_iter.set_postfix(
                    reward=f"{total_reward:.2f}",
                    actor=f"{actor_loss:.4f}",
                    critic=f"{critic_loss:.4f}",
                )
            else:
                print(
                    f"Episode {episode:03d} | "
                    f"Reward {total_reward:.2f} | "
                    f"ActorLoss {actor_loss:.4f} | "
                    f"CriticLoss {critic_loss:.4f}"
                )

            if args.plot_every > 0 and (episode + 1) % args.plot_every == 0:
                logger.plot(save=True)

            # Save only the best model (by total_reward) if checkpointing enabled
            if bool(args.save_checkpoints):
                if total_reward > best_reward:
                    best_reward = total_reward
                    save_checkpoint(
                        run_dir=logger.run_dir,
                        tag="best",
                        episode=episode,
                        actor=actor,
                        critic=critic,
                        opt_actor=opt_actor,
                        opt_critic=opt_critic,
                    )
                    print(f"[Checkpoint] New best model saved at episode {episode} (reward={best_reward:.2f})")

        logger.plot(save=True)
        # If no best checkpoint was ever saved, still save one final checkpoint.
        if bool(args.save_checkpoints) and best_reward == float("-inf"):
            save_checkpoint(
                run_dir=logger.run_dir,
                tag="final",
                episode=args.episodes - 1,
                actor=actor,
                critic=critic,
                opt_actor=opt_actor,
                opt_critic=opt_critic,
            )
        logger.save_metrics(
            extra={
                "seed": args.seed,
                "method": args.method,
                "encoder_type": cfg.encoder_type,
                "policy_type": cfg.policy_type,
                "actor_lr": cfg.actor_lr,
                "critic_lr": cfg.critic_lr,
                "beta": cfg.beta,
                "horizon": cfg.horizon,
                "diffusion_steps": cfg.diffusion_steps,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "max_cycles": args.max_cycles,
                "device": str(device),
                "n_agents": cfg.n_agents,
                "obs_dim": cfg.obs_dim,
                "action_dim": cfg.action_dim,
                "git_commit": get_git_commit(),
                "save_checkpoints": bool(args.save_checkpoints),
                "checkpoint_every": args.checkpoint_every,
                "resume_from": args.resume if args.resume else None,
                "start_episode": start_episode,
            }
        )
        print(f"[Training Complete] Results saved to {logger.run_dir}")
        print(f"[Training Complete] Quick access: {args.log_dir}/latest/")
    finally:
        env.close()


if __name__ == "__main__":
    train(parse_args(Config()))