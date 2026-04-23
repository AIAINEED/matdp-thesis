import torch
import numpy as np

from config import Config
from main import (
    parse_args,
    set_seed,
    get_git_commit,
    save_checkpoint,
    load_checkpoint,
    update,
)
from models.actor import Actor
from models.critic import Critic
from env.simple_env import SimpleMultiAgentEnv
from utils.buffer import ReplayBuffer
from utils.logger import TrainingLogger
 

def yi_squash_action_to_env(actions_raw: torch.Tensor) -> torch.Tensor:
    """Map raw action outputs to the environment range [0, 1] via tanh."""
    return (torch.tanh(actions_raw) + 1.0) / 2.0


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
        use_tqdm = False
        try:
            from tqdm.auto import tqdm
            use_tqdm = True
        except Exception:
            tqdm = None

        if use_tqdm:
            episode_iter = tqdm(
                episode_iter,
                total=max(args.episodes - start_episode, 0),
                desc=f"Train {args.method} seed={args.seed}",
                dynamic_ncols=True,
            )

        for episode in episode_iter:
            obs = env.reset()
            obs_t = torch.tensor(obs.tolist(), dtype=torch.float32, device=device)

            obs_seq = obs_t.unsqueeze(1).repeat(1, cfg.horizon, 1).unsqueeze(0)

            total_reward = 0.0

            for _ in range(args.max_steps):
                with torch.no_grad():
                    actions_raw, _ = actor(obs_seq)
                    value = critic(obs_seq).item()

                actions_exec = yi_squash_action_to_env(actions_raw[0]).detach().cpu().numpy()
                next_obs, reward, done, _ = env.step(actions_exec.tolist())

                scaled_reward = reward / 10.0

                # Buffer stores the raw, unconstrained action output.
                buffer.add(obs_seq.clone(), actions_raw, scaled_reward, done, value)
                total_reward += reward

                next_obs_t = torch.tensor(next_obs.tolist(), dtype=torch.float32, device=device)
                next_obs_t = next_obs_t.unsqueeze(0).unsqueeze(2)
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

            fraction = 1.0 - (episode / args.episodes)
            lr_actor_now = max(cfg.actor_lr * fraction, 1e-6)
            lr_critic_now = max(cfg.critic_lr * fraction, 1e-6)

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
                "action_squash": "tanh_to_[0,1]",
                "buffer_action_type": "raw_unconstrained",
            }
        )
        print(f"[Training Complete] Results saved to {logger.run_dir}")
        print(f"[Training Complete] Quick access: {args.log_dir}/latest/")
    finally:
        env.close()


if __name__ == "__main__":
    train(parse_args(Config())) 