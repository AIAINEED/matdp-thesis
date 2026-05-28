import argparse
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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
    parser.add_argument("--n-agents", type=int, default=cfg.n_agents)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--episodes", type=int, default=cfg.episodes)
    parser.add_argument("--max-steps", type=int, default=cfg.max_steps)
    parser.add_argument("--max-cycles", type=int, default=cfg.max_cycles)
    parser.add_argument("--plot-every", type=int, default=cfg.plot_every)
    parser.add_argument("--log-dir", type=str, default=cfg.log_dir)
    parser.add_argument("--save-checkpoints", type=int, choices=[0, 1], default=int(cfg.save_checkpoints))
    parser.add_argument("--checkpoint-every", type=int, default=cfg.checkpoint_every)
    parser.add_argument("--resume", type=str, default=cfg.resume_path)
    parser.add_argument("--debug-update", type=int, choices=[0, 1], default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--diffusion-recon-weight", type=float, default=None)
    parser.add_argument("--clip-param", type=float, default=cfg.clip_param)
    parser.add_argument("--value-clip-param", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--actor-lr-decay", type=int, choices=[0, 1], default=None)
    parser.add_argument("--critic-lr-decay", type=int, choices=[0, 1], default=None)
    parser.add_argument("--lr-decay-episodes", type=int, default=None)
    parser.add_argument("--lr-decay-short-frac", type=float, default=None)
    parser.add_argument("--lr-decay-min", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=cfg.ppo_epochs)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--diffusion-action-eps", type=float, default=None)
    parser.add_argument("--diffusion-min-std", type=float, default=None)
    parser.add_argument("--pbrs", type=int, choices=[0, 1], default=None)
    parser.add_argument("--pbrs-scale", type=float, default=None)
    parser.add_argument("--pbrs-anneal", type=int, choices=[0, 1], default=None)
    parser.add_argument("--pbrs-anneal-start", type=float, default=None)
    parser.add_argument("--pbrs-anneal-end", type=float, default=None)
    parser.add_argument("--coverage-bonus", type=float, default=None)
    parser.add_argument("--fov-mask", type=int, choices=[0, 1], default=None)
    parser.add_argument("--fov-radius", type=float, default=None)
    parser.add_argument("--fov-visibility", type=int, choices=[0, 1], default=None)
    parser.add_argument("--terminate-on-success", type=int, choices=[0, 1], default=None)
    parser.add_argument("--enable-joint-policy", type=int, choices=[0, 1], default=1)
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


# ================== UPDATE ==================
def update(actor, critic, buffer, cfg, opt_actor, opt_critic, episode=None):
    states, actions, advantages, returns, step_log_probs, chains, ks, noises = buffer.get()
    old_values = torch.tensor(buffer.values, dtype=torch.float32, device=returns.device)
    if getattr(cfg, "debug_update", False):
        print(
            f"[ARCH CHECK] update episode={episode} source_T={states.shape[0]} "
            f"rollout_horizon={cfg.rollout_horizon} n_agents={states.shape[1]}"
        )

    assert states.ndim == 4, f"states should be [T,N,H,D], got {states.shape}"
    assert actions.ndim == 3, f"actions should be [T,N,A], got {actions.shape}"
    assert advantages.ndim == 1, f"advantages should be [T], got {advantages.shape}"
    assert returns.ndim == 1, f"returns should be [T], got {returns.shape}"
    assert old_values.ndim == 1, f"old_values should be [T], got {old_values.shape}"
    assert old_values.shape[0] == returns.shape[0], (
        f"value time dimension mismatch: old_values={old_values.shape[0]}, returns={returns.shape[0]}"
    )

    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False) + 1e-8
    norm_advantages = (advantages - adv_mean) / adv_std
    norm_advantages = norm_advantages.detach()

    if not torch.isfinite(states).all():
        raise RuntimeError("states contain non-finite values before PPO update")
    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise RuntimeError("advantages/returns contain non-finite values before PPO update")

    def clipped_value_loss(current_values, old_values_batch, returns_batch):
        value_clip_param = getattr(cfg, "value_clip_param", cfg.clip_param)
        if value_clip_param is None or value_clip_param <= 0.0:
            return F.smooth_l1_loss(current_values, returns_batch)
        clipped_values = old_values_batch + torch.clamp(
            current_values - old_values_batch, -value_clip_param, value_clip_param
        )
        loss_unclipped = F.smooth_l1_loss(current_values, returns_batch, reduction="none")
        loss_clipped = F.smooth_l1_loss(clipped_values, returns_batch, reduction="none")
        return torch.max(loss_unclipped, loss_clipped).mean()

    def sanitize_gradients(module):
        sanitized = 0
        for p in module.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                sanitized += 1
        return sanitized
    
    # 诊断：验证 norm_advantages 的中心化
    norm_adv_mean = float(norm_advantages.mean().item())
    if abs(norm_adv_mean) > 1e-2:
        print(f"[WARNING] norm_advantages mean is {norm_adv_mean}, should be close to 0!")

    legacy_old_log_probs = (
        torch.cat(buffer.log_probs, dim=0).to(returns.device) if buffer.log_probs else None
    )

    use_stepwise_diffusion = (
        chains is not None
        and step_log_probs is not None
        and hasattr(actor.policy, "get_step_log_prob")
    )

    if use_stepwise_diffusion:
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_approx_kl = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        total_diffusion_mse = 0.0
        # ratio diagnostics totals
        total_clip_frac = 0.0
        total_ratio_mean = 0.0
        total_ratio_std = 0.0
        total_raw_log_ratio_mean = 0.0
        total_raw_log_ratio_std = 0.0
        epochs_used = 0
        nan_skips = 0
        valid_updates = 0
        skip_reasons = {
            "critic_loss": 0,
            "critic_grad": 0,
            "actor_log_prob": 0,
            "actor_loss": 0,
            "actor_grad": 0,
            "actor_grad_sanitized": 0,
        }

        T = states.shape[0]
        K = chains.shape[2] - 1
        n_agents = chains.shape[1]
        indices = torch.arange(T, device=states.device)

        for epoch in range(cfg.ppo_epochs):
            epochs_used += 1
            perm = indices[torch.randperm(T, device=states.device)]
            stop_epoch = False

            for start in range(0, T, cfg.batch_size):
                mb_idx = perm[start : start + cfg.batch_size]
                states_mb = states[mb_idx]
                returns_mb = returns[mb_idx]
                old_values_mb = old_values[mb_idx]
                chains_mb = chains[mb_idx]
                actions_mb = actions[mb_idx]
                step_log_probs_mb = step_log_probs[mb_idx]

                values_mb = critic(states_mb)
                critic_loss_mb = clipped_value_loss(values_mb, old_values_mb, returns_mb)

                if not torch.isfinite(critic_loss_mb):
                    nan_skips += 1
                    skip_reasons["critic_loss"] += 1
                    continue

                opt_critic.zero_grad()
                critic_loss_mb.backward()
                critic_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                )
                if not torch.isfinite(torch.tensor(critic_grad_norm, device=states.device)):
                    opt_critic.zero_grad()
                    nan_skips += 1
                    skip_reasons["critic_grad"] += 1
                    continue
                opt_critic.step()

                h_all_mb = actor.encode(states_mb)
                use_joint_policy = getattr(cfg, 'enable_joint_policy', False) and hasattr(actor, 'joint_policy')
                norm_adv_mb = norm_advantages[mb_idx]

                actor_loss_sum = 0.0
                approx_kl_sum = 0.0
                actor_non_finite = False
                debug_first_batch = bool(getattr(cfg, "debug_policy_update", False)) and (epoch == 0 and int(start) == 0)
                actor_param_before = None
                if debug_first_batch:
                    actor_param_before = torch.cat(
                        [p.detach().flatten() for p in actor.parameters() if p.requires_grad]
                    )

                if use_joint_policy:
                    joint_flat_mb = actor.team_encode(states_mb)[1]
                    joint_action_dim = actor.joint_policy.joint_action_dim
                    chain_flat_mb = chains_mb.permute(0, 2, 1, 3).reshape(states_mb.shape[0], K + 1, joint_action_dim)
                    joint_adv_mb = norm_adv_mb

                    for model_t in reversed(range(K)):
                        chain_idx = K - 1 - model_t
                        a_t = chain_flat_mb[:, chain_idx]
                        a_prev = chain_flat_mb[:, chain_idx + 1]
                        # Keep old/new step log-prob shapes aligned as [B].
                        # Using unsqueeze(-1) here would broadcast current-old to [B, B]
                        # and inject cross-sample errors into KL/ratio.
                        old_lp_t = step_log_probs_mb[:, model_t]
                        t_tensor = torch.full(
                            (states_mb.shape[0],), model_t, device=states_mb.device, dtype=torch.long
                        )

                        current_lp_t = actor.joint_policy.get_step_log_prob(
                            None,
                            a_t,
                            a_prev,
                            t_tensor,
                            h=joint_flat_mb,
                        )
                        if not torch.isfinite(current_lp_t).all():
                            actor_non_finite = True
                            break

                        raw_log_ratio = current_lp_t - old_lp_t
                        log_ratio = torch.clamp(raw_log_ratio, min=-2.0, max=2.0)
                        ratio = torch.exp(log_ratio)
                        # collect for diagnostics
                        try:
                            raw_list.append(raw_log_ratio.detach())
                            ratio_list.append(ratio.detach())
                        except NameError:
                            raw_list = [raw_log_ratio.detach()]
                            ratio_list = [ratio.detach()]
                        surr1 = ratio * joint_adv_mb
                        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * joint_adv_mb
                        step_loss = -torch.min(surr1, surr2).mean()

                        actor_loss_sum = actor_loss_sum + step_loss
                        approx_kl_t = 0.5 * (raw_log_ratio.detach() ** 2).mean()
                        approx_kl_sum += float(approx_kl_t.item())

                        if debug_first_batch and model_t == K - 1:
                            print(
                                "[DEBUG joint stepwise] "
                                f"current_lp_requires_grad={bool(current_lp_t.requires_grad)} "
                                f"step_loss_requires_grad={bool(step_loss.requires_grad)} "
                                f"old_lp_mean={float(old_lp_t.mean().item()):.6f} "
                                f"current_lp_mean={float(current_lp_t.mean().item()):.6f} "
                                f"ratio_mean={float(ratio.mean().item()):.6f} "
                                f"step_loss={float(step_loss.item()):.6f} "
                                f"adv_mean={float(joint_adv_mb.mean().item()):.6f}",
                                flush=True,
                            )
                else:
                    norm_adv_mb = norm_adv_mb.unsqueeze(-1).expand(-1, n_agents)

                    for model_t in reversed(range(K)):
                        # chains are stored as [a_K, a_{K-1}, ..., a_0]
                        # so step t uses current=a_{t+1} and next=a_t
                        chain_idx = K - 1 - model_t
                        a_t = chains_mb[:, :, chain_idx]
                        a_prev = chains_mb[:, :, chain_idx + 1]
                        old_lp_t = step_log_probs_mb[:, :, model_t]
                        t_tensor = torch.full(
                            (states_mb.shape[0],), model_t, device=states_mb.device, dtype=torch.long
                        )

                        current_lp_agents = []
                        for agent_idx in range(n_agents):
                            current_lp_agents.append(
                                actor.policy.get_step_log_prob(
                                    states_mb[:, agent_idx],
                                    a_t[:, agent_idx],
                                    a_prev[:, agent_idx],
                                    t_tensor,
                                    h=h_all_mb[:, agent_idx],
                                )
                            )

                        current_lp_t = torch.stack(current_lp_agents, dim=1)
                        if not torch.isfinite(current_lp_t).all():
                            actor_non_finite = True
                            break

                        raw_log_ratio = current_lp_t - old_lp_t
                        log_ratio = torch.clamp(raw_log_ratio, min=-2.0, max=2.0)
                        ratio = torch.exp(log_ratio)
                        try:
                            raw_list.append(raw_log_ratio.detach())
                            ratio_list.append(ratio.detach())
                        except NameError:
                            raw_list = [raw_log_ratio.detach()]
                            ratio_list = [ratio.detach()]
                        surr1 = ratio * norm_adv_mb
                        surr2 = torch.clamp(
                            ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param
                        ) * norm_adv_mb
                        step_loss = -torch.min(surr1, surr2).mean()

                        actor_loss_sum = actor_loss_sum + step_loss
                        approx_kl_t = 0.5 * (raw_log_ratio.detach() ** 2).mean()
                        approx_kl_sum += float(approx_kl_t.item())

                        if debug_first_batch and model_t == K - 1:
                            print(
                                "[DEBUG stepwise] "
                                f"current_lp_requires_grad={bool(current_lp_t.requires_grad)} "
                                f"step_loss_requires_grad={bool(step_loss.requires_grad)} "
                                f"old_lp_mean={float(old_lp_t.mean().item()):.6f} "
                                f"current_lp_mean={float(current_lp_t.mean().item()):.6f} "
                                f"ratio_mean={float(ratio.mean().item()):.6f} "
                                f"step_loss={float(step_loss.item()):.6f} "
                                f"adv_mean={float(norm_adv_mb.mean().item()):.6f}",
                                flush=True,
                            )

                if actor_non_finite:
                    nan_skips += 1
                    skip_reasons["actor_log_prob"] += 1
                    continue

                actor_loss_mb = actor_loss_sum / K
                approx_kl_mb = approx_kl_sum / K

                # compute minibatch ratio diagnostics
                if 'raw_list' in locals() and len(raw_list) > 0:
                    raw_all = torch.cat([r.view(-1) for r in raw_list])
                    raw_mean = float(raw_all.mean().cpu().item())
                    raw_std = float(raw_all.std(unbiased=False).cpu().item())
                    log_clamped = torch.clamp(raw_all, min=-2.0, max=2.0)
                    ratio_all = torch.exp(log_clamped)
                    ratio_mean = float(ratio_all.mean().cpu().item())
                    ratio_std = float(ratio_all.std(unbiased=False).cpu().item())
                    clip_frac = float(((ratio_all > 1.0 + cfg.clip_param) | (ratio_all < 1.0 - cfg.clip_param)).float().mean().cpu().item())
                else:
                    raw_mean = raw_std = ratio_mean = ratio_std = clip_frac = 0.0

                total_raw_log_ratio_mean += raw_mean
                total_raw_log_ratio_std += raw_std
                total_ratio_mean += ratio_mean
                total_ratio_std += ratio_std
                total_clip_frac += clip_frac
                # clear local lists for next minibatch
                if 'raw_list' in locals():
                    del raw_list
                if 'ratio_list' in locals():
                    del ratio_list

                if not torch.isfinite(actor_loss_mb):
                    nan_skips += 1
                    skip_reasons["actor_loss"] += 1
                    continue

                # diffusion reconstruction regularization
                diffusion_mse_mb = 0.0
                if getattr(cfg, 'diffusion_recon_weight', 0.0) > 0.0:
                    if getattr(cfg, 'enable_joint_policy', False) and hasattr(actor, 'joint_policy'):
                        # joint reconstruction loss
                        _, joint_flat_mb = actor.team_encode(states_mb)
                        a0_flat_mb = actions_mb.reshape(actions_mb.shape[0], -1)
                        diff_losses = actor.joint_policy.loss(a0_flat_mb, joint_flat_mb)
                        diffusion_mse_mb = float(diff_losses.mean().item())
                        actor_loss_mb = actor_loss_mb + cfg.diffusion_recon_weight * diff_losses.mean()
                    else:
                        # per-agent reconstruction loss
                        per_agent_losses = []
                        for i in range(n_agents):
                            per_agent_losses.append(actor.policy.loss(actions_mb[:, i], h_all_mb[:, i]))
                        per_agent_losses = torch.stack(per_agent_losses, dim=1)
                        diffusion_mse_mb = float(per_agent_losses.mean().item())
                        actor_loss_mb = actor_loss_mb + cfg.diffusion_recon_weight * per_agent_losses.mean()

                opt_actor.zero_grad()
                actor_loss_mb.backward()
                skip_reasons["actor_grad_sanitized"] += sanitize_gradients(actor)
                actor_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                )
                if not torch.isfinite(torch.tensor(actor_grad_norm, device=states.device)):
                    opt_actor.zero_grad()
                    nan_skips += 1
                    skip_reasons["actor_grad"] += 1
                    continue
                opt_actor.step()

                if debug_first_batch and actor_param_before is not None:
                    actor_param_after = torch.cat(
                        [p.detach().flatten() for p in actor.parameters() if p.requires_grad]
                    )
                    actor_delta_norm = float((actor_param_after - actor_param_before).norm().item())
                    print(
                        "[DEBUG stepwise] "
                        f"actor_delta_norm={actor_delta_norm:.10f} "
                        f"actor_grad_norm={float(actor_grad_norm):.10f} "
                        f"approx_kl_mb={float(approx_kl_mb):.10f} "
                        f"actor_loss_mb={float(actor_loss_mb.item()):.10f}",
                        flush=True,
                    )

                total_actor_loss += float(actor_loss_mb.item())
                total_critic_loss += float(critic_loss_mb.item())
                total_diffusion_mse += float(diffusion_mse_mb)
                total_approx_kl += float(approx_kl_mb)
                total_actor_grad_norm += float(actor_grad_norm)
                total_critic_grad_norm += float(critic_grad_norm)
                valid_updates += 1

                if approx_kl_mb > cfg.target_kl:
                    stop_epoch = True
                    break

            if stop_epoch:
                break

        if valid_updates == 0:
            print(
                f"[PPO WARNING] No valid actor updates in episode {episode}; "
                f"nan_skips={nan_skips}, epochs_used={epochs_used}, T={T}, K={K}, "
                f"reasons={skip_reasons}"
            )
            diag = {
                "adv_mean": float(adv_mean.item()),
                "adv_std": float(adv_std.item()),
                "approx_kl": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "mse_mean": 0.0,
                "clip_param": float(cfg.clip_param),
                "ppo_epochs": int(cfg.ppo_epochs),
                "ppo_epochs_used": int(epochs_used),
                "valid_updates": 0,
                "nan_skips": int(nan_skips),
                "update_failed": 1,
                "skip_critic_loss": int(skip_reasons["critic_loss"]),
                "skip_critic_grad": int(skip_reasons["critic_grad"]),
                "skip_actor_log_prob": int(skip_reasons["actor_log_prob"]),
                "skip_actor_loss": int(skip_reasons["actor_loss"]),
                "skip_actor_grad": int(skip_reasons["actor_grad"]),
                "actor_grad_sanitized": int(skip_reasons["actor_grad_sanitized"]),
            }
            return 0.0, 0.0, diag

        total_batches = valid_updates
        diag = {
            "adv_mean": float(adv_mean.item()),
            "adv_std": float(adv_std.item()),
            "approx_kl": float(total_approx_kl / total_batches),
            "actor_grad_norm": float(total_actor_grad_norm / total_batches),
            "critic_grad_norm": float(total_critic_grad_norm / total_batches),
            "mse_mean": float(total_critic_loss / total_batches),
            "diffusion_mse": float(total_diffusion_mse / total_batches),
            # ratio diagnostics averaged over valid updates
            "clip_frac": float(total_clip_frac / total_batches) if total_batches else 0.0,
            "ratio_mean": float(total_ratio_mean / total_batches) if total_batches else 0.0,
            "ratio_std": float(total_ratio_std / total_batches) if total_batches else 0.0,
            "raw_log_ratio_mean": float(total_raw_log_ratio_mean / total_batches) if total_batches else 0.0,
            "raw_log_ratio_std": float(total_raw_log_ratio_std / total_batches) if total_batches else 0.0,
            "clip_param": float(cfg.clip_param),
            "ppo_epochs": int(cfg.ppo_epochs),
            "ppo_epochs_used": int(epochs_used),
            "valid_updates": int(valid_updates),
            "nan_skips": int(nan_skips),
            "update_failed": 0,
            "skip_critic_loss": int(skip_reasons["critic_loss"]),
            "skip_critic_grad": int(skip_reasons["critic_grad"]),
            "skip_actor_log_prob": int(skip_reasons["actor_log_prob"]),
            "skip_actor_loss": int(skip_reasons["actor_loss"]),
            "skip_actor_grad": int(skip_reasons["actor_grad"]),
            "actor_grad_sanitized": int(skip_reasons["actor_grad_sanitized"]),
        }
        return (
            float(total_actor_loss / total_batches),
            float(total_critic_loss / total_batches),
            diag,
        )

    if legacy_old_log_probs is None or ks is None or noises is None:
        raise RuntimeError("buffer must store old_log_probs, k, and noise for PPO updates")

    old_log_probs = legacy_old_log_probs.squeeze(-1)
    ks = ks.squeeze(-1)

    if norm_advantages.ndim == 1:
        norm_advantages = norm_advantages.unsqueeze(-1).expand(-1, cfg.n_agents)
    if old_log_probs.ndim == 2 and old_log_probs.shape[1] == 1:
        old_log_probs = old_log_probs.expand(-1, cfg.n_agents)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_approx_kl = 0.0
    total_actor_grad_norm = 0.0
    total_critic_grad_norm = 0.0
    total_diffusion_mse = 0.0
    # ratio diagnostics totals
    total_clip_frac = 0.0
    total_ratio_mean = 0.0
    total_ratio_std = 0.0
    total_raw_log_ratio_mean = 0.0
    total_raw_log_ratio_std = 0.0
    epochs_used = 0
    nan_skips = 0
    valid_updates = 0
    skip_reasons = {
        "critic_loss": 0,
        "critic_grad": 0,
        "actor_log_prob": 0,
        "actor_loss": 0,
        "actor_grad": 0,
        "actor_grad_sanitized": 0,
    }

    # Prepare minibatch indices over time dimension
    T = states.shape[0]
    indices = torch.arange(T, device=states.device)

    for epoch in range(cfg.ppo_epochs):
        epochs_used += 1
        stop_epoch = False

        # shuffle indices each epoch
        perm = indices[torch.randperm(T, device=states.device)]

        # iterate minibatches
        for start in range(0, T, cfg.batch_size):
            mb_idx = perm[start:start + cfg.batch_size]

            # ===== Critic (minibatch) =====
            values_mb = critic(states[mb_idx])
            returns_mb = returns[mb_idx]
            old_values_mb = old_values[mb_idx]
            critic_loss_mb = clipped_value_loss(values_mb, old_values_mb, returns_mb)

            if not torch.isfinite(critic_loss_mb):
                nan_skips += 1
                skip_reasons["critic_loss"] += 1
                continue

            opt_critic.zero_grad()
            critic_loss_mb.backward()
            critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5))
            if not torch.isfinite(torch.tensor(critic_grad_norm, device=states.device)):
                opt_critic.zero_grad()
                nan_skips += 1
                skip_reasons["critic_grad"] += 1
                continue
            opt_critic.step()

            # ===== Actor (minibatch) =====
            # use pre-normalized advantages but index minibatch
            norm_adv_mb = norm_advantages[mb_idx]

            # prepare per-agent data for minibatch
            h_all_mb = actor.encode(states[mb_idx])
            actions_mb = actions[mb_idx]
            old_log_probs_mb = old_log_probs[mb_idx]
            ks_mb = ks[mb_idx]
            noises_mb = noises[mb_idx]
            actions_mb = actions[mb_idx]

            actor_loss_mb = 0.0
            approx_kl_sum_mb = 0.0
            approx_kl_count_mb = 0
            actor_non_finite = False

            for i in range(cfg.n_agents):
                h = h_all_mb[:, i]
                a = actions_mb[:, i]

                k_i = ks_mb[:, i]
                noise_i = noises_mb[:, i]
                old_log_prob_i = old_log_probs_mb[:, i]
                new_log_prob_i = actor.policy.get_log_prob(a, h, k_i, noise=noise_i).squeeze(-1)

                if not torch.isfinite(new_log_prob_i).all():
                    actor_non_finite = True
                    break

                raw_log_ratio = new_log_prob_i - old_log_prob_i
                log_ratio = torch.clamp(raw_log_ratio, min=-2.0, max=2.0)
                ratio = torch.exp(log_ratio)
                try:
                    raw_list.append(raw_log_ratio.detach())
                    ratio_list.append(ratio.detach())
                except NameError:
                    raw_list = [raw_log_ratio.detach()]
                    ratio_list = [ratio.detach()]
                if norm_adv_mb.dim() == 1:
                    adv_i = norm_adv_mb
                elif norm_adv_mb.size(1) == 1:
                    adv_i = norm_adv_mb.squeeze(-1)
                else:
                    adv_i = norm_adv_mb[:, i]

                surr1 = ratio * adv_i
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_i
                loss_i = -torch.min(surr1, surr2).mean()

                actor_loss_mb = actor_loss_mb + loss_i
                approx_kl = (0.5 * (raw_log_ratio.detach() ** 2).mean()).item()
                approx_kl_sum_mb += approx_kl
                approx_kl_count_mb += 1

            if actor_non_finite:
                nan_skips += 1
                skip_reasons["actor_log_prob"] += 1
                continue

            actor_loss_mb /= cfg.n_agents
            # diffusion reconstruction regularization
            diffusion_mse_mb = 0.0
            if (
                getattr(cfg, 'policy_type', '') == 'diffusion'
                and getattr(cfg, 'diffusion_recon_weight', 0.0) > 0.0
            ):
                if getattr(cfg, 'enable_joint_policy', False) and hasattr(actor, 'joint_policy'):
                    _, joint_flat_mb = actor.team_encode(states[mb_idx])
                    a0_flat_mb = actions_mb.reshape(actions_mb.shape[0], -1)
                    diff_losses = actor.joint_policy.loss(a0_flat_mb, joint_flat_mb)
                    diffusion_mse_mb = float(diff_losses.mean().item())
                    actor_loss_mb = actor_loss_mb + cfg.diffusion_recon_weight * diff_losses.mean()
                else:
                    per_agent_losses = []
                    for i in range(cfg.n_agents):
                        per_agent_losses.append(actor.policy.loss(actions_mb[:, i], h_all_mb[:, i]))
                    per_agent_losses = torch.stack(per_agent_losses, dim=1)
                    diffusion_mse_mb = float(per_agent_losses.mean().item())
                    actor_loss_mb = actor_loss_mb + cfg.diffusion_recon_weight * per_agent_losses.mean()
            approx_kl_mb = float(approx_kl_sum_mb / approx_kl_count_mb) if approx_kl_count_mb else 0.0
            # compute minibatch ratio diagnostics
            if 'raw_list' in locals() and len(raw_list) > 0:
                raw_all = torch.cat([r.view(-1) for r in raw_list])
                raw_mean = float(raw_all.mean().cpu().item())
                raw_std = float(raw_all.std(unbiased=False).cpu().item())
                log_clamped = torch.clamp(raw_all, min=-2.0, max=2.0)
                ratio_all = torch.exp(log_clamped)
                ratio_mean = float(ratio_all.mean().cpu().item())
                ratio_std = float(ratio_all.std(unbiased=False).cpu().item())
                clip_frac = float(((ratio_all > 1.0 + cfg.clip_param) | (ratio_all < 1.0 - cfg.clip_param)).float().mean().cpu().item())
            else:
                raw_mean = raw_std = ratio_mean = ratio_std = clip_frac = 0.0

            total_raw_log_ratio_mean += raw_mean
            total_raw_log_ratio_std += raw_std
            total_ratio_mean += ratio_mean
            total_ratio_std += ratio_std
            total_clip_frac += clip_frac
            if 'raw_list' in locals():
                del raw_list
            if 'ratio_list' in locals():
                del ratio_list
            if not torch.isfinite(actor_loss_mb):
                nan_skips += 1
                skip_reasons["actor_loss"] += 1
                continue

            opt_actor.zero_grad()
            actor_loss_mb.backward()
            skip_reasons["actor_grad_sanitized"] += sanitize_gradients(actor)
            actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5))
            if not torch.isfinite(torch.tensor(actor_grad_norm, device=states.device)):
                opt_actor.zero_grad()
                nan_skips += 1
                skip_reasons["actor_grad"] += 1
                continue
            opt_actor.step()

            total_actor_loss += float(actor_loss_mb.item())
            total_critic_loss += float(critic_loss_mb.item())
            total_diffusion_mse += float(diffusion_mse_mb)
            total_approx_kl += approx_kl_mb
            total_actor_grad_norm += float(actor_grad_norm)
            total_critic_grad_norm += float(critic_grad_norm)
            valid_updates += 1

            if approx_kl_mb > getattr(cfg, 'target_kl', float('inf')):
                print(f"[PPO] Early stop at epoch {epoch + 1}/{cfg.ppo_epochs} due to approx_kl={approx_kl_mb:.4f} > target_kl={cfg.target_kl:.4f}")
                stop_epoch = True
                break

        if stop_epoch:
            break

    if valid_updates == 0:
        print(
            f"[PPO WARNING] No valid actor updates in episode {episode}; "
            f"nan_skips={nan_skips}, epochs_used={epochs_used}, T={T}, "
            f"reasons={skip_reasons}"
        )
        diag = {
            'adv_mean': float(adv_mean.item()),
            'adv_std': float(adv_std.item()),
            'clip_param': float(cfg.clip_param),
            'ppo_epochs': int(cfg.ppo_epochs),
            'ppo_epochs_used': int(epochs_used),
            'valid_updates': 0,
            'approx_kl': 0.0,
            'actor_grad_norm': 0.0,
            'critic_grad_norm': 0.0,
            'nan_skips': int(nan_skips),
            'update_failed': 1,
            'skip_critic_loss': int(skip_reasons["critic_loss"]),
            'skip_critic_grad': int(skip_reasons["critic_grad"]),
            'skip_actor_log_prob': int(skip_reasons["actor_log_prob"]),
            'skip_actor_loss': int(skip_reasons["actor_loss"]),
            'skip_actor_grad': int(skip_reasons["actor_grad"]),
            'actor_grad_sanitized': int(skip_reasons["actor_grad_sanitized"]),
        }
        return 0.0, 0.0, diag

    diag = {
        'adv_mean': float(adv_mean.item()),
        'adv_std': float(adv_std.item()),
        'clip_param': float(cfg.clip_param),
        'ppo_epochs': int(cfg.ppo_epochs),
        'ppo_epochs_used': int(epochs_used),
        'valid_updates': int(valid_updates),
        'approx_kl': float(total_approx_kl / valid_updates),
        'actor_grad_norm': float(total_actor_grad_norm / valid_updates),
        'critic_grad_norm': float(total_critic_grad_norm / valid_updates),
        'nan_skips': int(nan_skips),
        'update_failed': 0,
        'skip_critic_loss': int(skip_reasons["critic_loss"]),
        'skip_critic_grad': int(skip_reasons["critic_grad"]),
        'skip_actor_log_prob': int(skip_reasons["actor_log_prob"]),
        'skip_actor_loss': int(skip_reasons["actor_loss"]),
        'skip_actor_grad': int(skip_reasons["actor_grad"]),
        'actor_grad_sanitized': int(skip_reasons["actor_grad_sanitized"]),
        'diffusion_mse': float(total_diffusion_mse / valid_updates) if valid_updates else 0.0,
        'clip_frac': float(total_clip_frac / valid_updates) if valid_updates else 0.0,
        'ratio_mean': float(total_ratio_mean / valid_updates) if valid_updates else 0.0,
        'ratio_std': float(total_ratio_std / valid_updates) if valid_updates else 0.0,
        'raw_log_ratio_mean': float(total_raw_log_ratio_mean / valid_updates) if valid_updates else 0.0,
        'raw_log_ratio_std': float(total_raw_log_ratio_std / valid_updates) if valid_updates else 0.0,
    }

    return total_actor_loss / valid_updates, total_critic_loss / valid_updates, diag


def train(args):
    cfg = Config()
    set_seed(args.seed)

    if args.n_agents <= 0:
        raise ValueError(f"n_agents must be positive, got {args.n_agents}")
    cfg.n_agents = args.n_agents
    if args.debug_update is not None:
        cfg.debug_update = bool(args.debug_update)

    if args.actor_lr is not None:
        cfg.actor_lr = args.actor_lr
    if args.critic_lr is not None:
        cfg.critic_lr = args.critic_lr
    if args.diffusion_recon_weight is not None:
        cfg.diffusion_recon_weight = args.diffusion_recon_weight
    if args.clip_param is not None:
        cfg.clip_param = args.clip_param
    if args.value_clip_param is not None:
        cfg.value_clip_param = args.value_clip_param
    if args.reward_scale is not None:
        cfg.reward_scale = args.reward_scale
    if args.actor_lr_decay is not None:
        cfg.actor_lr_decay_on = bool(args.actor_lr_decay)
    if args.critic_lr_decay is not None:
        cfg.critic_lr_decay_on = bool(args.critic_lr_decay)
    if args.lr_decay_episodes is not None:
        cfg.lr_decay_episodes = args.lr_decay_episodes
    if args.lr_decay_short_frac is not None:
        cfg.lr_decay_short_frac = args.lr_decay_short_frac
    if args.lr_decay_min is not None:
        cfg.lr_decay_min = args.lr_decay_min
    if args.ppo_epochs is not None:
        cfg.ppo_epochs = args.ppo_epochs
    if args.history_len is not None:
        cfg.history_len = args.history_len
    if args.rollout_horizon is not None:
        cfg.rollout_horizon = args.rollout_horizon
    if args.diffusion_steps is not None:
        cfg.diffusion_steps = args.diffusion_steps
    if args.diffusion_action_eps is not None:
        cfg.diffusion_action_eps = float(args.diffusion_action_eps)
    if args.diffusion_min_std is not None:
        cfg.diffusion_min_std = float(args.diffusion_min_std)
    if args.pbrs is not None:
        cfg.pbrs_on = bool(args.pbrs)
    if args.pbrs_scale is not None:
        cfg.pbrs_potential_scale = args.pbrs_scale
    if args.pbrs_anneal is not None:
        cfg.pbrs_anneal_on = bool(args.pbrs_anneal)
    if args.pbrs_anneal_start is not None:
        cfg.pbrs_anneal_start_frac = args.pbrs_anneal_start
    if args.pbrs_anneal_end is not None:
        cfg.pbrs_anneal_end_frac = args.pbrs_anneal_end
    if args.coverage_bonus is not None:
        cfg.coverage_bonus = args.coverage_bonus
    if args.fov_mask is not None:
        cfg.fov_mask_on = bool(args.fov_mask)
    if args.fov_radius is not None:
        cfg.fov_radius = float(args.fov_radius)
    if args.fov_visibility is not None:
        cfg.fov_visibility_on = bool(args.fov_visibility)
    if args.terminate_on_success is not None:
        cfg.terminate_on_success = bool(args.terminate_on_success)

    if not (0.0 <= cfg.pbrs_anneal_start_frac <= 1.0):
        raise ValueError(f"pbrs_anneal_start_frac must be in [0, 1], got {cfg.pbrs_anneal_start_frac}")
    if not (0.0 <= cfg.pbrs_anneal_end_frac <= 1.0):
        raise ValueError(f"pbrs_anneal_end_frac must be in [0, 1], got {cfg.pbrs_anneal_end_frac}")
    if cfg.pbrs_anneal_start_frac > cfg.pbrs_anneal_end_frac:
        raise ValueError(
            f"pbrs_anneal_start_frac must be <= pbrs_anneal_end_frac, got {cfg.pbrs_anneal_start_frac} > {cfg.pbrs_anneal_end_frac}"
        )
    if cfg.fov_radius <= 0.0:
        raise ValueError(f"fov_radius must be positive, got {cfg.fov_radius}")
    if cfg.rollout_horizon <= 0:
        raise ValueError(f"rollout_horizon must be positive, got {cfg.rollout_horizon}")
    if cfg.lr_decay_episodes <= 0:
        raise ValueError(f"lr_decay_episodes must be positive, got {cfg.lr_decay_episodes}")
    if not (0.0 < cfg.lr_decay_short_frac <= 1.0):
        raise ValueError(f"lr_decay_short_frac must be in (0, 1], got {cfg.lr_decay_short_frac}")
    if cfg.lr_decay_min < 0.0:
        raise ValueError(f"lr_decay_min must be non-negative, got {cfg.lr_decay_min}")

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
    env = SimpleMultiAgentEnv(
        cfg.n_agents,
        max_cycles=args.max_cycles,
        seed=args.seed,
        gamma=cfg.gamma,
        pbrs_on=cfg.pbrs_on,
        potential_scale=cfg.pbrs_potential_scale,
        coverage_bonus=cfg.coverage_bonus,
        fov_mask_on=cfg.fov_mask_on,
        fov_radius=cfg.fov_radius,
        fov_visibility_on=cfg.fov_visibility_on,
        terminate_on_success=cfg.terminate_on_success,
    )

    cfg.n_agents = env.n_agents
    cfg.obs_dim = env.obs_dim
    cfg.action_dim = env.action_dim
    cfg.action_low = env._act_low
    cfg.action_high = env._act_high
    if cfg.reward_scale is None:
        cfg.reward_scale = 1.0 / max(float(cfg.n_agents), 1.0)

    # enable joint policy if requested
    cfg.enable_joint_policy = bool(args.enable_joint_policy)

    logger = TrainingLogger(
        log_dir=args.log_dir,
        plot_meta={
            "n_agents": cfg.n_agents,
            "max_steps": args.max_steps,
            "episodes": args.episodes,           
            "actor_lr": cfg.actor_lr,
            "critic_lr": cfg.critic_lr,
            "method": args.method,
            "reward_scale": cfg.reward_scale,
            "actor_lr_decay_on": cfg.actor_lr_decay_on,
            "critic_lr_decay_on": cfg.critic_lr_decay_on,
            "pbrs_on": cfg.pbrs_on,
            "pbrs_scale": cfg.pbrs_potential_scale,
            "coverage_bonus": cfg.coverage_bonus,
            "fov_mask_on": cfg.fov_mask_on,
            "fov_radius": cfg.fov_radius,
            "fov_visibility_on": cfg.fov_visibility_on,
        },
    )
    actor = Actor(cfg).to(device)
    critic = Critic(cfg).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, weight_decay=1e-4)
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
        f"actor_lr={cfg.actor_lr} critic_lr={cfg.critic_lr} history_len={cfg.history_len} rollout_horizon={cfg.rollout_horizon} diffusion_steps={cfg.diffusion_steps} "
        f"clip_param={cfg.clip_param} value_clip_param={cfg.value_clip_param} reward_scale={cfg.reward_scale} ppo_epochs={cfg.ppo_epochs} "
        f"actor_lr_decay_on={cfg.actor_lr_decay_on} critic_lr_decay_on={cfg.critic_lr_decay_on} "
        f"fov_mask_on={cfg.fov_mask_on} fov_radius={cfg.fov_radius} fov_visibility_on={cfg.fov_visibility_on} "
        f"coverage_bonus={cfg.coverage_bonus} "
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

        rollout_steps = 0
        updates_done = 0

        for episode in episode_iter:
            obs = env.reset()  # [N, obs_dim]
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            # init obs_seq with the first obs
            obs_seq = obs_t.unsqueeze(1).repeat(1, cfg.history_len, 1).unsqueeze(0)  # [1, N, H, D]

            total_reward = 0.0
            episode_env_reward = 0.0
            episode_shaping_reward = 0.0
            episode_effective_shaping_reward = 0.0
            episode_team_reward = 0.0
            episode_success = 0.0
            episode_steps = 0
            episode_landmark_coverage = 0.0
            episode_min_landmark_distance = float('inf')
            done = False

            for _ in range(args.max_steps):
                episode_steps += 1
                with torch.no_grad():
                    step_log_probs_batch = None
                    chains_batch = None
                    if hasattr(actor.policy, "sample_with_chain"):
                        actions, chains_batch, step_log_probs_batch = actor.sample_with_chain(obs_seq, joint=cfg.enable_joint_policy)
                        # Use sum over diffusion steps for the joint chain log-prob
                        # step_log_probs_batch is [B, K] in joint mode, [B, N, K] in independent mode.
                        if getattr(cfg, "enable_joint_policy", False):
                            old_log_probs = step_log_probs_batch.sum(dim=-1, keepdim=True)
                        else:
                            old_log_probs = step_log_probs_batch.sum(dim=-1, keepdim=True)
                        ks = None
                        noises = None
                    else:
                        actions, _ = actor(obs_seq)
                        h_all = actor.encode(obs_seq)

                        old_log_probs = []
                        ks = []
                        noises = []
                        for i in range(cfg.n_agents):
                            k_i = torch.randint(1, cfg.diffusion_steps, (actions.shape[0],), device=device)
                            noise_i = torch.randn_like(actions[:, i])
                            log_prob_i = actor.policy.get_log_prob(actions[:, i], h_all[:, i], k_i, noise=noise_i)
                            old_log_probs.append(log_prob_i.unsqueeze(1))
                            ks.append(k_i.view(-1, 1, 1))
                            noises.append(noise_i.unsqueeze(1))

                        old_log_probs = torch.cat(old_log_probs, dim=1)
                        ks = torch.cat(ks, dim=1)
                        noises = torch.cat(noises, dim=1)

                    value = critic(obs_seq).item()

                actions_env = actions[0].detach().cpu().numpy()
                # 使用环境的动作上下界进行截断，保持与 env.step 一致
                try:
                    actions_env = np.clip(actions_env, env._act_low, env._act_high)
                except Exception:
                    # Fallback: 不作额外裁剪，让 env.step 自行处理
                    pass
                
                # 再转成 list 传给环境
                next_obs, reward, done, _ = env.step(actions_env.tolist())

                env_team_reward = getattr(env, 'last_env_reward', float(reward))
                shaping_reward = getattr(env, 'last_shaping_reward', 0.0)
                episode_success = max(episode_success, float(getattr(env, 'last_success', False)))
                episode_landmark_coverage = max(
                    episode_landmark_coverage,
                    float(getattr(env, 'last_landmark_coverage', 0.0)),
                )
                episode_min_landmark_distance = min(
                    episode_min_landmark_distance,
                    float(getattr(env, 'last_min_landmark_distance', 0.0)),
                )
                if cfg.pbrs_on and cfg.pbrs_anneal_on:
                    progress = episode / max(args.episodes - 1, 1)
                    if progress < cfg.pbrs_anneal_start_frac:
                        shaping_weight = 1.0
                    elif progress >= cfg.pbrs_anneal_end_frac:
                        shaping_weight = 0.0
                    else:
                        span = max(cfg.pbrs_anneal_end_frac - cfg.pbrs_anneal_start_frac, 1e-8)
                        shaping_weight = 1.0 - (progress - cfg.pbrs_anneal_start_frac) / span
                else:
                    shaping_weight = 1.0

                effective_shaping_reward = float(shaping_reward) * shaping_weight
                team_reward = float(env_team_reward) + effective_shaping_reward
                scaled_reward = team_reward * float(cfg.reward_scale)
                episode_env_reward += float(env_team_reward)
                episode_shaping_reward += float(shaping_reward)
                episode_effective_shaping_reward += float(effective_shaping_reward)
                episode_team_reward += float(team_reward)
                rollout_steps += 1

                buffer.add(
                    obs_seq.clone(),
                    actions,
                    scaled_reward,
                    done,
                    value,
                    log_prob=old_log_probs,
                    step_log_prob=step_log_probs_batch,
                    chain=chains_batch,
                    k=ks,
                    noise=noises,
                )
                total_reward += team_reward

                next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                next_obs_t = next_obs_t.unsqueeze(0).unsqueeze(2)  # [1, N, 1, D]
                obs_seq = torch.cat([obs_seq[:, :, 1:], next_obs_t], dim=2)

                if done:
                    break

            if (not done) and episode_steps >= args.max_steps and buffer.dones:
                # The outer training loop resets the env here, so this is a rollout boundary.
                buffer.dones[-1] = 1.0

            is_last_episode = episode == args.episodes - 1
            should_update = rollout_steps >= cfg.rollout_horizon or is_last_episode
            actor_loss = 0.0
            critic_loss = 0.0
            diag = {
                'did_update': 0,
                'rollout_steps': int(rollout_steps),
                'reward_scale': float(cfg.reward_scale),
            }

            if should_update and rollout_steps > 0:
                with torch.no_grad():
                    last_value = 0.0 if (buffer.dones and buffer.dones[-1] >= 1.0) else critic(obs_seq).item()

                buffer.compute_returns_advantages(last_value)
                actor_loss, critic_loss, diag = update(
                    actor,
                    critic,
                    buffer,
                    cfg,
                    opt_actor,
                    opt_critic,
                    episode=episode,
                )
                updates_done += 1
                if isinstance(diag, dict):
                    diag['did_update'] = 1
                    diag['rollout_steps'] = int(rollout_steps)
                    diag['reward_scale'] = float(cfg.reward_scale)
                    diag['updates_done'] = int(updates_done)
                buffer.clear()
                rollout_steps = 0

            if isinstance(diag, dict):
                diag['env_reward'] = float(episode_env_reward)
                diag['shaping_reward'] = float(episode_shaping_reward)
                diag['effective_shaping_reward'] = float(episode_effective_shaping_reward)
                diag['shaping_weight'] = float(shaping_weight)
                diag['team_reward'] = float(episode_team_reward)
                diag['success'] = float(episode_success)
                diag['landmark_coverage'] = float(episode_landmark_coverage)
                diag['min_landmark_distance'] = float(
                    episode_min_landmark_distance if np.isfinite(episode_min_landmark_distance) else 0.0
                )
                diag['episode_steps'] = int(episode_steps)
                diag['shaping_ratio'] = float(
                    episode_shaping_reward / (abs(episode_env_reward) + 1e-8)
                )

            logger.record(episode, total_reward, actor_loss, critic_loss, diag=diag)

            decay_episodes = int(cfg.lr_decay_episodes)
            if args.episodes <= 1000:
                decay_episodes = int(args.episodes * cfg.lr_decay_short_frac)
            decay_episodes = max(1, min(decay_episodes, args.episodes))
            start_decay_episode = args.episodes - decay_episodes

            if episode < start_decay_episode:
                lr_fraction = 1.0
            else:
                steps_into_decay = episode - start_decay_episode
                lr_fraction = max(0.0, 1.0 - (steps_into_decay / decay_episodes))

            if cfg.actor_lr_decay_on:
                lr_actor_now = max(cfg.actor_lr * lr_fraction, cfg.lr_decay_min)
            else:
                lr_actor_now = cfg.actor_lr
            if cfg.critic_lr_decay_on:
                lr_critic_now = max(cfg.critic_lr * lr_fraction, cfg.lr_decay_min)
            else:
                lr_critic_now = cfg.critic_lr

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
                "clip_param": cfg.clip_param,
                "value_clip_param": cfg.value_clip_param,
                "reward_scale": cfg.reward_scale,
                "actor_lr_decay_on": cfg.actor_lr_decay_on,
                "critic_lr_decay_on": cfg.critic_lr_decay_on,
                "lr_decay_episodes": cfg.lr_decay_episodes,
                "lr_decay_short_frac": cfg.lr_decay_short_frac,
                "lr_decay_min": cfg.lr_decay_min,
                "ppo_epochs": cfg.ppo_epochs,
                "history_len": cfg.history_len,
                "rollout_horizon": cfg.rollout_horizon,
                "diffusion_steps": cfg.diffusion_steps,
                "diffusion_action_eps": cfg.diffusion_action_eps,
                "diffusion_min_std": cfg.diffusion_min_std,
                "pbrs_on": cfg.pbrs_on,
                "pbrs_potential_scale": cfg.pbrs_potential_scale,
                "pbrs_anneal_on": cfg.pbrs_anneal_on,
                "pbrs_anneal_start_frac": cfg.pbrs_anneal_start_frac,
                "pbrs_anneal_end_frac": cfg.pbrs_anneal_end_frac,
                "fov_mask_on": cfg.fov_mask_on,
                "fov_radius": cfg.fov_radius,
                "fov_visibility_on": cfg.fov_visibility_on,
                "terminate_on_success": cfg.terminate_on_success,
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
        final_stats = logger.get_stats()
        print("[Run Summary]")
        print(
            f"  run_dir={logger.run_dir} | method={args.method} | seed={args.seed} | "
            f"n_agents={cfg.n_agents} | obs_dim={cfg.obs_dim} | action_dim={cfg.action_dim} | "
            f"max_steps={args.max_steps} | max_cycles={args.max_cycles} | "
            f"terminate_on_success={cfg.terminate_on_success}"
        )
        print(
            f"  final_reward={final_stats.get('final_reward', 0.0):.3f} | "
            f"avg_reward_last_10={final_stats.get('avg_reward', 0.0):.3f} | "
            f"avg_success_rate_last_10={final_stats.get('avg_success_rate', 0.0):.3f} | "
            f"avg_episode_steps_last_10={final_stats.get('avg_episode_steps', 0.0):.3f} | "
            f"avg_actor_loss_last_10={final_stats.get('avg_actor_loss', 0.0):.4f} | "
            f"avg_critic_loss_last_10={final_stats.get('avg_critic_loss', 0.0):.4f}"
        )
        print(f"[Training Complete] Results saved to {logger.run_dir}")
        print(f"[Training Complete] Quick access: {args.log_dir}/latest/")
    finally:
        env.close()


if __name__ == "__main__":
    train(parse_args(Config()))
