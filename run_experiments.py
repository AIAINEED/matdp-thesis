import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
from config import Config


def parse_args():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Run multi-seed experiments and aggregate metrics")
    
    # 支持单复数别名，防止终端传参报错
    parser.add_argument("--methods", "--method", type=str, nargs="+", default=cfg.exp_methods, dest="methods")
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=cfg.exp_seeds, dest="seeds")
    
    parser.add_argument("--episodes", type=int, default=cfg.episodes)
    parser.add_argument("--max-steps", type=int, default=cfg.max_steps)
    parser.add_argument("--max-cycles", type=int, default=cfg.max_cycles)
    parser.add_argument("--plot-every", type=int, default=cfg.plot_every)
    parser.add_argument("--log-dir", type=str, default=cfg.log_dir)
    parser.add_argument("--save-raw-logs", type=int, choices=[0, 1], default=int(cfg.exp_save_raw_logs))
    
    # 独立的学习率参数
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    return parser.parse_args()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def run_one(method, seed, args):
    cmd = [
        sys.executable,
        "main.py",
        "--method",
        method,
        "--seed",
        str(seed),
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--max-cycles",
        str(args.max_cycles),
        "--plot-every",
        str(args.plot_every),
        "--log-dir",
        args.log_dir,
    ]

    # 透传独立参数到 main.py
    if args.actor_lr is not None:
        cmd.extend(["--actor-lr", str(args.actor_lr)])
    if args.critic_lr is not None:
        cmd.extend(["--critic-lr", str(args.critic_lr)])
    if args.horizon is not None:
        cmd.extend(["--horizon", str(args.horizon)])
    if args.diffusion_steps is not None:
        cmd.extend(["--diffusion-steps", str(args.diffusion_steps)])

    print(f"[Experiment] Running method={method} seed={seed}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    save_raw_logs = bool(args.save_raw_logs)

    experiments_dir = Path(args.log_dir) / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    if result.returncode != 0:
        if save_raw_logs:
            fail_log = experiments_dir / f"failed_{method}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            fail_log.write_text(result.stdout + "\n\n=== STDERR ===\n" + result.stderr)
            print(f"[Experiment] Saved failure log: {fail_log}")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"method={method} seed={seed} failed with code {result.returncode}")

    output = result.stdout
    match = re.search(r"\[Training Complete\] Results saved to (.+)", output)
    if not match:
        raise RuntimeError(f"Could not find run directory for seed={seed}")

    run_dir = Path(match.group(1).strip())
    if save_raw_logs:
        (run_dir / "run_stdout.log").write_text(result.stdout)
        (run_dir / "run_stderr.log").write_text(result.stderr)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics.json not found for seed={seed}: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    summary = metrics.get("summary", {})
    print(
        f"[Experiment] method={method} seed={seed} final_reward={summary.get('final_reward', 0):.3f} "
        f"avg_last10={summary.get('avg_reward_last_10', 0):.3f} run_dir={run_dir}"
    )

    return {
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir),
        "summary": summary,
    }


def aggregate(results):
    if not results:
        return {}

    final_rewards = [r["summary"].get("final_reward", 0.0) for r in results]
    avg_last10 = [r["summary"].get("avg_reward_last_10", 0.0) for r in results]

    n = len(results)
    mean_final = sum(final_rewards) / n
    mean_last10 = sum(avg_last10) / n
    var_last10 = sum((x - mean_last10) ** 2 for x in avg_last10) / n

    return {
        "num_runs": n,
        "mean_final_reward": mean_final,
        "mean_avg_reward_last_10": mean_last10,
        "var_avg_reward_last_10": var_last10,
        "std_avg_reward_last_10": var_last10 ** 0.5,
    }


def main():
    args = parse_args()
    results = []
    tasks = [(method, seed) for method in args.methods for seed in args.seeds]

    if tqdm is not None:
        task_iter = tqdm(tasks, total=len(tasks), desc="Experiments", dynamic_ncols=True)
    else:
        task_iter = tasks

    for method, seed in task_iter:
        result = run_one(method, seed, args)
        results.append(result)
        if tqdm is not None:
            task_iter.set_postfix(method=method, seed=seed, avg_last10=f"{result['summary'].get('avg_reward_last_10', 0):.2f}")

    aggregate_stats = aggregate(results)

    per_method = {}
    for method in args.methods:
        method_runs = [r for r in results if r["method"] == method]
        per_method[method] = aggregate(method_runs)

    out_dir = Path(args.log_dir) / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    payload = {
        "config": {
            "methods": args.methods,
            "seeds": args.seeds,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "max_cycles": args.max_cycles,
            "plot_every": args.plot_every,
            "log_dir": args.log_dir,
            "save_raw_logs": bool(args.save_raw_logs),
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "horizon": args.horizon,
            "diffusion_steps": args.diffusion_steps,
            "git_commit": get_git_commit(),
        },
        "runs": results,
        "aggregate": aggregate_stats,
        "per_method": per_method,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[Experiment] Aggregated summary saved to {out_file}")
    print(f"[Experiment] Aggregate stats (all): {json.dumps(aggregate_stats, ensure_ascii=True)}")
    print(f"[Experiment] Aggregate stats (per_method): {json.dumps(per_method, ensure_ascii=True)}")


if __name__ == "__main__":
    main()