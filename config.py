class Config:
    device = "cuda"

    n_agents = 3
    obs_dim = 18
    action_dim = 5

    hidden_dim = 128
    n_heads = 4
    n_layers = 2

    horizon = 20
    diffusion_steps = 10

    gamma = 0.99
    lam = 0.95

    actor_lr = 1e-6      
    critic_lr = 1e-3     
    clip_param = 0.2
    ppo_epochs = 2
    target_kl = 0.25   # 0.05 0.25
    batch_size = 128   # 128  256

    # ===== Runtime / Experiment defaults =====
    seed = 42
    episodes = 5000
    max_steps = 50
    max_cycles = 50
    plot_every = 500
    log_dir = "logs"
    save_checkpoints = True
    checkpoint_every = 100
    resume_path = ""

    # Default seeds for multi-run experiments
    exp_seeds = [0, 1, 2]
    exp_save_raw_logs = True
    method = "full"  # full | no_transformer | no_diffusion
    exp_methods = ["full", "no_transformer", "no_diffusion"]