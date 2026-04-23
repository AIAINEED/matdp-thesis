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

    actor_lr = 1e-4      
    critic_lr = 1e-3     
    beta = 1.0
    batch_size = 64
    epochs = 10

    # ===== Runtime / Experiment defaults =====
    seed = 42
    episodes = 5
    max_steps = 10000
    max_cycles = 50
    plot_every = 50
    log_dir = "logs"
    save_checkpoints = True
    checkpoint_every = 100
    resume_path = ""

    # Default seeds for multi-run experiments
    exp_seeds = [0, 1, 2]
    exp_save_raw_logs = True
    method = "full"  # full | no_transformer | no_diffusion
    exp_methods = ["full", "no_transformer", "no_diffusion"]