class Config:
    device = "cuda"

    n_agents = 3
    obs_dim = 18
    action_dim = 5
    action_low = 0.0
    action_high = 1.0

    hidden_dim = 128
    n_heads = 4
    n_layers = 2

    # Rollout / return computation horizon. 
    rollout_horizon = 250
    # Transformer observation history length.
    history_len = 10
    diffusion_steps = 10
    diffusion_action_eps = 1e-4
    diffusion_min_std = 0.1
    # weight for diffusion reconstruction (predicted noise) regularization term
    diffusion_recon_weight = 1e-4

    gamma = 0.99
    lam = 0.95
    pbrs_on = False
    pbrs_potential_scale = 2.5
    pbrs_anneal_on = True
    pbrs_anneal_start_frac = 0.5
    pbrs_anneal_end_frac = 1.0
    coverage_bonus = 5.0
    fov_mask_on = True
    fov_radius = 0.5
    fov_visibility_on = True
    terminate_on_success = True

    actor_lr = 1e-4      
    critic_lr = 3e-4     
    clip_param = 0.2
    value_clip_param = 10.0
    reward_scale = None
    actor_lr_decay_on = False
    critic_lr_decay_on = False
    lr_decay_episodes = 500
    lr_decay_short_frac = 0.25
    lr_decay_min = 1e-6
    ppo_epochs = 2
    target_kl = 0.15
    batch_size = 256   

    # ===== Runtime / Experiment defaults =====
    seed = 42
    episodes = 5000
    max_steps = 50
    max_cycles = 50
    plot_every = 500
    log_dir = "logs"
    save_checkpoints = False
    checkpoint_every = 100
    resume_path = ""
    debug_update = False
    debug_policy_update = False

    # Default seeds for multi-run experiments
    exp_seeds = [0, 1, 2]
    exp_save_raw_logs = True
    method = "full"  # full | no_transformer | no_diffusion
    exp_methods = ["full", "no_transformer", "no_diffusion"]
