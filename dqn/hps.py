from types import SimpleNamespace

HPS_BASIC = {
    'pong': SimpleNamespace(**{
        'env_name':             "PongNoFrameskip-v4",
        'stop_reward':          18.0,
        'max_steps':            2_000_000,
        'memory_capacity':      100_000,
        'memory_initial':       10_000,
        'target_sync':          1000,
        'train_frequency':      1,
        'epsilon_start':        1.0,
        'epsilon_end':          0.01,
        'epsilon_frames':       100_000,
        'learning_rate':        1e-4,
        'gamma':                0.99,
        'batch_size':           32,
    })
}
