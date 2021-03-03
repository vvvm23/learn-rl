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
        'net_type':             'conv',
        'save_frequency':       100,
    }),

    'cartpole': SimpleNamespace(**{
        'env_name':             "CartPole-v1",
        'stop_reward':          198.0,
        'max_steps':            10_000,
        'memory_capacity':      1_000,
        'memory_initial':       100,
        'target_sync':          10,
        'train_frequency':      1,
        'epsilon_start':        1.0,
        'epsilon_end':          0.01,
        'epsilon_frames':       1_000,
        'learning_rate':        1e-2,
        'gamma':                0.99,
        'batch_size':           256,
        'net_type':             'linear',
        'save_frequency':       100,
    })
}
