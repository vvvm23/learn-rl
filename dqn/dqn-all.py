import torch
import gym
import numpy as np
from tqdm import tqdm

import argparse
import datetime
from pathlib import Path
from collections import deque
from time import sleep

from agent import DQNAgent 
from memory.memory import ExperienceBuffer
from models.rainbow import DQN
from wrappers import make_env
from helper import EpisilonAnnealer, VisdomLinePlotter

from hps import HPS_RAINBOW as HPS

@torch.no_grad()
def evaluate(agent, env, render=False):
    while True:
        obs = env.reset()
        state = torch.zeros(params.frame_stack, *obs_shape)
        done = False
        total_return = 0.
        while not done:
            if render:
                env.render()
                sleep(1 / 60.)
            state = torch.cat([state[1:], torch.from_numpy(obs).unsqueeze(0)], dim=0)
            action = agent.act_greedy(state)
            obs, reward, done, _ = env.step(action)
            total_return += reward
        print(f"episode score: {total_return}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pong') 
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--evaluate', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, nargs=2)
    args = parser.parse_args()
    params = HPS[args.task]

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    
    env = make_env(params.env_name)
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n

    if params.net_type == 'conv':
        net = DQN((params.frame_stack, *obs_shape), nb_actions)
    elif params.net_type == 'linear':
        raise NotImplementedError
    agent = DQNAgent(
        net=net, 
        nb_actions=nb_actions, 
        gamma=params.gamma,
        unroll_steps=params.unroll_steps,
        device=device
    )

    if args.evaluate:
        agent.net.load_state_dict(torch.load(args.evaluate))
        evaluate(agent, env, render=args.render)
        exit()

    if args.resume:
        agent.load(args.resume[0])

    memory = ExperienceBuffer(params.memory_capacity, obs_shape, params.frame_stack,
                              prioritized=True,
                              unroll_steps=params.unroll_steps,
                              beta_start=params.beta_start, beta_end=params.beta_end, beta_steps=params.beta_frames)

    opt = torch.optim.Adam(net.parameters(), lr=params.learning_rate)

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    path_str = lambda p: p.absolute().as_posix() # converts a pathlib Path to string

    # Create the run directories
    runs_dir = Path(f"runs")
    root_dir = runs_dir / f'{args.task}-{save_id}'
    chk_dir = root_dir / 'checkpoints'

    runs_dir.mkdir(exist_ok=True) 
    root_dir.mkdir(exist_ok=True)
    chk_dir.mkdir(exist_ok=True)

    episode_reward = 0.0
    episode_count = 0

    obs = env.reset()
    pb = tqdm(range(-params.memory_initial, params.max_steps))
    plotter = VisdomLinePlotter()
    reward_history = deque(maxlen=100)
    for i in pb:
        if args.render:
            env.render()

        idx = memory.store_obs(obs)
        state = memory.get_stacked_obs(idx)

        action = agent.act_greedy(state)
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        memory.store_effect(idx, action, np.sign(reward), done)

        if done:
            next_obs = env.reset()
            episode_count += 1
            reward_history.append(episode_reward)

            pb.set_description(f"episode: {episode_count}, reward: {episode_reward}")
            plotter.plot('episode reward', 'episode return', "Episode Return", episode_count, episode_reward)
            plotter.plot('episode reward', 'average return', "Episode Return", episode_count, sum(reward_history) / len(reward_history))
            episode_reward = 0
            if episode_count > 0 and episode_count % params.save_frequency == 0:
                agent.save(chk_dir/ f"checkpoint-episode-{episode_count}.pt")

        obs = next_obs

        if i < 0: continue

        if i % params.target_sync == 0:
            agent.sync_target()

        if i % params.train_frequency == 0:
            opt.zero_grad()
            *batch, idx, weights = memory.sample(params.batch_size)
            loss = agent.calculate_loss(*batch, double=True)
            loss = loss * weights.to(device)
            loss.mean().backward()
            opt.step()
            memory.update_priorities(idx, loss)

        memory.update_beta(i)
        pb.update(1)
