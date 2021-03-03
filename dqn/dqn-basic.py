import torch
import gym
from tqdm import tqdm

import argparse
import datetime
from pathlib import Path

from agent import DQNAgent 
from memory.memory import ExperienceBuffer
from models.dqn import DQN, DQNLinear
from wrappers import make_env
from helper import EpisilonAnnealer, VisdomLinePlotter

from hps import HPS_BASIC as HPS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pong') 
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    params = HPS[args.task]

    device = torch.device('cpu') if args.cpu else torch.device('cuda')
    
    env = make_env(params.env_name)
    obs_shape = env.observation_space.shape
    nb_actions = env.action_space.n

    if params.net_type == 'conv':
        net = DQN(obs_shape, nb_actions)
    elif params.net_type == 'linear':
        net = DQNLinear(obs_shape, nb_actions)
    agent = DQNAgent(
        net=net, 
        nb_actions=nb_actions, 
        gamma=params.gamma,
        device=device
    )
    memory = ExperienceBuffer(params.memory_capacity, obs_shape[1:], obs_shape[0])

    opt = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
    eps_schedule = EpisilonAnnealer(params.epsilon_start, params.epsilon_end, params.epsilon_frames)

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    path_str = lambda p: p.absolute().as_posix() # converts a pathlib Path to string

    # Create the run directories
    runs_dir = Path(f"runs")
    root_dir = runs_dir / f'run-{save_id}'
    chk_dir = root_dir / 'checkpoints'

    runs_dir.mkdir(exist_ok=True) 
    root_dir.mkdir(exist_ok=True)
    chk_dir.mkdir(exist_ok=True)

    # TODO: Merge these two loops? maybe..
    # initial population of buffer
    obs = env.reset()
    for i in range(params.memory_initial):
        obs = obs.squeeze(0) # TODO: Check wrappers so we don't have to do this
        idx = memory.store_obs(obs)
        state = memory.get_stacked_obs(idx)
        action = agent.act_epsilon_greedy(state, 1.0)
        next_obs, reward, done, _ = env.step(action)
        memory.store_effect(idx, action, reward, done)

        if done:
            next_obs = env.reset()
        obs = next_obs

    episode_reward = 0.0
    episode_count = 0

    pb = tqdm(range(params.max_steps))
    plotter = VisdomLinePlotter()
    for i in pb:
        if args.render:
            env.render()

        obs = obs.squeeze(0)
        idx = memory.store_obs(obs)
        state = memory.get_stacked_obs(idx)

        action = agent.act_epsilon_greedy(state, eps_schedule.get(i))
        # TODO: Do we clip reward?
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        memory.store_effect(idx, action, reward, done)

        if done:
            next_obs = env.reset()
            episode_count += 1
            pb.set_description(f"episode: {episode_count}, reward: {episode_reward}, eps: {eps_schedule.get(i)*100:.2f}%")
            plotter.plot('episode reward', 'foo', "Episode Return", episode_count, episode_reward)
            episode_reward = 0
            if episode_count > 0 and episode_count % params.save_frequency == 0:
                agent.save(chk_dir/ f"checkpoint-episode-{episode_count}.pt")

        obs = next_obs

        if i % params.target_sync == 0:
            agent.sync_target()

        if i % params.train_frequency == 0:
            opt.zero_grad()
            batch = memory.sample(params.batch_size)
            loss = agent.calculate_loss(*batch)
            loss.backward()
            opt.step()

        pb.update(1)
