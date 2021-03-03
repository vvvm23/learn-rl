import torch
import gym
from tqdm import tqdm

import argparse

from agent import DQNAgent 
from memory.memory import ExperienceBuffer
from models.dqn import DQN
from wrappers import make_env
from helper import EpisilonAnnealer

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

    net = DQN(obs_shape, nb_actions)
    agent = DQNAgent(
        net=net, 
        nb_actions=nb_actions, 
        gamma=params.gamma,
        device=device
    )
    memory = ExperienceBuffer(params.memory_capacity, obs_shape[1:], obs_shape[0])

    opt = torch.optim.Adam(net.parameters(), lr=params.learning_rate)

    eps_schedule = EpisilonAnnealer(params.epsilon_start, params.epsilon_end, params.epsilon_frames)
