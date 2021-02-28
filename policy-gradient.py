import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

import gym

class LinearAct(nn.Linear):
    def __init__(self, *args, activation=F.relu, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation
        
    def foward(self, x):
        x = super().forward(x)
        return self.activation(x)

class MLP(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        self.layers = nn.Sequential(*(LinearAct(s1, s2) for s1, s2 in zip(sizes, sizes[1:])))

    def forward(self, x):
        return self.layers(x)

class PolicyGradient:
    def __init__(self, obs_dim: int, nb_actions: int, hidden_sizes: list[int] = [32]
                 lr: float = 1e-2, device = torch.device('cpu')):
        self.net = MLP([obs_dim, *hidden_sizes, nb_actions]).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def get_policy(obs):
        pass

    def get_action(obs):
        pass

    def get_loss(obs, action, weight):
        pass

    def learn():
        pass

def main(env_id: str = 'CartPole-v0', hidden_sizes=[32], lr=1e-2,
        epochs=50, batch_size=5000, render=True, device=torch.device('cpu')):

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    agent = PolicyGradient(obs_dim, nb_actions, hidden_sizes, lr=lr, device=device)

    for eid in range(epochs):
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu else torch.device('cuda')

    train(env_id=args.env_name, render=args.render)
