import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from utils.plotter import VisdomLinePlotter

import gym

class LinearAct(nn.Linear):
    def __init__(self, *args, activation=F.relu, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
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
    def __init__(self, obs_dim: int, nb_actions: int, hidden_sizes: list[int] = [32],
            lr: float = 1e-2, batch_size: int = 4096, render=False, device = torch.device('cpu')):
        self.net = MLP([obs_dim, *hidden_sizes, nb_actions]).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = device
        self.render = render

    def get_policy(self, obs):
        return Categorical(logits=self.net(obs))

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def get_loss(self, obs, action, weight):
        log_prob = self.get_policy(obs).log_prob(action)
        return -(log_prob * weight).mean()

    def reward_to_go(self, rews):
        n = len(rews)
        rtg = torch.zeros(n)
        for i in range(n-1, -1, -1):
            rtg[i] = rews[i] + (rtg[i+1] if i+1 < n else 0)
        return rtg

    def learn(self, env):
        batch_obs       = []
        batch_acts      = []
        batch_weights   = []
        batch_returns   = []
        batch_lengths   = []

        obs = env.reset()
        done = False
        episode_rewards = []

        rendered = False

        while True:
            if self.render and not rendered:
                env.render()
            batch_obs.append(obs.copy()) # do we need a copy here?

            act = self.get_action(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
            obs, rew, done, _ = env.step(act)

            batch_acts.append(act)
            episode_rewards.append(rew)

            if done:
                episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lengths.append(episode_length)

                # if not rendered:
                    # env.close()
                rendered = True

                # batch_weights += [episode_return] * episode_length # duplicate return length times
                batch_weights += list(self.reward_to_go(episode_rewards))
                obs, done, episode_rewards = env.reset(), False, []

                if len(batch_obs) > self.batch_size:
                    break

        self.opt.zero_grad()
        loss = self.get_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(self.device),
                             action=torch.as_tensor(batch_acts, dtype=torch.int32).to(self.device),
                             weight=torch.as_tensor(batch_weights, dtype=torch.float32).to(self.device)
                            )
        loss.backward()
        self.opt.step()

        return batch_returns, batch_lengths

def main(env_id: str = 'CartPole-v0', hidden_sizes=[32], lr=1e-2,
        epochs=100, batch_size=4096, render=True, 
        repeat_test: int = 3, device=torch.device('cpu')):

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    plotter = VisdomLinePlotter()
    agent = PolicyGradient(obs_dim, nb_actions, hidden_sizes, lr=lr, batch_size=batch_size, render=render, device=device)
    for eid in range(epochs):
        returns, _ = agent.learn(env)
        print(f"> epoch: {eid+1}/{epochs}, \t mean_return: {sum(returns) / len(returns)}, \t max_return: {max(returns)}")
        plotter.plot('return', 'mean return', "Return (lr)", eid+1, sum(returns) / len(returns))
        plotter.plot('return', 'max return', "Return (lr)", eid+1, max(returns))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu else torch.device('cuda')

    main(env_id=args.env_name, render=args.render, device=device, lr=args.lr, hidden_sizes=[64,64], epochs=args.epochs)
