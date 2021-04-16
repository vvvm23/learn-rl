import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random

class DQNAgent:
    def __init__(self, 
            net: torch.nn.Module,
            nb_actions: int, 
            gamma: float = 0.99, unroll_steps: int = 1,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ):

        self.net = net
        self.target_net = copy.deepcopy(net)

        self.net = self.net.to(device)
        self.target_net = self.target_net.to(device)

        self.net.train()
        self.target_net.train()

        self.nb_actions = nb_actions
        self.gamma = gamma
        self.unroll_steps = unroll_steps

        self.device = device

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def act_greedy(self, obs):
        return self.net(obs.to(self.device).unsqueeze(0) / 255.).squeeze().argmax().item()

    def act_epsilon_greedy(self, obs, epsilon: float):
        if random.random() < epsilon:
            return random.randint(0, self.nb_actions - 1)
        return self.act_greedy(obs)

    def calculate_loss(self, 
            obs_batch, 
            action_batch, 
            reward_batch,
            next_obs_batch,
            done_batch,
            double=False,
        ):
        obs_batch       = obs_batch.to(self.device) / 255.
        action_batch    = action_batch.to(self.device)
        reward_batch    = reward_batch.to(self.device)
        next_obs_batch  = next_obs_batch.to(self.device) / 255.
        done_batch      = done_batch.to(self.device)

        state_action_values = self.net(obs_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            if double:
                next_state_actions = self.net(next_obs_batch).argmax(-1).unsqueeze(-1)
                next_state_values = self.target_net(next_obs_batch).gather(1, next_state_actions).squeeze(-1)
            else:
                next_state_values = self.target_net(next_obs_batch).max(-1)[0]
            next_state_values[done_batch] = 0.0
            next_state_values = next_state_values.detach() # is this detach needed if we are in no_grad?
        expected_state_action_values = next_state_values * (self.gamma ** self.unroll_steps) + reward_batch

        # loss = F.mse_loss(state_action_values, expected_state_action_values)
        loss = (state_action_values - expected_state_action_values).pow(2)
        return loss

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.sync_target()

class CategoricalDQNAgent:
    def __init__(self, 
            net: torch.nn.Module,
            nb_actions: int, 
            gamma: float = 0.99, unroll_steps: int = 1,
            nb_atoms: int = 51, vmin: float = -10., vmax: float = 10.,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ):

        self.net = net
        self.target_net = copy.deepcopy(net)

        self.net = self.net.to(device)
        self.target_net = self.target_net.to(device)

        self.net.train()
        self.target_net.train()

        self.nb_actions = nb_actions
        self.gamma = gamma
        self.unroll_steps = unroll_steps

        self.device = device

        # start of categorical DQN agent changes #
        self.vmin = vmin
        self.vmax = vmax
        self.nb_atoms = nb_atoms

        self.support = torch.linspace(vmin, vmax, nb_atoms).to(device)
        self.dz = (vmax - vmin) / (nb_atoms - 1)

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def act_greedy(self, obs):
        # return self.net(obs.to(self.device).unsqueeze(0) / 255.).squeeze().argmax().item()
        return (F.softmax(self.net(obs.to(self.device).unsqueeze(0) / 255.), dim=-1) * self.support).sum(-1).argmax(-1).item()

    def act_epsilon_greedy(self, obs, epsilon: float):
        if random.random() < epsilon:
            return random.randint(0, self.nb_actions - 1)
        return self.act_greedy(obs)

    def calculate_loss(self, 
            obs_batch, 
            action_batch, 
            reward_batch,
            next_obs_batch,
            done_batch,
            double=False,
        ):
        obs_batch       = obs_batch.to(self.device) / 255.
        action_batch    = action_batch.to(self.device)
        reward_batch    = reward_batch.to(self.device)
        next_obs_batch  = next_obs_batch.to(self.device) / 255.
        done_batch      = done_batch.to(self.device)

        batch_size = obs_batch.shape[0]

        state_action_values = F.log_softmax(self.net(obs_batch), dim=-1).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            if double:
                pns = self.net(next_obs_batch)
                dns = self.support.expand_as(pns) * pns
                next_state_actions = dns.sum(-1).argmax(-1).unsqueeze(-1)

                pns = F.softmax(self.target_net(next_obs_batch), dim=-1)
                next_state_values = pns.gather(1, next_state_actions).squeeze(-1)
                # next_state_actions = self.net(next_obs_batch).argmax(-1).unsqueeze(-1)
                # next_state_values = self.target_net(next_obs_batch).gather(1, next_state_actions).squeeze(-1)
            else:
                next_state_values = F.softmax(self.target_net(next_obs_batch)).max(-1)[0]
            # next_state_values[done_batch] = 0.0
            # next_state_values = next_state_values.detach() # is this detach needed if we are in no_grad?

            tz = (1.0 - done_batch) * (self.gamma ** self.unroll_steps) * self.support.unsqueeze(0) + reward_batch
            tz = tz.clamp(min=self.vmin, max=self.vmax)
            b = (tz - self.vmin) / self.dz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.nb_atoms - 1)) * (l == u)] += 1

            m = obs_batch.new_zeros(batch_size, self.nb_atoms)
            offset = torch.linspace(0, ((batch_size - 1) * self.nb_atoms), batch_size).unsqueeze(1).expand(batch_size, self.nb_atoms).to(action_batch)
            m.view(-1).index_add_(0, (l+offset).view(-1), (next_state_values * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u+offset).view(-1), (next_state_values * (b - l.float())).view(-1))

        # loss = (state_action_values - expected_state_action_values).pow(2)
        loss = -torch.sum(m * state_action_values)
        return loss

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.sync_target()

if __name__ == '__main__':
    nb_actions = 4
    net = nn.Linear(8, nb_actions)
    agent = DQNAgent(net, nb_actions, gamma=0.99)
    opt = torch.optim.Adam(net.parameters())

    agent.sync_target()

    opt.zero_grad()
    loss = agent.calculate_loss(
        torch.randn(8, 8),
        torch.randint(0, nb_actions, (8,)),
        torch.randn(8),
        torch.randn(8,8),
        torch.randint(0, 2, (8,)),
    )
    loss.backward()
    opt.step()
    print(loss.item())

