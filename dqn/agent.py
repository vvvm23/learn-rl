import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random

class DQNAgent:
    def __init__(self, 
            net: torch.nn.Module,
            nb_actions: int, gamma: float,
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

        self.device = device

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def act_greedy(self, obs):
        return self.net(obs.to(self.device).unsqueeze(0)).squeeze().argmax().item()

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
        ):
        obs_batch       = obs_batch.to(self.device)
        action_batch    = action_batch.to(self.device)
        reward_batch    = reward_batch.to(self.device)
        next_obs_batch  = next_obs_batch.to(self.device)
        done_batch      = done_batch.to(self.device)

        state_action_values = self.net(obs_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_net(next_obs_batch).max(1)[0]
            next_state_values[done_batch] = 0.0
            next_state_values = next_state_values.detach() # is this detach needed if we are in no_grad?
        expected_state_action_values = next_state_values * self.gamma + reward_batch

        loss = state_action_values.sub(expected_state_action_values).mean().pow(2)
        return loss

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

