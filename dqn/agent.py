import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random

class DQNAgent:
    def __init__(self, 
            net: torch.nn.Module,
            nb_actions: int,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ):

        self.net = net
        self.target_net = copy.deepcopy(net)

        self.nb_actions = nb_actions

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def act_greedy(self, obs):
        return self.net(obs.unsqueeze(0)).squeeze().argmax().item()

    def act_epsilon_greedy(self, obs, epsilon: float):
        if random.random() < epsilon:
            return random.randint(0, self.nb_actions - 1)
        return act_greedy(obs)

    def learn(self):
        pass
    

if __name__ == '__main__':
    nb_actions = 4
    net = nn.Linear(8, nb_actions)
    agent = DQNAgent(net, nb_actions)
