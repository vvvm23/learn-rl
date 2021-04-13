import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from typing import Tuple

from .helper import get_conv_out

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias != None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(x, v, bias)

class DQN(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], nb_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            NoisyLinear(get_conv_out(self.conv, in_shape), 512),
            nn.ReLU(),
            NoisyLinear(512, nb_actions),
        )

    def forward(self, x):
        return self.q_head(self.conv(x).view(x.shape[0], -1))

if __name__ == '__main__':
    net = DQN((4, 84, 84), 4)
    print(net)

    x = torch.randn(1, 4, 84, 84)
    y = net(x)

    print(f"{x.shape} -> {y.shape}")
