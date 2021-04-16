import torch
import torch.nn as nn

from .helper import get_conv_out
from .noisy import NoisyLinear

from typing import Tuple

class DQN(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], nb_actions: int, nb_atoms: int = 51):
        super().__init__()
        self.nb_actions = nb_actions
        self.nb_atoms = nb_atoms
        self.conv = nn.Sequential(
            nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.conv_size = get_conv_out(self.conv, in_shape)

        self.v_head = nn.Sequential(
            NoisyLinear(self.conv_size, 512),
            nn.ReLU(),
            NoisyLinear(512, nb_atoms),
        )

        self.adv_head = nn.Sequential(
            NoisyLinear(self.conv_size, 512),
            nn.ReLU(),
            NoisyLinear(512, nb_actions*nb_atoms)
        )

    def forward(self, x):
        z = self.conv(x).view(x.shape[0], -1)
        v = self.v_head(z).view(-1, 1, self.nb_atoms)
        adv = self.adv_head(z).view(-1, self.nb_actions, self.nb_atoms)
        return v + (adv - adv.mean(dim=1, keepdim=True))


if __name__ == '__main__':
    net = DQN((4, 84, 84), 4)
    print(net)

    x = torch.randn(1, 4, 84, 84)
    y = net(x)

    print(f"{x.shape} -> {y.shape}")
