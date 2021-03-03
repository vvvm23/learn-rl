import torch
import torch.nn as nn

from .helper import get_conv_out

from typing import Tuple

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
            nn.Linear(get_conv_out(self.conv, in_shape), 512),
            nn.ReLU(),
            nn.Linear(512, nb_actions),
        )

    def forward(self, x):
        return self.q_head(self.conv(x / 255.).view(x.shape[0], -1))

if __name__ == '__main__':
    net = DQN((4, 84, 84), 4)
    print(net)

    x = torch.randn(1, 4, 84, 84)
    y = net(x)

    print(f"{x.shape} -> {y.shape}")
