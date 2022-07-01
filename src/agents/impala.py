import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import layer_init


class ImpalaNet(nn.Module):
    def __init__(self, n_actions, input_image_channels=3):
        super().__init__()

        def impala_block(in_ch, out_ch):
            return nn.Sequential(
                nn.BatchNorm2d(in_ch),
                layer_init(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResidualBlock(out_ch),
                ResidualBlock(out_ch),
            )

        self.impala_1 = impala_block(input_image_channels, 16)
        self.impala_2 = impala_block(16, 32)
        self.impala_3 = impala_block(32, 32)

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.2),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 512)),
            nn.ReLU(),
        )

        self.value_net = layer_init(nn.Linear(512, 1), std=1)
        self.action_net = layer_init(nn.Linear(512, n_actions), std=0.01)

    def forward(self, x):
        x = self.impala_1(x)
        x = self.impala_2(x)
        x = self.impala_3(x)
        x = self.fully_connected(x)

        a = self.action_net(x)
        a = F.softmax(a, dim=-1)
        v = self.value_net(x).squeeze()
        return a, v


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        from torchvision.models import efficientnet_b0

        efficientnet_b0()
        self.block = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x):
        return x + self.block(x)
