import torch
import torch.nn as nn

from utils import layer_init
import torch.nn.functional as F

class NatureNet(nn.Module):
    def __init__(self, n_actions, input_image_channels=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Conv2d(input_image_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
        )

        self.value_net = layer_init(nn.Linear(512, 1), std=1)
        self.action_net = layer_init(nn.Linear(512, n_actions), std=0.01)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fully_connected(x)

        a = self.action_net(x)
        a = F.softmax(a, dim=-1)
        v = self.value_net(x).squeeze()
        return a, v