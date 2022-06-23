from torch import nn
import torch.nn.functional as F
import torch
from torchvision import ops

class ImpalaNet(nn.Module):
    def __init__(self, n_actions, input_image_channels=3):
        super().__init__()

        def impala_block(in_ch, out_ch):
            return nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResidualBlock(out_ch),
                ResidualBlock(out_ch),
                # nn.BatchNorm2d(out_ch),
            )

        self.impala_1 = impala_block(input_image_channels, 16)
        self.impala_2 = impala_block(16, 32)
        self.impala_3 = impala_block(32, 32)

        self.fully_connected = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.InstanceNorm2d(32, affine=True, track_running_stats=True),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.Tanh(),
            # nn.BatchNorm1d(256),
        )

        self.action_net = nn.Linear(256, n_actions)
        self.value_net = nn.Linear(256, 1)

        self.action_net.weight = torch.nn.Parameter(self.action_net.weight / 100)
        self.value_net.weight = torch.nn.Parameter(self.value_net.weight / 10)

    def forward(self, x):
        x = self.impala_1(x)
        x = self.impala_2(x)
        x = self.impala_3(x)
        x = self.fully_connected(x)

        a = self.action_net(x)
        a = F.softmax(a, dim=-1)
        v = self.value_net(x)
        v = F.tanh(v) * 15
        # v = v.clip(-10, 10)
        return a, v


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ch),
            # ops.SqueezeExcitation(input_channels=ch, squeeze_channels=ch//8),
        )

    def forward(self, x):
        return x + self.block(x)
