import torch.nn.functional as F
from torch import nn

from utils import test_net


# ======================================================================
class FeatureExtractor(nn.Module):
    def __init__(self, out_features, downsampling=2):
        self.ds = downsampling
        super().__init__()

        # ResNet
        self.cnn = nn.Sequential(
            ResidualBottleneckBlock(3, 32),
            ResidualBottleneckBlock(32, 32),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(32),
            ResidualBottleneckBlock(32, 32),
            ResidualBottleneckBlock(32, 32),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(32),
            ResidualBottleneckBlock(32, 8),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(8),
        )

        # Linear layer
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features),
            nn.Tanh(),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1 / self.ds, mode='nearest')  # Downscale
        x = x / x.max()
        x = self.cnn(x)
        x = self.linear(x)
        return x


# ======================================================================
class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.shortcut = nn.Identity()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            conv_bn_block(in_ch, out_ch, 1),  # 1x1
            conv_bn_block(out_ch, out_ch, 3, padding=1, stride=stride),  # 3x3
            conv_bn_block(out_ch, out_ch, 1, activate=False),  # 1x1
        )

        # If the output channels are different from the input, also convolve the skip connection
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride)

    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.block(x)
        out = residual + identity
        return out


# ======================================================================
def conv_bn_block(in_ch, out_ch, kernel_size, stride=1, padding=0, activate=True, batch_norm=False) -> nn.Module:
    """Builds a 2D Convolution, followed by BatchNorm and LeakyRelu (optional)"""
    conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    if not activate and not batch_norm:
        return conv2d
    net = nn.Sequential(conv2d)
    if batch_norm:
        net.append(nn.BatchNorm2d(out_ch))
    if activate:
        net.append(nn.LeakyReLU(0.01, inplace=True))
    return net


if __name__ == "__main__":
    test_net(
        net=FeatureExtractor(out_features=100),
        input_size=(1, 3, 64, 64),
    )
