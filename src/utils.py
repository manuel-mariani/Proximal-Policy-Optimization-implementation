import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from replay_buffer import ListTrajectory


def render_trajectory(trajectory: ListTrajectory, fps=60, max_length=-1):
    obs = torch.cat(trajectory.obs)
    obs = torch.permute(obs, (0, 2, 3, 1)).cpu().numpy()
    ani = render_np(obs[:max_length])


def render_np(obs, fps=60):
    """Renders a numpy trajectory using matplotlib"""
    it = iter(obs)
    fig = plt.figure()
    im = plt.imshow(next(it))

    def update(_):
        im.set_array(next(it))
        return [im]

    ani = FuncAnimation(fig, update, blit=True, frames=len(obs), interval=1000 // fps, repeat=False)
    plt.show()
    return ani


def test_net(net: torch.nn.Module, input_size, output_size=None):
    """Simple test on a torch network, used to see if the output is congruent with assumption"""
    x = torch.rand(*input_size)
    out = net(x)
    print("Output_size:", out.size(), out.numel())
    if output_size:
        assert tuple(out.size()) == output_size
    print("✅ Passed")

def set_seeds():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
