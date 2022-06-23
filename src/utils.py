import random

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

from trajectories import ListTrajectory


def render_trajectory(trajectory: ListTrajectory, fps=60, max_length=-1):
    """Renders a trajectory (episodic)"""
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


def set_seeds():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
