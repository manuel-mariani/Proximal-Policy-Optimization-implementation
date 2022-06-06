import random
from typing import Any, List

import gym
import gym3
import numpy as np
import procgen
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


def reward_shaping(ep_rewards, timeout: int, kappa=-1):
    # return ep_rewards
    shaped = []
    for er in ep_rewards:
        # Penalization if the episode goes on for too long
        if len(er) > timeout:
            # er[timeout:] = kappa / (len(er) - timeout)
            er[-1] = kappa
        # Penalization if the episode ends without a reward
        elif er[-1] == 0:
            er[-1] = kappa
        # else:
        #     er[-1] = -kappa
        # Scale the reward by time elapsed (inverse proportional)
        # else:
        #     # er[-1] = er[-1] * (timeout / len(er)) * (-kappa)
        #     er[-1] = (timeout / len(er)) * (-kappa)
        shaped.append(er)
    return shaped


def gae(episodes_trajectory: ListTrajectory, gamma, _lambda):
    advantages = []
    for episode in episodes_trajectory:
        r_flip = torch.flip(episode.rewards, (0,))
        v_flip = torch.flip(episode.values, (0,))

        v_prev = v_flip[0]
        adv = [r_flip[0] - v_flip[0]]

        for r, v in zip(r_flip[1:], v_flip[1:]):
            delta = r - v + gamma * v_prev  # δₜ = rₜ + γ V(sₜ₊₁) - V(s)
            a = delta + gamma * _lambda * adv[-1]  # Aₜ = δₜ + λγ Aₜ₊₁
            adv.append(r - v + gamma * adv[-1])
            v_prev = v

        adv = torch.flip(torch.tensor(adv), (0,))
        advantages.append(adv)
    return advantages


def discount(ep_rewards, gamma):
    discounted = []
    for er in ep_rewards:
        flip_er = torch.flip(er, (0,))
        dr = [flip_er[0]]
        for r in flip_er[1:]:
            dr.append(r + gamma * dr[-1])
        dr = torch.flip(torch.tensor(dr), (0,))
        discounted.append(dr)
    return discounted


def standardize(ep_rewards, eps=0.01):
    standardized = []
    tensor = torch.cat(ep_rewards)
    mean = tensor.mean()
    # mean = 0
    std = tensor.std() + eps
    for er in ep_rewards:
        standardized.append((er - mean) / std)
    return standardized


def _standardize(ep_rewards, eps=0.01):
    standardized = []
    tensor = torch.cat(ep_rewards)
    m = tensor.abs().max()
    if m == 0:
        return ep_rewards
    for er in ep_rewards:
        standardized.append(er / m)
    return standardized


def set_seeds():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
