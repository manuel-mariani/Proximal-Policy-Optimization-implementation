from typing import List

import numpy as np
import torch
from torch import Tensor
from welford import Welford

from trajectories import ListTrajectory


def shape_rewards(episodes: ListTrajectory, max_ep_len):
    """Given a trajectory of episodes, change the rewards to aid the training"""
    shaped = []
    for reward, action in zip(episodes.rewards, episodes.actions):
        # If last reward is positive, the episode is successful so add a great reward
        # otherwise if the episode ends before the maximum allowed steps, decrease the reward
        if reward[-1] > 0:
            reward[-1] = 10
        elif len(reward) < max_ep_len:
            reward[-1] = -1

        # Decrease all rewards ∀ time steps, encouraging speed
        reward = reward - 1e-3
        shaped.append(reward)
    episodes.rewards = shaped


def discount_returns(episodes: ListTrajectory, gamma):
    """Reward discounting: Vₜ = Rₜ + γVₜ₊₁"""
    discounted = []
    for er in episodes.rewards:
        er_flip = er.flip((0,))
        dr = [er_flip[0]]
        for r in er_flip[1:]:
            dr.append(r + gamma * dr[-1])
        dr = torch.tensor(dr).flip((0,))
        discounted.append(dr)
    episodes.returns = discounted


def gae(episodes: ListTrajectory, gamma, _lambda):
    """Generalized Advantage Estimate"""
    advantages = []
    for episode in episodes:
        r_flip = episode.rewards.flip((0,))
        v_flip = episode.values.flip((0,))

        v_prev = v_flip[0]
        adv = [r_flip[0] - v_flip[0]]

        for r, v in zip(r_flip[1:], v_flip[1:]):
            delta = r - v + gamma * v_prev  # δₜ = rₜ + γV(sₜ₊₁) - V(s)
            a = delta + gamma * _lambda * adv[-1]  # Aₜ = δₜ + λγAₜ₊₁
            adv.append(r - v + gamma * adv[-1])
            v_prev = v
        adv = torch.tensor(adv).flip((0,))
        advantages.append(adv)
    episodes.advantages = advantages


def reward_pipeline(episodes: ListTrajectory, max_ep_len, gamma, _lambda):
    """
    Perform a sequence of in-place operations to the rewards.
    In order: reward shaping, discounting, advantages and standardization
    """
    shape_rewards(episodes, max_ep_len=max_ep_len)

    discount_returns(episodes, gamma)
    # episodes.returns = standardize(episodes.returns)
    episodes.returns = welford_standardizer(episodes.returns, returns_welford, shift_mean=True)
    gae(episodes, gamma, _lambda)
    # episodes.advantages = standardize(episodes.advantages)
    episodes.advantages = welford_standardizer(episodes.advantages, advantages_welford, shift_mean=True)


# ======================================================================


def standardize(tensors: List[Tensor], eps=1e-8, c=None, shift_mean=True):
    """Standardize a tensor (x - μ) / σ, and if c is provided clip it to (-c, c)"""
    tensor = torch.cat(tensors)
    mean, std = tensor.mean(), tensor.std(0) + eps
    res = tensors
    if shift_mean:
        res = [t - mean for t in res]
    res = [t / std for t in res]
    if c is not None:
        res = [t.clip(-c, c) for t in res]
    return res


def welford_standardizer(tensors: List[Tensor], w: Welford, shift_mean=False):
    """Standardize a list of tensors using a running stat (Welford Online Algorithm)"""
    values = torch.cat(tensors).numpy()
    w.add_all(values)
    mean, std = w.mean, np.sqrt(w.var_p) + 1e-8
    if shift_mean:
        res = [(t - mean) / std for t in tensors]
    else:
        res = [t / std for t in tensors]
    return res


rewards_welford = Welford()
returns_welford = Welford()
advantages_welford = Welford()


# ======================================================================


def win_metrics(episodes: ListTrajectory, max_ep_len, key_prefix=None) -> dict:
    """Compute a dictionary of metrics for the given episodes. Used to evaluate the performance of the agent"""
    n_wins = 0
    n_losses = 0
    n_unfinished = 0
    for r in episodes.rewards:
        if r[-1] > 0:
            n_wins += 1
        elif len(r) == max_ep_len:
            n_unfinished += 1
        else:
            n_losses += 1

    metrics = dict(
        n_wins=n_wins,
        n_losses=n_losses,
        n_unfinished=n_unfinished,
        win_ratio=n_wins / (n_wins + n_losses + n_unfinished),
    )
    if key_prefix is None:
        return metrics
    return {f"{key_prefix}{k}": v for k, v in metrics.items()}


def reward_metrics(episodes: ListTrajectory, key_prefix=None) -> dict:
    """Compute a dictionary of metrics for the given episodes. Used to evaluate the performance of the agent"""
    metrics = dict(
        reward_sum=torch.cat(episodes.rewards).sum().item(),
        returns_sum=torch.cat(episodes.returns).sum().item(),
        advantages_sum=torch.cat(episodes.advantages).sum().item(),
    )

    if key_prefix is None:
        return metrics
    return {f"{key_prefix}{k}": v for k, v in metrics.items()}


def action_metrics(episodes: ListTrajectory, key_prefix="") -> dict:
    """Compute a dict representing the distribution of actions in the episodes"""
    d = dict()
    actions, counts = torch.cat(episodes.actions).unique(return_counts=True)
    for a, c in torch.stack((actions, counts), dim=1):
        d[f"{key_prefix}action_{a.item()}"] = c.item()
    return d