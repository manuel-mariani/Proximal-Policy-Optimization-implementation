from typing import List

import torch
from torch import Tensor

from trajectories import ListTrajectory


def standardize(tensors: List[Tensor], eps=1e-8, c=None):
    """Standardize a tensor (x - μ) / σ, and if c is provided clip it to (-c, c)"""
    tensor = torch.cat(tensors)
    mean, std = tensor.mean(), tensor.std() + eps
    if c is not None:
        return [((t - mean) / std).clip(-c, c) for t in tensors]
    return [((t - mean) / std) for t in tensors]


def _standardize(tensors: List[Tensor], c=None):
    """Standardize a tensor (x - μ) / σ, and if c is provided clip it to (-c, c)"""
    res = [((t - t.mean()) / (t.std() + 1e-8)) for t in tensors]
    if c is not None:
        return [t.clip(-c, c) for t in res]
    return res


def shape_rewards(episodes: ListTrajectory):
    """Given a trajectory of episodes, change the rewards to aid the training"""
    shaped = []
    for reward, action in zip(episodes.rewards, episodes.actions):
        # If last reward is positive, the episode is successful so add a great reward
        # otherwise if the episode ends before the maximum allowed steps, decrease the reward
        if reward[-1] > 1:
            reward[-1] = 10
        elif len(reward) < 1000:
            reward[-1] = -10

        # Decrease all rewards ∀ time steps, encouraging speed
        reward = reward - 0.01
        # If agent goes right, increase reward
        # reward[action == 1] = reward[action == 1] + 2
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


def reward_pipeline(episodes: ListTrajectory, gamma, _lambda):
    """
    Perform a sequence of in-place operations to the rewards.
    In order: reward shaping, discounting, advantages and standardization
    """
    # shape_rewards(episodes)
    # episodes.rewards = standardize(episodes.rewards)
    discount_returns(episodes, gamma)
    episodes.returns = standardize(episodes.returns)
    gae(episodes, gamma, _lambda)
    episodes.advantages = standardize(episodes.advantages)


# ======================================================================
def win_metrics(episodes: ListTrajectory, key_prefix=None) -> dict:
    """Compute a dictionary of metrics for the given episodes. Used to evaluate the performance of the agent"""
    n_wins = 0
    n_losses = 0
    n_unfinished = 0
    for r in episodes.rewards:
        if r[-1] > 0:
            n_wins += 1
        elif len(r) == 1000:
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
    metrics = dict(
        reward_sum=torch.cat(episodes.rewards).sum().item(),
        returns_sum=torch.cat(episodes.returns).sum().item(),
        advantages_sum=torch.cat(episodes.advantages).sum().item(),
    )
    if key_prefix is None:
        return metrics
    return {f"{key_prefix}{k}": v for k, v in metrics.items()}
