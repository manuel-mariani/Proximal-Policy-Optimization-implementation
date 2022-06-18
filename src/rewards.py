from typing import List

import torch
from torch import Tensor

from trajectories import ListTrajectory


def standardize(tensors: List[Tensor], eps=1e-5, c=3):
    """Standardize a tensor (x - μ) / σ and clip it to (-c, c)"""
    tensor = torch.cat(tensors)
    mean, std = tensor.mean(), tensor.std(0) + eps
    return [((t - mean) / std).clip(-c, c) for t in tensors]


def shape_rewards(episodes: ListTrajectory):
    """Given a trajectory of episodes, change the rewards to aid the training"""
    shaped = []
    for reward, action in zip(episodes.rewards, episodes.actions):
        # If last reward is positive, the episode is successful so add a great reward
        # otherwise add a penalty
        if reward[-1] > 1:
            reward[-1] = 10
        else:
            reward[-1] = -10
        # Decrease all rewards ∀ time steps, encouraging speed
        reward = reward - 1
        # If agent goes right, increase reward
        reward[action == 1] = reward[action == 1] + 2
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
    shape_rewards(episodes)
    discount_returns(episodes, gamma)
    gae(episodes, gamma, _lambda)

    episodes.rewards = standardize(episodes.rewards)
    episodes.returns = standardize(episodes.returns)
    episodes.advantages = standardize(episodes.advantages)


# ======================================================================
def episodes_metric(episodes: ListTrajectory, key_prefix=None) -> dict:
    """Compute a dictionary of metrics for the given episodes. Used to evaluate the performance of the agent"""
    n_wins = 0
    for r in episodes.rewards:
        if r[-1] > 0:
            n_wins += 1
    n_losses = len(episodes.rewards) - n_wins
    metrics = dict(
        n_wins=n_wins,
        n_losses=n_losses,
        win_ratio=(n_wins - n_losses) / (n_wins + n_losses),
        reward_sum=torch.cat(episodes.rewards).sum().item(),
        returns_sum=torch.cat(episodes.returns).sum().item(),
    )
    if key_prefix is None:
        return metrics
    return {f"{key_prefix}{k}": v for k, v in metrics.items()}
