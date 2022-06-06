from typing import List

import torch
from torch import Tensor

from replay_buffer import ListTrajectory


def standardize(tensors: List[Tensor], eps=1e-5):
    tensor = torch.cat(tensors)
    mean, std = tensor.mean(), tensor.std(0) + eps
    return [(t - mean) / std for t in tensors]


def shape_rewards(episodes: ListTrajectory):
    shaped = []
    for er, act in zip(episodes.rewards, episodes.actions):
        # If last reward is positive, the episode is successful so add a great reward
        # otherwise add a penalty
        if er[-1] > 1:
            er[-1] = 50
        else:
            er[-1] = -10
        # Decrease all rewards ∀ time steps, encouraging speed
        er = er - 1
        # If agent goes right, increase reward
        er[act == 1] = er[act == 1] + 2
        shaped.append(er)
    episodes.rewards = shaped


def discount_returns(episodes: ListTrajectory, gamma):
    discounted = []
    for er in episodes.rewards:
        er_flip = er.flip((0, ))
        dr = [er_flip[0]]
        for r in er_flip[1:]:
            dr.append(r + gamma * dr[-1])
        dr = torch.tensor(dr).flip((0, ))
        discounted.append(dr)
    episodes.returns = discounted

def gae(episodes: ListTrajectory, gamma, _lambda):
    advantages = []
    for episode in episodes:
        r_flip = episode.rewards.flip((0, ))
        v_flip = episode.values.flip((0, ))

        v_prev = v_flip[0]
        adv = [r_flip[0] - v_flip[0]]

        for r, v in zip(r_flip[1:], v_flip[1:]):
            delta = r - v + gamma * v_prev  # δₜ = rₜ + γV(sₜ₊₁) - V(s)
            a = delta + gamma * _lambda * adv[-1]  # Aₜ = δₜ + λγAₜ₊₁
            adv.append(r - v + gamma * adv[-1])
            v_prev = v
        adv = torch.tensor(adv).flip((0, ))
        advantages.append(adv)
    episodes.advantages = advantages


def reward_pipeline(episodes: ListTrajectory, gamma, _lambda):
    shape_rewards(episodes)
    discount_returns(episodes, gamma)
    gae(episodes, gamma, _lambda)

    episodes.rewards = standardize(episodes.rewards)
    episodes.returns = standardize(episodes.returns)
    episodes.advantages = standardize(episodes.advantages)
