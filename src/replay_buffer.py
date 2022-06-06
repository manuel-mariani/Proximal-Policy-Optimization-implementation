from collections import namedtuple
from dataclasses import dataclass
from typing import Iterable, List

import gym3
import numpy as np
import torch
from gym import Env
from torch import Tensor
from torch.distributions import Categorical, Distribution
from tqdm.auto import trange

from src.agents.agent import Agent


@dataclass
class Trajectory:
    obs: List[Tensor] | Tensor
    actions: List[Tensor] | Tensor
    rewards: List[Tensor] | Tensor
    is_first: List[Tensor] | Tensor
    values: List[Tensor] | Tensor
    probs: List[Tensor] | Tensor
    advantages: List[Tensor] | Tensor
    returns: List[Tensor] | Tensor

    def map(self, func):
        return dict(
            obs=func(self.obs),
            actions=func(self.actions),
            rewards=func(self.rewards),
            is_first=func(self.is_first),
            values=func(self.values),
            probs=func(self.probs),
            advantages=func(self.advantages),
            returns=func(self.returns),
        )


class ListTrajectory(Trajectory):
    def append(self, obs, actions, rewards, is_first, probs, values=None, advantages=None, returns=None):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.is_first.append(is_first)
        self.probs.append(probs)
        values = values if values is not None else torch.zeros_like(rewards)
        self.values.append(values)
        advantages = advantages if advantages is not None else torch.zeros_like(rewards)
        self.advantages.append(advantages)
        returns = returns if returns is not None else torch.zeros_like(rewards)
        self.returns.append(returns)

    @staticmethod
    def empty() -> "ListTrajectory":
        return ListTrajectory([], [], [], [], [], [], [], [])

    def tensor(self) -> "TensorTrajectory":
        if isinstance(self.rewards[0], List):
            return TensorTrajectory(**self.map(lambda x: torch.tensor(x)))
        if isinstance(self.rewards[0], Tensor):
            s = self.rewards[0].size()
            same_lengths = sum([t.size() != s for t in self.rewards]) == 0
            if same_lengths:
                return TensorTrajectory(**self.map(lambda x: torch.stack(x)))
            return TensorTrajectory(**self.map(lambda x: torch.cat(x)))
        raise NotImplementedError

    def __iter__(self) -> "TensorTrajectory":
        for values in zip(
            self.obs,
            self.actions,
            self.rewards,
            self.is_first,
            self.values,
            self.probs,
            self.advantages,
            self.returns,
        ):
            yield TensorTrajectory(*values)


class TensorTrajectory(Trajectory):
    def batch(self, batch_size) -> ListTrajectory:
        if self.rewards.ndim > 1:
            func = lambda x: list(torch.flatten(x, 0, 1).split(batch_size))
        else:
            func = lambda x: list(x.split(batch_size))
        return ListTrajectory(**self.map(func))

    def to(self, device):
        return TensorTrajectory(**self.map(lambda x: x.to(device)))

    def episodic(self) -> "ListTrajectory":
        _trajectory = ListTrajectory.empty()
        self.is_first[0, :] = 1

        for col, is_first in enumerate(self.is_first.T):
            rows = list(torch.argwhere(is_first)) + [len(is_first)]
            for row, next_row in zip(rows[:-1], rows[1:]):
                _trajectory.append(**self.map(lambda x: x[row:next_row, col]))

        # Testing
        out_lengths = 0
        for is_f in _trajectory.is_first:
            assert is_f[0]
            assert not torch.any(is_f[1:])
            out_lengths += len(is_f)
        in_lengths = sum(len(t) for t in self.is_first)
        assert out_lengths == in_lengths
        return _trajectory

    def shuffle(self):
        if self.actions.ndim == 1:
            t = self
        else:
            t = TensorTrajectory(**self.map(lambda x: torch.flatten(x, 0, 1)))
        indices = torch.randperm(len(t.actions))
        return TensorTrajectory(**t.map(lambda x: x[indices]))