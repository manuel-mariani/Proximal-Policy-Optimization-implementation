from dataclasses import dataclass
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor


@dataclass
class Trajectory:
    obs: List[Tensor] | Tensor  # Observations
    actions: List[Tensor] | Tensor  # Actions
    rewards: List[Tensor] | Tensor  # Rewards
    is_first: List[Tensor] | Tensor  # Flag if the state is the first of an episode
    values: List[Tensor] | Tensor  # Values estimated by the value-net
    probs: List[Tensor] | Tensor  # Probabilities output of the action-net
    advantages: List[Tensor] | Tensor  # Advantages (GAE)
    returns: List[Tensor] | Tensor  # Returns (discounted rewards)

    def map(self, func):
        """
        Convenience function to map a function to all of the trajectory attributes
        :param func: function (Tensor | List) -> Any
        :return: A dictionary attribute_name -> func(attribute_val)
        """
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
    """Trajectory containing a List of tensors"""

    def append(self, obs, actions, rewards, is_first, probs, values=None, advantages=None, returns=None):
        """Append values to the lists"""
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
        """Create a new empty ListTrajectory"""
        return ListTrajectory([], [], [], [], [], [], [], [])

    def tensor(self) -> "TensorTrajectory":
        """Convert the ListTrajectory to a TensorTrajectory"""
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
        """Iterate through the trajectory, yielding a "point" TensorTrajectory for each instant of time"""
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
    """Trajectory containing only Tensors. Can be either a point (one time instant) or N dimensional"""

    def batch(self, batch_size) -> ListTrajectory:
        """Split the tensors to be of shape (batch_size, ...), returning a ListTrajectory with the splitted tensors"""
        if self.rewards.ndim > 1:
            func = lambda x: list(torch.flatten(x, 0, 1).split(batch_size))
        else:
            func = lambda x: list(x.split(batch_size))
        return ListTrajectory(**self.map(func))

    def to(self, device):
        """Put the tensors onto a device"""
        return TensorTrajectory(**self.map(lambda x: x.to(device)))

    def episodic(self) -> "ListTrajectory":
        """
        Convert the trajectory to an episodic list trajectory,
        where each trajectory in the list starts at the first step of the episode
        """
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
        """Random shuffle the tensors along the 1st dimension"""
        if self.actions.ndim == 1:
            t = self
        else:
            t = TensorTrajectory(**self.map(lambda x: torch.flatten(x, 0, 1)))
        indices = torch.randperm(len(t.actions))
        return TensorTrajectory(**t.map(lambda x: x[indices]))
