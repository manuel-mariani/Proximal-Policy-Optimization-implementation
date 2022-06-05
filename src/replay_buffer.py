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

    def map(self, func):
        return dict(
            obs=func(self.obs),
            actions=func(self.actions),
            rewards=func(self.rewards),
            is_first=func(self.is_first),
            values=func(self.values),
            probs=func(self.probs),
        )


class ListTrajectory(Trajectory):
    def append(self, obs, actions, rewards, is_first, values, probs):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.is_first.append(is_first)
        self.values.append(values)
        self.probs.append(probs)

    @staticmethod
    def empty() -> "ListTrajectory":
        return ListTrajectory([], [], [], [], [], [])

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


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.trajectory = ListTrajectory.empty()
        self.buffer_size = buffer_size

    def reset(self):
        self.trajectory = ListTrajectory.empty()

    def generate_single(self, env: Env, agent: Agent, device, progress_bar=False, enable_grad=False):
        raise NotImplementedError
        self.reset()
        steps = trange(self.buffer_size) if progress_bar else range(progress_bar)

        with torch.set_grad_enabled(enable_grad):
            for _ in steps:
                agent.reset()
                obs = env.reset()
                is_first = True
                while True:
                    # obs = obs_to_tensor(obs).to(device)
                    action = agent.act(obs)
                    chosen_action = action.sample().item()
                    next_obs, reward, done, _ = env.step(chosen_action)
                    self.trajectory.append(obs=obs, actions=action, rewards=reward, is_first=is_first, values=None, probs=None)
                    next_obs, is_first = obs, False
                    if done:
                        break
            self.trajectory = self.trajectory.tensor()
        return self.trajectory

    def generate_vectorized(self, venv, agent: Agent, device, progress_bar=False, enable_grad=False, sampling_strategy=None):
        self.reset()
        steps = trange(self.buffer_size) if progress_bar else range(self.buffer_size)
        sampling_strategy = sampling_strategy if sampling_strategy is not None else lambda x: x.sample()

        with torch.set_grad_enabled(enable_grad):
            for _ in steps:
                rew, obs, first = venv.observe()

                # Convert obs tensor  [N, W, H, C] to [N, C, W, H]
                obs = torch.tensor(obs["rgb"], dtype=torch.float).detach().to(device)
                obs = torch.permute(obs, (0, 3, 1, 2)) / 255

                # Act
                agent_output = agent.act(obs)
                if isinstance(agent_output, tuple):
                    action_dist, values = agent_output
                else:
                    action_dist = agent_output
                    values = torch.zeros(obs.size(0))
                chosen_actions = agent.sampling_strategy(action_dist)
                probs = action_dist.probs
                venv.act(chosen_actions.cpu().detach().numpy())

                # Remove tensors from GPU (no effect if using cpu)
                obs = obs.to("cpu")
                chosen_actions = chosen_actions.detach().to("cpu")
                rew = torch.from_numpy(rew).detach()
                first = torch.from_numpy(first).detach().bool()
                values = values.detach().to("cpu")
                probs = probs.detach().to("cpu")

                # Number items
                self.trajectory.append(
                    obs=obs,
                    actions=chosen_actions,
                    rewards=rew,
                    is_first=first,
                    values=values,
                    probs=probs,
                )
        # To tensor
        self.trajectory = self.trajectory.tensor()
        return self.trajectory
