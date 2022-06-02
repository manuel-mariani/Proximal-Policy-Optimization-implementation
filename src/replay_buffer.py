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
from utils import obs_to_tensor


@dataclass
class Trajectory:
    obs: Iterable[Tensor] | Tensor
    actions: Iterable[Tensor] | Tensor
    rewards: Iterable[Tensor] | Tensor
    is_first: Iterable[Tensor] | Tensor

    def map(self, func):
        return dict(
            obs=func(self.obs), actions=func(self.actions), rewards=func(self.rewards), is_first=func(self.is_first)
        )


class ListTrajectory(Trajectory):
    obs: List[Tensor]
    actions: List[Tensor]
    rewards: List[Tensor]
    is_first: List[Tensor]

    def append(self, obs, actions, rewards, is_first):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.is_first.append(is_first)

    @staticmethod
    def empty() -> "ListTrajectory":
        return ListTrajectory([], [], [], [])

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

    def __iter__(self):
        for obs, actions, rewards, is_first in zip(self.obs, self.actions, self.rewards, self.is_first):
            yield obs, actions, rewards, is_first


class TensorTrajectory(Trajectory):
    def batch(self, batch_size) -> ListTrajectory:
        if self.rewards.ndim > 1:
            func = lambda x: list(torch.flatten(x, 0, 1).split(batch_size))
        else:
            func = lambda x: list(x.split(batch_size))
        return ListTrajectory(**self.map(func))

    def episodic(self) -> "ListTrajectory":
        _trajectory = ListTrajectory.empty()

        # is_first is a tensor with 1 where a new episode begins
        is_first = self.is_first.clone().T
        is_first[:, 0] = 1  # Enforce starting
        # Take the indices where episodes begin (tensor of size "n starts" x 2)
        m = torch.argwhere(is_first)

        # Take the first episode start
        prev_row = m[0, 0].item()
        prev_col = m[0, 1].item()
        length = is_first.size(1)

        # Iterate through the start indices, taking slices
        for row, col in m[1:, :].numpy():
            # If the current row changes, then append the remaining row
            if row != prev_row:
                row_idx, col_end = prev_row, length
            # Otherwise, append the current window
            else:
                row_idx, col_end = row, col + 1
            # Appending and stepping
            take_slice = lambda x: x[prev_col:col_end, row_idx]
            _trajectory.append(**self.map(take_slice))
            prev_row, prev_col = row, col
        return _trajectory


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.trajectory = ListTrajectory.empty()
        self.buffer_size = buffer_size

    def reset(self):
        self.trajectory = ListTrajectory.empty()

    def generate_single(self, env: Env, agent: Agent, device, progress_bar=False, enable_grad=False):
        self.reset()
        steps = trange(self.buffer_size) if progress_bar else range(progress_bar)

        with torch.set_grad_enabled(enable_grad):
            for _ in steps:
                agent.reset()
                obs = env.reset()
                is_first = True
                while True:
                    obs = obs_to_tensor(obs).to(device)
                    action = agent.act(obs)
                    chosen_action = action.sample().item()
                    next_obs, reward, done, _ = env.step(chosen_action)
                    self.trajectory.append(obs=obs, actions=action, rewards=reward, is_first=is_first)
                    next_obs, is_first = obs, False
                    if done:
                        break
            self.trajectory = self.trajectory.tensor()
        return self.trajectory

    def generate_vectorized(self, venv: gym3.Env, agent: Agent, device, progress_bar=False, enable_grad=False):
        self.reset()
        steps = trange(self.buffer_size) if progress_bar else range(progress_bar)
        with torch.set_grad_enabled(enable_grad):
            for _ in steps:
                rew, obs, first = venv.observe()

                # Convert obs tensor  [N, W, H, C] to [N, C, W, H]
                obs = torch.tensor(obs["rgb"], dtype=torch.float).detach().to(device)
                obs = torch.permute(obs, (0, 3, 1, 2)) / 255

                # Act
                action_dists = agent.act(obs)
                chosen_actions = action_dists.sample()
                venv.act(chosen_actions.cpu().detach().numpy())

                # Remove tensors from GPU (no effect if using cpu)
                obs = obs.to("cpu")
                chosen_actions = chosen_actions.detach().to("cpu")
                rew = torch.from_numpy(rew).detach()
                first = torch.from_numpy(first).detach().bool()

                # Number items
                self.trajectory.append(obs=obs, actions=chosen_actions, rewards=rew, is_first=first)
                # To tensor
        self.trajectory = self.trajectory.tensor()
        return self.trajectory
