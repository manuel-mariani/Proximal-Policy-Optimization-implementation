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
        raise NotImplementedError
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
                    self.trajectory.append(obs=obs, actions=action, rewards=reward, is_first=is_first, values=None, probs=None)
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
                agent_output = agent.act(obs)
                if isinstance(agent_output, tuple):
                    action_dist, values = agent_output
                else:
                    action_dist = agent_output
                    values = torch.zeros(obs.size(0))
                chosen_actions = action_dist.sample()
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
