from abc import ABC, abstractmethod

import numpy as np
import torch
from gym.spaces import Box, Discrete

# ======================================================================
from torch.distributions import Categorical


class Agent(ABC):
    @abstractmethod
    def __init__(self, act_space_size):
        """
        :param train:
        :param act_space_size:
        """
        self.act_space_size = act_space_size
        self.rng = np.random.default_rng()

    @abstractmethod
    def act(self, obs: torch.Tensor) -> Categorical:
        pass

    def reset(self):
        pass

    def _one_hot(self, action_idx: int) -> Categorical:
        dist = torch.zeros(self.act_space_size)
        dist[action_idx] = 1.0
        return Categorical(dist)


# ======================================================================
class RandomAgent(Agent):
    def __init__(self, act_space_size: Discrete):
        super().__init__(act_space_size)

    def act(self, obs: torch.Tensor) -> Categorical:
        return Categorical(torch.rand(self.act_space_size.n))
