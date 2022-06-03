from abc import ABC, abstractmethod

import numpy as np
import torch
from gym.spaces import Discrete
from torch.distributions import Categorical
from torchinfo import summary


class Agent(ABC):
    @abstractmethod
    def __init__(self, act_space_size):
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


# ======================================================================
class TrainableAgent(Agent, ABC):
    @abstractmethod
    def __init__(self, act_space_size):
        super().__init__(act_space_size)
        self.model: torch.nn.Module = None
        self.is_training = True

    @abstractmethod
    def loss(self, trajectory: "TensorTrajectory"):
        pass

    @property
    def parameters(self):
        return self.model.parameters()

    def compile(self, device, sample_input=torch.rand(32, 3, 64, 64)):
        if self.model == torch.jit.ScriptModule:
            return
        self.model = self.model.to(device)
        self.act(sample_input.to(device))
        self.model = torch.jit.script(self.model)
        s = summary(
            self.model,
            input_size=sample_input.size(),
            mode="train",
            depth=7,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
            ],
        )
        print(s)

    def train(self):
        self.is_training = True
        self.model.train()

    def eval(self):
        self.is_training = False
        self.model.eval()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = torch.jit.load(path)
