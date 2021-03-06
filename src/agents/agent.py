from abc import ABC, abstractmethod
from datetime import datetime

from gym.spaces import Discrete
import numpy as np
import torch
from torch.distributions import Categorical
from torchinfo import summary

from logger import Logger
from trajectories import TensorTrajectory


class Agent(ABC):
    @abstractmethod
    def __init__(self, act_space_size):
        self.act_space_size = act_space_size
        self.rng = np.random.default_rng(seed=42)

    @abstractmethod
    def act(self, obs: torch.Tensor) -> Categorical:
        pass

    def sampling_strategy(self, dist: Categorical):
        return dist.sample()


# ======================================================================
class RandomAgent(Agent):
    def __init__(self, act_space_size: Discrete):
        super().__init__(act_space_size)

    def act(self, obs: torch.Tensor) -> Categorical:
        return Categorical(torch.rand(self.act_space_size.n))


# ======================================================================
class TrainableAgent(Agent, ABC):
    @abstractmethod
    def __init__(self, act_space_size, val_epsilon, epsilon=0.1):
        super().__init__(act_space_size)
        self.model: torch.nn.Module = torch.nn.Identity()
        self.epsilon = epsilon
        self.val_epsilon = val_epsilon
        self.is_training = True

    @abstractmethod
    def loss(self, trajectory: TensorTrajectory, logger: Logger = None):
        pass

    @property
    def parameters(self):
        return self.model.parameters()

    def compile(self, device, sample_input=torch.rand(32, 3, 64, 64)):
        if self.model == torch.jit.ScriptModule:
            return
        self.model = self.model.to(device)
        self.act(sample_input.to(device))
        s = summary(
            self.model,
            input_data=sample_input,
            mode="train",
            depth=7,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
            ],
        )
        self.model = torch.jit.script(self.model)
        return self

    def train(self):
        self.is_training = True
        self.model.train()
        return self

    def eval(self):
        self.is_training = False
        self.model.eval()
        return self

    def save(self, path=None):
        if path is None:
            name = type(self).__name__
            today = datetime.now().strftime("%m%d-%H%M")
            path = f"../trained_models/{name}-{today}.pt"
        self.model.save(path)
        print("Model saved in ", path)

    def load(self, path, device):
        self.model = torch.jit.load(path)
        self.model.to(device)

    def sampling_strategy(self, dist: Categorical):
        if self.is_training:
            return self._training_sampling(dist)
        return self._evaluation_sampling(dist)

    def _training_sampling(self, dist: Categorical):
        if self.rng.uniform() < self.epsilon:
            return torch.randint(low=0, high=self.act_space_size, size=(dist.probs.size(0),))
        return dist.sample()

    def _evaluation_sampling(self, dist: Categorical):
        if self.rng.uniform() < self.val_epsilon:
            return dist.sample()
        return torch.argmax(dist.probs, dim=-1)