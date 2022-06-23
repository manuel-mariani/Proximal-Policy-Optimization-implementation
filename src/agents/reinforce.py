import torch
from torch.distributions import Categorical

from agents.agent import TrainableAgent
from models.impala import ImpalaNet
from trajectories import TensorTrajectory


class ReinforceAgent(TrainableAgent):
    def __init__(self, act_space_size: int, epsilon=0.2, val_epsilon=0.1):
        super().__init__(act_space_size, val_epsilon=val_epsilon, epsilon=epsilon)
        # Initialize the network
        self.model = ImpalaNet(n_actions=act_space_size)

    def act(self, obs: torch.Tensor, add_rand=True):
        # with autocast("cpu" if obs.device == "cpu" else "cuda"):
        dist, _ = self.model(obs)
        return Categorical(dist, validate_args=False)

    def loss(self, trajectory: TensorTrajectory, logger=None):
        dist = self.act(trajectory.obs, add_rand=False)
        log_prob = dist.log_prob(trajectory.actions)
        loss = - (log_prob * trajectory.returns).mean()
        return loss
