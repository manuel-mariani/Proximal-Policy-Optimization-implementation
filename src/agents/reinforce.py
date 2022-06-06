import torch
from torch import nn
from torch.distributions import Categorical

from agents.agent import TrainableAgent
from agents.ppo import PPONet
from models.action_net import ActionNet
from models.feature_extractor import FeatureExtractor
from replay_buffer import TensorTrajectory
from trainer import train_eval


# ======================================================================
class ReinforceAgent(TrainableAgent):
    def __init__(self, act_space_size: int, epsilon=0.2):
        super().__init__(act_space_size, epsilon)
        # Initialize the network
        n_features = 64
        # self.feature_extractor = FeatureExtractor(n_features, downsampling=1)
        # self.action_net = ActionNet(n_features, act_space_size)
        self.model = PPONet(downsampling=1, n_actions=act_space_size, n_features=n_features)

    def act(self, obs: torch.Tensor, add_rand=True):
        # with autocast("cpu" if obs.device == "cpu" else "cuda"):
        dist, _ = self.model(obs)
        return Categorical(dist, validate_args=False)

    def loss(self, trajectory: TensorTrajectory):
        dist = self.act(trajectory.obs, add_rand=False)
        log_prob = dist.log_prob(trajectory.actions)
        loss = - (log_prob * trajectory.returns).sum()
        return loss


if __name__ == "__main__":
    train_eval(ReinforceAgent(4), "reinforce.pt")
