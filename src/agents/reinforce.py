import numpy as np
import torch
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.distributions import Categorical
from torchinfo import summary
from tqdm.auto import tqdm

from agents.agent import Agent, TrainableAgent
from models.action_net import ActionNet
from models.feature_extractor import FeatureExtractor
from replay_buffer import ReplayBuffer, TensorTrajectory
from trainer import evaluate, train, train_eval
from utils import discount, generate_environment, generate_vec_environment, obs_to_tensor, onehot


# ======================================================================
class ReinforceAgent(TrainableAgent):
    def __init__(self, act_space_size: int, epsilon=0.2):
        super().__init__(act_space_size, epsilon)
        # Initialize the network
        n_features = 64
        self.feature_extractor = FeatureExtractor(n_features, downsampling=1)
        self.action_net = ActionNet(n_features, act_space_size)
        self.model = nn.Sequential(self.feature_extractor, self.action_net)

    def act(self, obs: torch.Tensor, add_rand=True):
        # with autocast("cpu" if obs.device == "cpu" else "cuda"):
        dist = self.model(obs)
        return Categorical(dist, validate_args=False)

    def loss(self, trajectory: TensorTrajectory):
        dist = self.act(trajectory.obs, add_rand=False)
        log_prob = dist.log_prob(trajectory.actions)
        loss = - torch.sum(log_prob * trajectory.rewards)
        return loss


if __name__ == "__main__":
    train_eval(ReinforceAgent(15), "reinforce.pt")
