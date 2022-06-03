import numpy as np
import torch
from torch import autocast, nn
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
from torchinfo import summary
from tqdm.auto import tqdm

from agents.agent import Agent, TrainableAgent
from models.action_net import ActionNet
from models.feature_extractor import FeatureExtractor
from replay_buffer import ReplayBuffer, TensorTrajectory
from trainer import evaluate, train_eval
from utils import discount, generate_environment, generate_vec_environment, obs_to_tensor, onehot


class PPONet(nn.Module):
    def __init__(self, n_features, downsampling, n_actions):
        super().__init__()
        self.feature_extractor = FeatureExtractor(n_features, downsampling)
        self.action_net = ActionNet(n_features, n_actions)
        self.value_net = nn.Sequential(
            nn.Linear(n_features, n_features // 4),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(n_features // 4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        actions = self.action_net(feats)
        values = self.value_net(feats).flatten() * 2
        return actions, values


class PPOAgent(TrainableAgent):
    def __init__(self, act_space_size, epsilon=0.02, clip_eps=0.2):
        super().__init__(act_space_size, epsilon)
        self.clip_eps = clip_eps
        self.model = PPONet(64, 1, act_space_size)

    def act(self, obs: torch.Tensor, add_rand=True):
        # Forward
        # with autocast("cpu" if obs.device == "cpu" else "cuda"):
        actions, state_values = self.model(obs)
        # If evaluating, return just the argmax (one-hotted)
        if not self.is_training:
            actions = onehot(torch.argmax(actions, dim=-1), actions.size())
            return Categorical(actions, validate_args=False)
        # If eps-greedy, return random onehot distribution
        # if add_rand and self.rng.uniform() < self.eps:
        #     actions = onehot(self.rng.integers(0, self.act_space_size), actions.size())
        return Categorical(actions), state_values

    def loss(self, trajectory: "TensorTrajectory"):
        action_dist, state_values = self.act(trajectory.obs)
        ratio = action_dist.probs / trajectory.probs
        a = trajectory.actions
        advantage = trajectory.rewards - state_values
        # advantage = trajectory.rewards

        e = self.clip_eps
        # r = ratio[:, a]
        r = torch.index_select(ratio, dim=1, index=a)
        l_clip = torch.minimum(
            r * advantage,
            torch.clip(r, 1 - e, 1 + e) * advantage,
        )
        l_vf = 1 * mse_loss(state_values, trajectory.rewards, reduction='none')
        l_s = 0.1 * action_dist.entropy()
        # l_s = 0.1 * torch.clip(action_dist.entropy(), 0, 1000)
        l_clip_s = -torch.sum(l_clip - l_vf + l_s)
        return l_clip_s


if __name__ == "__main__":
    train_eval(PPOAgent(15), "ppo.pt")
    # agent = PPOAgent(15)
    # agent.load("ppo-06-03.pt")
    # evaluate(agent, "coinrun")