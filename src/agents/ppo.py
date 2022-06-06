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
from trainer import train_eval


class PPONet(nn.Module):
    def __init__(self, n_features, downsampling, n_actions):
        super().__init__()
        # self.feature_extractor = FeatureExtractor(n_features, downsampling)
        # self.action_net = ActionNet(n_features, n_actions)
        # self.value_net = nn.Sequential(
        #     nn.Linear(n_features, n_features // 4),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Linear(n_features // 4, 1),
        #     nn.Tanh()
        # )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 8, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.LazyLinear(n_features, bias=False),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(n_features, n_actions)
        self.value_net = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Tanh(),
        )

        self.action_net.weight = torch.nn.Parameter(self.action_net.weight / 100)

    def forward(self, x):
        feats = self.feature_extractor(x)
        actions = torch.nn.functional.softmax(self.action_net(feats), dim=-1)
        values = self.value_net(feats).flatten()
        return actions, values


class PPOAgent(TrainableAgent):
    def __init__(self, act_space_size, epsilon=0.05, clip_eps=0.25):
        super().__init__(act_space_size, epsilon)
        self.clip_eps = clip_eps
        self.model = PPONet(128, 1, act_space_size)

    def act(self, obs: torch.Tensor, add_rand=True):
        # Forward
        # with autocast("cpu" if obs.device == "cpu" else "cuda"):
        actions, state_values = self.model(obs)
        return Categorical(probs=actions, validate_args=True), state_values

    def _training_sampling(self, dist: Categorical):
        if self.rng.uniform() < self.epsilon:
            return torch.randint(low=0, high=self.act_space_size, size=(dist.probs.size(0),))
        return dist.sample()

    def loss(self, trajectory: "TensorTrajectory"):
        action_dist, state_values = self.act(trajectory.obs)
        ratio = action_dist.probs / trajectory.probs
        a = trajectory.actions
        # advantage = trajectory.returns
        # advantage = trajectory.rewards
        advantage = trajectory.advantages

        # r = ratio[:, a]
        # r = torch.index_select(ratio, dim=1, index=a)
        r = ratio[:, a]
        e = self.clip_eps
        l_clip = torch.minimum(
            r * advantage,
            torch.clip(r, 1 - e, 1 + e) * advantage,
        )
        l_vf = mse_loss(state_values, trajectory.returns, reduction='none') * 0.5
        # l_vf = torch.clip(l_vf, 0, 2)
        # l_vf = 0
        # l_s = 0
        l_s = action_dist.entropy() * 0.01
        # l_s = 0.1 * torch.clip(action_dist.entropy(), 0, 1000)
        # print("     L_clip", -l_clip.sum().item())
        # print("     L_s", l_vf.sum().item())
        # print("     L_vf", l_vf.mean().item())
        l_clip_s = - (l_clip - l_vf + l_s).sum()
        return l_clip_s


if __name__ == "__main__":
    train_eval(PPOAgent(4), "ppo.pt")
    # agent = PPOAgent(15)
    # agent.load("ppo-06-03.pt")
    # evaluate(agent, "coinrun")