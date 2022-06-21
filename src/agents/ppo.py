import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import mse_loss

from agents.agent import TrainableAgent
from models.impala import ImpalaNet
from trajectories import TensorTrajectory


class PPONet(nn.Module):
    def __init__(self, n_features, downsampling, n_actions):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 8, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.LazyLinear(n_features, bias=False),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(n_features, n_actions)
        self.value_net = nn.Linear(n_features, 1)
        # self.value_net = nn.Sequential(
        #     nn.Linear(n_features, n_features // 2),
        #     nn.Tanh(),
        #     nn.Linear(n_features // 2, 1),
        # )
        # self.value_net = nn.Sequential(
        #     nn.Linear(n_features, 1),
        #     nn.Tanh(),
        # )

        self.action_net.weight = torch.nn.Parameter(self.action_net.weight / 100)
        self.value_net.weight = torch.nn.Parameter(self.value_net.weight / 100)

    def forward(self, x):
        feats = self.feature_extractor(x)
        actions = torch.nn.functional.softmax(self.action_net(feats), dim=-1)
        values = self.value_net(feats).flatten().clip(-30, 30)
        return actions, values


class PPOAgent(TrainableAgent):
    def __init__(self, act_space_size, epsilon=0.05, val_epsilon=0.1, clip_eps=0.25):
        super().__init__(act_space_size, val_epsilon=val_epsilon, epsilon=epsilon)
        self.clip_eps = clip_eps
        # self.model = PPONet(32, 1, act_space_size)
        self.model = ImpalaNet(n_actions=act_space_size)

    def act(self, obs: torch.Tensor, add_rand=True):
        actions, state_values = self.model(obs)
        return Categorical(probs=actions, validate_args=True), state_values

    def _training_sampling(self, dist: Categorical):
        if self.rng.uniform() < self.epsilon:
            return torch.randint(low=0, high=self.act_space_size, size=(dist.probs.size(0),))
        return dist.sample()

    def loss(self, trajectory: "TensorTrajectory", logger=None):
        action_dist, state_values = self.act(trajectory.obs)
        e = self.clip_eps
        a = trajectory.actions
        r = action_dist.probs[:, a] / trajectory.probs[:, a]
        # advantage = trajectory.returns - state_values
        advantage = trajectory.advantages
        # if advantage.max() != 0:
        #     advantage = advantage / advantage.max()
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # advantage = advantage / (advantage.std() + 1e-8)
        # Compute the losses (tensors of dim [Batch, 1]).
        # Respectively they represent the CLIP loss, the Value Function error and the Entropy regularization
        l_clip = torch.minimum(
            r * advantage,
            torch.clip(r, 1 - e, 1 + e) * advantage,
        )

        returns = trajectory.returns.clip(-10, 10)
        # if returns.max() != 0:
        #     returns = returns / returns.max()
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # TODO: REVERT THIS
        # returns = returns / (returns.std() + 1e-8)
        l_vf = mse_loss(state_values, returns, reduction="none")
        l_s = action_dist.entropy()

        # Reduce the losses, scaling them by constants and sum them. Terms with '-' are to be maximized
        l_clip = -l_clip.mean()
        l_vf = l_vf.mean() * 0.5
        # l_s = torch.tensor(0.0)
        l_s = -l_s.mean() * 0.01
        l_clip_vf_s = l_clip + l_vf + l_s

        if logger:
            logger.append(
                loss_clip=l_clip.item(),
                loss_value_mse=l_vf.item(),
                loss_entropy=l_s.item(),
            )

        return l_clip_vf_s
