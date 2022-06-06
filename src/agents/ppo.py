import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import mse_loss

from agents.agent import TrainableAgent
from replay_buffer import TensorTrajectory
from trainer import train_eval


class PPONet(nn.Module):
    def __init__(self, n_features, downsampling, n_actions):
        super().__init__()
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
        values = self.value_net(feats).flatten() * 4
        return actions, values


class PPOAgent(TrainableAgent):
    def __init__(self, act_space_size, epsilon=0.05, clip_eps=0.25):
        super().__init__(act_space_size, epsilon)
        self.clip_eps = clip_eps
        self.model = PPONet(128, 1, act_space_size)

    def act(self, obs: torch.Tensor, add_rand=True):
        actions, state_values = self.model(obs)
        return Categorical(probs=actions, validate_args=True), state_values

    def _training_sampling(self, dist: Categorical):
        if self.rng.uniform() < self.epsilon:
            return torch.randint(low=0, high=self.act_space_size, size=(dist.probs.size(0),))
        return dist.sample()

    def loss(self, trajectory: "TensorTrajectory"):
        action_dist, state_values = self.act(trajectory.obs)
        e = self.clip_eps
        a = trajectory.actions
        r = action_dist.probs[:, a] / trajectory.probs[:, a]
        advantage = trajectory.returns - state_values

        l_clip = torch.minimum(
            r * advantage,
            torch.clip(r, 1 - e, 1 + e) * advantage,
        )
        l_vf = mse_loss(state_values, trajectory.returns, reduction='none') * 0.5
        l_s = action_dist.entropy() * 0.01
        l_clip_s = - (l_clip - l_vf + l_s).sum()
        return l_clip_s


if __name__ == "__main__":
    train_eval(PPOAgent(4), "ppo.pt")