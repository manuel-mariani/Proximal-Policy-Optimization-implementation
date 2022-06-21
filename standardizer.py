import torch

from trajectories import TensorTrajectory


class Standardizer:
    def __init__(self, clip: float = 0, mean_shift: bool = False, std_scale: bool = True):
        self.mean = torch.zeros(0)
        self.std = torch.ones(1)
        self.clip = clip
        self.mean_shift = mean_shift
        self.std_scale = std_scale
        self.n = 0

    def __call__(self, tensor: torch.Tensor):
        t = tensor
        if self.mean_shift:
            # self.mean = (self.mean + tensor.mean()) / 2
            self.mean = tensor.mean()
            t = t - self.mean
        if self.std_scale:
            # self.std = (self.std + tensor.std()) / 2
            self.std = tensor.std()
            t = t / (self.std + 1e-8)
        if self.clip > 0:
            t = t.clip(-self.clip, self.clip)
        return t


class TrajectoryStandardizer:
    def __init__(self):
        self.reward_standardizer = Standardizer(clip=10, mean_shift=False)
        self.returns_standardizer = Standardizer(clip=10, mean_shift=False)
        self.advantage_standardizer = Standardizer(clip=10, mean_shift=False)
        # self.obs_standardizer = Standardizer(clip=0, mean_shift=True)

    def __call__(self, trajectory: TensorTrajectory):
        trajectory.rewards = self.reward_standardizer(trajectory.rewards)
        trajectory.returns = self.returns_standardizer(trajectory.returns)
        trajectory.advantages = self.advantage_standardizer(trajectory.advantages)
        # trajectory.obs = self.obs_standardizer(trajectory.obs)
        return trajectory
