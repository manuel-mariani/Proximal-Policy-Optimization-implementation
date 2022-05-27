import numpy as np
from gym.spaces import Discrete, Box


class Agent:
    def __init__(self, obs_space: Box, act_space: Discrete, train: bool, eps=0.1):
        """
        :param train:
        :param obs_space:
        :param act_space:
        :param eps: Epsilon for epsilon-greedy policy
        """
        self.obs_space = obs_space
        self.act_space = act_space
        self.train = train
        self.eps = eps
        self.rng = np.random.default_rng()
        pass

    def act(self, obs):
        if self.train:
            return self._training_policy(obs)
        else:
            return self._target_policy(obs)

    def _target_policy(self, obs):
        # TODO: implement
        return 0

    def _training_policy(self, obs):
        # eps-greedy
        if self.rng.uniform() < self.eps:
            return self.act_space.sample()
        return self._target_policy(obs)
