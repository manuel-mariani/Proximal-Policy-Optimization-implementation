import numpy as np
import torch
from gym import Env

from src.agent import Agent


class ReplayBuffer:
    def __init__(self, buffer_size, n_episodes=-1):
        self.buffer_size = buffer_size
        self.n_episodes = n_episodes
        self.curr_state_buffer = []
        self.next_state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []

        self.buffers = (
            self.curr_state_buffer,
            self.next_state_buffer,
            self.action_buffer,
            self.reward_buffer,
            self.done_buffer,
        )

    def reset(self):
        for buffer in self.buffers:
            buffer.clear()

    def __len__(self):
        lengths = np.array([len(buffer) for buffer in self.buffers])
        assert np.all(lengths == lengths[0])
        return lengths[0]

    def __call__(self, env: Env, actor: Agent):
        self.reset()
        for curr_obs, action, reward, done, next_obs in self._generate(env, actor):
            self.curr_state_buffer.append(curr_obs)
            self.next_state_buffer.append(next_obs)
            self.action_buffer.append(action)
            self.reward_buffer.append(reward)
            self.done_buffer.append(done)
        return self

    def _generate(self, env: Env, actor: Agent):
        ep = 0
        tot = 0
        while ep != self.n_episodes and tot < self.buffer_size:
            ep += 1
            obs = env.reset()
            while True:
                tot += 1
                action = actor.act(obs)
                next_obs, reward, done, _ = env.step(action)
                yield obs, action, reward, done, next_obs
                obs = next_obs
                if done or tot == self.buffer_size:
                    break

    def as_tensors(self, dtype=None):
        return dict(
            curr_state_buffer=torch.tensor(self.curr_state_buffer, dtype),
            next_state_buffer=torch.tensor(self.next_state_buffer, dtype),
            action_buffer=torch.tensor(self.action_buffer, dtype),
            reward_buffer=torch.tensor(self.reward_buffer, dtype),
            done_buffer=torch.tensor(self.done_buffer, dtype),
        )

    def as_numpy(self, dtype=None):
        return dict(
            curr_state_buffer=np.array(self.curr_state_buffer, dtype),
            next_state_buffer=np.array(self.next_state_buffer, dtype),
            action_buffer=np.array(self.action_buffer, dtype),
            reward_buffer=np.array(self.reward_buffer, dtype),
            done_buffer=np.array(self.done_buffer, dtype),
        )
