import gym3
import numpy as np
import torch
from gym import Env
from torch.distributions import Distribution
from tqdm.auto import trange

from src.agents.agent import Agent
from utils import obs_to_tensor


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
                action_item = action
                if isinstance(action, Distribution):
                    action_item = action.sample().item()

                next_obs, reward, done, _ = env.step(action_item)
                yield obs, action, reward, done, next_obs
                obs = next_obs
                if done or tot == self.buffer_size:
                    break

    def as_tensors(self, dtype=None):
        with torch.no_grad():
            d = dict(
                curr_state_buffer=obs_to_tensor(self.curr_state_buffer, dtype=dtype),
                next_state_buffer=obs_to_tensor(self.next_state_buffer, dtype=dtype),
                reward_buffer=torch.tensor(self.reward_buffer, dtype=dtype),
                done_buffer=torch.tensor(self.done_buffer, dtype=dtype),
            )
            if isinstance(self.action_buffer[0], Distribution):
                d['action_buffer'] = self.action_buffer
            else:
                d['action_buffer'] = torch.tensor(self.action_buffer, dtype=dtype)
        return d


class VectorizedReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.obs = []
        # self.next_obs = []
        self.rewards = []
        self.chosen_actions = []
        self.log_probs = []
        self.first = []

    def reset(self):
        self.obs = []
        # self.next_obs = []
        self.rewards = []
        self.chosen_actions = []
        self.log_probs = []
        self.first = []

    def __call__(self, venv: gym3.Env, agent: Agent, device, progress_bar=False):
        steps = trange(self.buffer_size) if progress_bar else range(progress_bar)
        self.reset()
        for _ in steps:
            rew, obs, first = venv.observe()

            # Convert obs tensor  [N, W, H, C] to [N, C, W, H]
            obs = torch.tensor(obs['rgb'], dtype=torch.float16).detach().to(device)
            obs = torch.permute(obs, (0, 3, 1, 2)) / 255

            # Act
            action_dists = agent.act(obs)
            chosen_actions = action_dists.sample()
            venv.act(chosen_actions.cpu().detach().numpy())
            log_probs = action_dists.log_prob(chosen_actions)

            # Remove tensors from GPU (no effect if using cpu)
            obs = obs.to("cpu")
            log_probs = log_probs.to("cpu")
            chosen_actions = chosen_actions.to("cpu")

            # Number items
            self.rewards.append(torch.from_numpy(rew).detach())
            self.first.append(torch.from_numpy(first).detach().bool())
            self.chosen_actions.append(chosen_actions)
            # Other
            self.obs.append(obs)
            self.log_probs.append(log_probs)

        # To tensor
        self.rewards = torch.stack(self.rewards)
        self.first = torch.stack(self.first)
        self.chosen_actions = torch.stack(self.chosen_actions)
        self.obs = torch.stack(self.obs)
        self.log_probs = torch.stack(self.log_probs)
