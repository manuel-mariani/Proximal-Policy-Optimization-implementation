# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import gym
import numpy as np
import procgen

procgen.env

# + pycharm={"name": "#%%\n"}
env: gym.Env = gym.make(
    "procgen:procgen-coinrun-v0",
    # render="human",
    start_level=0,
    distribution_mode="hard",  # "easy", "hard", "extreme", "memory", "exploration"
    restrict_themes=True,
    use_backgrounds=False,
    use_monochrome_assets=False,
)

# Iterate until the goal is reached
obs = env.reset()
observations = [obs]
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    observations.append(obs)
    if done:
        break
print(len(observations), observations[0].shape)

# + pycharm={"name": "#%%\n"}
env.observation_space.shape

# + pycharm={"name": "#%%\n"}
env.action_space

# + pycharm={"name": "#%%\n"}
# %matplotlib
from utils import render_buffer
from tqdm.auto import trange

n_parallel = 256
venv = procgen.ProcgenGym3Env(
    num=n_parallel,
    env_name="coinrun",
    start_level=0,
    num_levels=0,
    distribution_mode="hard",  # "easy", "hard", "extreme", "memory", "exploration"
    restrict_themes=True,
    use_backgrounds=False,
    use_monochrome_assets=False,
)
n_actions = 15
rng = np.random.default_rng()
observations = []
rewards = []
for _ in trange(1024):
    rew, obs, first = venv.observe()
    obs = obs["rgb"]
    actions = rng.integers(0, n_actions, size=(n_parallel))
    venv.act(actions)
    observations.append(obs[0])
    rewards.append(rew)

ani = render_buffer(observations)
rewards = np.array(rewards)

# + pycharm={"name": "#%%\n"}
venv.ac_space.eltype.n

# + pycharm={"name": "#%%\n"}
venv.ob_space

# + pycharm={"name": "#%%\n"}
from torch.distributions import Categorical
import torch

dist = torch.rand((n_parallel, n_actions))
dist = Categorical(dist)
dist.sample().numpy()
