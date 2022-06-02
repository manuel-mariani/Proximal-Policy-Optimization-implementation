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
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
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
from tqdm.auto import tqdm, trange

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
# Episode X Step
import torch

is_first = torch.tensor(
    [
        [1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
    ]
).bool()
rewards = torch.tensor(
    [
        [0, 0, 0, 0, 20, 0, 0, 0, 10],
        [0, 0, 0, 0, 0, 10, 0, 0, 20],
        [0, 0, 0, 0, 0, 0, 0, 1, 30],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
    ]
)

# + pycharm={"name": "#%%\n"}
t = is_first.clone().float()
t[:, 0] = 1
# t = torch.cat((t, torch.ones(t.size(0)).unsqueeze(-1)), dim=1)
print(t)
size = t.size(1)
m = torch.argwhere(t)
res = []

prev_i = m[0, 0]
prev_j = m[0, 0]
for i, j in m[1:, :].numpy():
    if i != prev_i:
        res.append(rewards[prev_i, prev_j:size])
    else:
        res.append(rewards[i, prev_j:j])
    prev_i = i
    prev_j = j
    print(i, j)
print(res)

# + pycharm={"name": "#%%\n"}
episodes_rewards = res
discounted_rewards = []
gamma = 0.99

for er in episodes_rewards:
    er = torch.flip(er, (0,))
    dr = [er[0]]
    for r in er[1:]:
        dr.append(r + gamma * dr[-1])
    dr = torch.flip(torch.tensor(dr), (0,))
    discounted_rewards.append(dr)

    # g = torch.ones(er.size()) * gamma
    # g = torch.cumprod(g, dim=-1)
    # print("G", g)
    # m = torch.ones((er.size(0), er.size(0)))
    # m = torch.triu(m)
    # print("m", m)
    # disc = m * er
    # print("disc", disc)
    # print()
    # print()
print(episodes_rewards)
print()
print(discounted_rewards)

# + pycharm={"name": "#%%\n"}
torch.flatten(t, 0, 1)

# + pycharm={"name": "#%%\n"}
a, b, c = {"a": 1, "b": 2, "c": 3}.values()
print(a, b, c)
