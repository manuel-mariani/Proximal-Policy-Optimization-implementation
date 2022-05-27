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
import procgen
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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
type(env)

# + pycharm={"name": "#%%\n"}
env.observation_space.shape

# + pycharm={"name": "#%%\n"}
env.action_space.sample()


# + pycharm={"name": "#%%\n"}
# %matplotlib qt
#


def fast_plot(obs_list):
    it = iter(obs_list)
    fig = plt.figure()
    im = plt.imshow(next(it))

    def update(i):
        im.set_array(next(it))
        return im

    ani = FuncAnimation(fig, update, blit=False, frames=len(obs_list), interval=60)
    plt.show()
    return ani


fast_plot(observations)
