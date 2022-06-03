import gym
import gym3
import numpy as np
import procgen
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def generate_environment(env_name="coinrun", render=None) -> gym.Env:
    return gym.make(
        f"procgen:procgen-{env_name}-v0",
        render=render,
        start_level=0,
        num_levels=0,
        distribution_mode="hard",  # "easy", "hard", "extreme", "memory", "exploration"
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=False,
    )


def generate_vec_environment(n_parallel, env_name="coinrun") -> gym3.Env:
    return procgen.ProcgenGym3Env(
        num=n_parallel,
        env_name=env_name,
        start_level=0,
        num_levels=0,
        distribution_mode="hard",  # "easy", "hard", "extreme", "memory", "exploration"
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=False,
    )


def render_buffer(replay_buffer, fps=60):
    """Renders a ReplayBuffer using matplotlib"""
    if isinstance(replay_buffer, (list, np.array, torch.Tensor)):
        obs_list = replay_buffer
    else:
        obs_list = replay_buffer.curr_state_buffer
    it = iter(obs_list)
    fig = plt.figure()
    im = plt.imshow(next(it))

    def update(_):
        im.set_array(next(it))
        return [im]

    ani = FuncAnimation(fig, update, blit=True, frames=len(obs_list), interval=1000 // fps, repeat=False)
    plt.show()
    return ani


def test_net(net: torch.nn.Module, input_size, output_size=None):
    """Simple test on a torch network, used to see if the output is congruent with assumption"""
    x = torch.rand(*input_size)
    out = net(x)
    print("Output_size:", out.size(), out.numel())
    if output_size:
        assert tuple(out.size()) == output_size
    print("✅ Passed")


def obs_to_tensor(obs, dtype=None):
    if isinstance(obs, list):
        return torch.stack([obs_to_tensor(x) for x in obs])
    x = torch.tensor(obs, dtype=dtype)
    x = torch.permute(x, (2, 0, 1))
    x = x / 255
    if x.ndim < 4:
        x = torch.unsqueeze(x, 0)
    return x


def onehot(val, size):
    oh = torch.zeros(size)
    oh[:, val] = 1
    return oh


def reward_shaping(ep_rewards, timeout: int, kappa=-1):
    shaped = []
    for er in ep_rewards:
        # Penalization if the episode goes on for too long, without a reward
        if len(er) > timeout:
            # er[timeout:] = kappa / (len(er) - timeout)
            er[-1] = kappa
        # Penalization if the episode ends without a reward
        elif er[-1] == 0:
            er[-1] = kappa
        shaped.append(er)
    return shaped


def discount(ep_rewards, gamma):
    discounted = []
    for er in ep_rewards:
        flip_er = torch.flip(er, (0,))
        dr = [flip_er[0]]
        for r in flip_er[1:]:
            dr.append(r + gamma * dr[-1])
        dr = torch.flip(torch.tensor(dr), (0,))
        discounted.append(dr)
    return discounted


def standardize(ep_rewards, eps=0.01):
    standardized = []
    tensor = torch.cat(ep_rewards)
    mean = tensor.mean()
    std = tensor.std() + eps
    for er in ep_rewards:
        standardized.append((er - mean) / std)
    return standardized
