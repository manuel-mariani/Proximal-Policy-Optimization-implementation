import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.replay_buffer import ReplayBuffer


def render_buffer(replay_buffer: ReplayBuffer, fps=60):
    obs_list = replay_buffer.curr_state_buffer
    it = iter(obs_list)
    fig = plt.figure()
    im = plt.imshow(next(it))

    def update(i):
        im.set_array(next(it))
        return [im]

    ani = FuncAnimation(fig, update, blit=True, frames=len(obs_list), interval=1000 // fps, repeat=False)
    plt.show()
    return ani


def test_net(net: torch.nn.Module, input_size, output_size=None):
    x = torch.rand(*input_size)
    out = net(x)
    print("Output_size:", out.size(), out.numel())
    if output_size:
        assert tuple(out.size()) == output_size
    print("✅ Passed")
