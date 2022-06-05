import gym3
import numpy as np
import torch
from tqdm.auto import trange

from agents.agent import TrainableAgent
from environment import CoinRunEnv
from replay_buffer import ReplayBuffer
from utils import discount, render_trajectory, reward_shaping, set_seeds, standardize


def train(
        agent: TrainableAgent,
        env_name,
        n_episodes=100,
        n_parallel=32,
        buffer_size=1000,
        batch_size=512,
        epochs_per_episode=1,
):
    # Initialize the environment
    set_seeds()
    venv = CoinRunEnv(n_parallel)
    val_venv = CoinRunEnv(n_parallel)
    replay_buffer = ReplayBuffer(buffer_size)

    # Initialize torch stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(agent.parameters, lr=3e-3)
    agent.compile(device)
    agent.train()

    for episode in range(n_episodes):
        # Generate the episodes
        agent.reset()
        episodes = replay_buffer.generate_vectorized(venv, agent, device, progress_bar=True).episodic()
        # render_trajectory(episodes)
        # validate(agent, val_venv, device, buffer_size)

        # Shape, Discount and Standardize the rewards
        episodes.rewards = reward_shaping(episodes.rewards, timeout=buffer_size - 25)
        # (logging)
        _tot_shaped_rewards = torch.cat(episodes.rewards)
        reward_sum = _tot_shaped_rewards.sum().item()
        n_wins = (_tot_shaped_rewards > 0).sum().item()
        n_losses = len(episodes.rewards) - n_wins

        episodes.rewards = discount(episodes.rewards, gamma=0.9999)
        episodes.rewards = standardize(episodes.rewards, eps=0.01)

        # Do a training step
        losses = []
        # l_n = 0
        # sum_loss = 0
        ep_tensor = episodes.tensor()
        for _ in range(epochs_per_episode):
            batches = ep_tensor.shuffle().batch(batch_size)
            for batch in batches:
                loss = agent.loss(batch.to(device))
                # sum_loss = sum_loss + loss
                # l_n += 1
                # n_losses += 1
                losses.append(loss.item())
                loss.backward()
                # loss = sum_loss / l_n
                torch.nn.utils.clip_grad_norm_(agent.parameters, 1)
                optimizer.step()
                optimizer.zero_grad()

        print()
        print("Reward sum", reward_sum)
        print("N wins", n_wins)
        print("N losses", n_losses)
        print("Loss", np.mean(losses))
        print("Episode", episode)
    return agent


def validate(agent, venv: gym3.Env, device, buffer_size):
        agent.reset().eval()
        state = venv.callmethod("get_state")
        replay_buffer = ReplayBuffer(buffer_size)
        episodes = replay_buffer.generate_vectorized(venv, agent, device, progress_bar=False).episodic()
        episodes.rewards = reward_shaping(episodes.rewards, timeout=buffer_size - 25)
        _tot_shaped_rewards = torch.cat(episodes.rewards)
        reward_sum = _tot_shaped_rewards.sum().item()
        n_wins = (_tot_shaped_rewards > 0).sum().item()
        n_losses = len(episodes.rewards) - n_wins

        print("VAL: Reward sum", reward_sum)
        print("VAL: N wins", n_wins)
        print("VAL: N losses", n_losses)

        venv.callmethod("set_state", state)
        agent.reset().train()


def train_eval(agent: TrainableAgent, save_path, env_name="coinrun"):
    agent = train(agent, env_name=env_name)
    input("Waiting for input")
    agent.save(save_path)
