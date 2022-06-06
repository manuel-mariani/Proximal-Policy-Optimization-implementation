import numpy as np
import torch

from agents.agent import TrainableAgent
from environment import CoinRunEnv
from utils import discount, gae, reward_shaping, set_seeds, standardize


def train(
        agent: TrainableAgent,
        n_episodes=100,
        n_parallel=32,
        validation_n_parallel=64,
        buffer_size=1000,
        batch_size=256,
        epochs_per_episode=10,
):
    # Initialize the environment
    set_seeds()
    venv = CoinRunEnv(n_parallel)
    val_venv = CoinRunEnv(validation_n_parallel)

    # Initialize torch stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.RAdam(agent.parameters, lr=3e-3)
    agent.compile(device)
    agent.train()

    for episode in range(n_episodes):
        # Generate the episodes
        episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=True)
        # render_trajectory(episodes)
        validate(agent, val_venv, device, buffer_size)

        # Shape, Discount and Standardize the rewards
        # episodes.rewards = reward_shaping(episodes.rewards, timeout=buffer_size - 25)
        # (logging)
        _tot_shaped_rewards = torch.cat(episodes.rewards)
        reward_sum = _tot_shaped_rewards.sum().item()
        n_wins = (_tot_shaped_rewards > 0).sum().item()
        n_losses = len(episodes.rewards) - n_wins

        episodes.advantages = gae(episodes, gamma=0.99, _lambda=0.95)
        episodes.advantages = standardize(episodes.advantages)
        episodes.returns = discount(episodes.rewards, gamma=0.99)
        episodes.returns = standardize(episodes.returns)

        print()
        print("Episode", episode)
        print("Reward sum", reward_sum)
        print("N wins", n_wins)
        print("N losses", n_losses)

        # Do a training step
        ep_tensor = episodes.tensor()
        for _ in range(epochs_per_episode):
            losses = []
            batches = ep_tensor.shuffle().batch(batch_size)
            for batch in batches:
                loss = agent.loss(batch.to(device))
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters, 1)
                optimizer.step()
            print("Loss", np.mean(losses))

    return agent


def validate(agent, venv, device, buffer_size):
        agent.eval()
        state = venv.callmethod("get_state")
        episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=False)
        episodes.rewards = reward_shaping(episodes.rewards, timeout=buffer_size - 25)
        _tot_shaped_rewards = torch.cat(episodes.rewards)
        reward_sum = _tot_shaped_rewards.sum().item()
        n_wins = (_tot_shaped_rewards > 0).sum().item()
        n_losses = len(episodes.rewards) - n_wins

        print("VAL: Reward sum", reward_sum)
        print("VAL: N wins", n_wins)
        print("VAL: N losses", n_losses)

        venv.callmethod("set_state", state)
        agent.train()


def train_eval(agent: TrainableAgent, save_path):
    agent = train(agent)
    input("Waiting for input")
    agent.save(save_path)
