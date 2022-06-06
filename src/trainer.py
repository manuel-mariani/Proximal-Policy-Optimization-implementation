import numpy as np
import torch
from tqdm.auto import trange

from agents.agent import TrainableAgent
from environment import CoinRunEnv
from rewards import reward_pipeline
from utils import set_seeds


def train(
        agent: TrainableAgent,
        n_episodes=100,
        n_parallel=8,
        validation_n_parallel=4,
        buffer_size=5000,
        batch_size=256,
        epochs_per_episode=2,
        lr=3e-4,
        gamma=0.99,
        _lambda=0.95,
):
    # Initialize the environment
    set_seeds()
    venv = CoinRunEnv(n_parallel)
    val_venv = CoinRunEnv(validation_n_parallel)

    # Initialize torch stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(agent.parameters, lr=lr)
    agent.compile(device)
    agent.train()

    for episode in trange(n_episodes):
        # Generate the episodes
        episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=True)
        validate(agent, val_venv, device, buffer_size)

        # Shape, Discount and Standardize the rewards
        n_wins = (torch.cat(episodes.rewards) > 0).sum().item()
        n_losses = len(episodes.rewards) - n_wins

        reward_pipeline(episodes, gamma=gamma, _lambda=_lambda)
        # (logging)
        reward_sum = torch.cat(episodes.rewards).sum().item()
        returns_sum = torch.cat(episodes.returns).sum().item()

        print()
        print("Episode", episode)
        print("Reward sum", reward_sum)
        print("Returns sum", returns_sum)
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
                torch.nn.utils.clip_grad_norm_(agent.parameters, 5)
                optimizer.step()
            print("Loss", np.mean(losses))

    return agent


def validate(agent, venv, device, buffer_size):
        agent.eval()
        state = venv.callmethod("get_state")
        episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=False)
        _tot_shaped_rewards = torch.cat(episodes.rewards)
        reward_sum = _tot_shaped_rewards.sum().item()
        n_wins = (_tot_shaped_rewards > 0).sum().item()
        n_losses = len(episodes.rewards) - n_wins

        print("VAL: Reward sum", reward_sum)
        print("VAL: N wins", n_wins)
        print("VAL: N losses", n_losses)

        venv.callmethod("set_state", state)
        agent.train()
