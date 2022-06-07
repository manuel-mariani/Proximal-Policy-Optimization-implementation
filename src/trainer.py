import numpy as np
import torch
from tqdm.auto import trange

from agents.agent import TrainableAgent
from environment import CoinRunEnv
from logger import Logger
from rewards import episodes_metric, reward_pipeline
from utils import set_seeds


def train(
    agent: TrainableAgent,
    logger: Logger,
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
    total_epochs = 0
    agent.compile(device)
    agent.train()

    for n_episode in trange(n_episodes, colour='green', desc="Training"):
        # Generate the episodes
        episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=True)
        validate(agent, val_venv, device, buffer_size, logger=logger)

        # Shape, Discount and Standardize the rewards
        reward_pipeline(episodes, gamma=gamma, _lambda=_lambda)

        # Do a training step
        ep_tensor = episodes.tensor()
        losses = []
        for _ in trange(epochs_per_episode, leave=False, colour='yellow', desc='Backprop'):
            total_epochs += 1
            batches = ep_tensor.shuffle().batch(batch_size)
            for batch in batches:
                loss = agent.loss(batch.to(device))
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters, 5)
                optimizer.step()

        # Logging
        logger.log(
            commit=True,
            episode=n_episode,
            loss=np.mean(losses),
            **episodes_metric(episodes)
        )

    logger.finish()
    return agent


def validate(agent, venv, device, buffer_size, logger=None):
    if logger is None:
        logger = Logger()
    agent.eval()
    state = venv.callmethod("get_state")

    episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=True)
    reward_pipeline(episodes, 0.99, 0.95)  # Parameters don't matter since we are not training
    logger.log(**episodes_metric(episodes, key_prefix="val_"))

    venv.callmethod("set_state", state)
    agent.train()
