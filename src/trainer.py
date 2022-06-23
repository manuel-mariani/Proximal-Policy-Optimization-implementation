import numpy as np
import torch
import torch_optimizer as optim
from tqdm.auto import trange

from agents.agent import TrainableAgent
from environment import CoinRunEnv
from logger import Logger
from rewards import win_metrics, reward_metrics, reward_pipeline
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
    state = venv.callmethod("get_state")
    val_venv = CoinRunEnv(validation_n_parallel)
    # trajectory_standardizer = TrajectoryStandardizer()

    # Initialize torch stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.compile(device)
    agent.train()
    # optimizer = optim.Ranger(agent.parameters, lr=lr)
    optimizer = torch.optim.Adam(agent.parameters, lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr,
        max_lr=lr * 20,
        step_size_up=epochs_per_episode,
        mode="exp_range",
        gamma=0.99,
        cycle_momentum=False,
    )
    total_epochs = 0

    for n_episode in trange(n_episodes, colour="green", desc="Training"):
        # Generate the episodes
        # if n_episode % 10 == 0:
        #     state = venv.callmethod("get_state")
        venv.callmethod("set_state", state)
        episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=True)
        validate(agent, val_venv, device, buffer_size, logger=logger)


        # Log episode metrics
        logger.log(**win_metrics(episodes))

        # Shape, Discount and Standardize the rewards
        reward_pipeline(episodes, gamma=gamma, _lambda=_lambda)

        # Backward steps (multiple times per replay buffer)
        ep_tensor = episodes.tensor()
        losses = []
        for _ in trange(epochs_per_episode, leave=False, colour="yellow", desc="Backprop"):
            total_epochs += 1
            # batches = ep_tensor.shuffle().batch(batch_size)
            batches = episodes.prioritized_sampling().batch(batch_size)
            for batch in batches:
                loss = agent.loss(batch.to(device), logger)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters, 5)
                optimizer.step()
            scheduler.step()
        # Log loss
        logger.log(
            commit=True,
            episode=n_episode,
            loss=np.mean(losses),
            lr=scheduler.get_last_lr(),
            **reward_metrics(episodes),
        )

    logger.finish()
    return agent


def validate(agent, venv, device, buffer_size, logger):
    agent.eval()
    state = venv.callmethod("get_state")
    episodes = venv(agent, device, n_steps=buffer_size, use_tqdm=True)
    logger.log(**win_metrics(episodes, key_prefix="val_"))

    venv.callmethod("set_state", state)
    agent.train()
