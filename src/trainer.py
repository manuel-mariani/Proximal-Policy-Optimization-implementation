import torch
from tqdm.auto import trange
from welford import Welford

from agents.agent import TrainableAgent
from environment import CoinRunEnv
from logger import Logger
from rewards import action_metrics, discount_returns, gae, reward_metrics, welford_standardizer, win_metrics
from utils import set_seeds


def train(
    agent: TrainableAgent,
    logger: Logger,
    device,
    n_steps=100,
    n_parallel=8,
    validation_n_parallel=4,
    buffer_size=5000,
    batch_size=256,
    epochs_per_episode=2,
    lr=3e-4,
    gamma=0.99,
    _lambda=0.95,
    max_ep_len=1000,
):
    # Initialize the environment
    set_seeds()
    venv = CoinRunEnv(n_parallel, device)
    state = venv.callmethod("get_state")
    val_venv = CoinRunEnv(validation_n_parallel, device, seed=0)

    # Initialize torch stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.compile(device)
    agent.train()
    optimizer = torch.optim.Adam(agent.parameters, lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Initialize running stats for rewards
    rewards_welford = Welford()

    for step in trange(n_steps, colour="green", desc="Training"):
        # Generate the episodes
        # if n_episode % 10 == 0:
        #     state = venv.callmethod("get_state")
        venv.callmethod("set_state", state)
        episodes = venv(agent, n_steps=buffer_size, use_tqdm=True)
        validate(agent, val_venv, device, buffer_size, logger=logger, max_ep_len=max_ep_len)

        # Log episode metrics
        logger.log(**win_metrics(episodes, max_ep_len=max_ep_len))
        logger.log(**action_metrics(episodes))

        # Standardize rewards & compute rewards and advantages
        episodes.rewards = welford_standardizer(episodes.rewards, rewards_welford)
        episodes.returns = discount_returns(episodes, gamma)
        episodes.advantages = gae(episodes, gamma, _lambda)

        # Backward steps (multiple times per replay buffer)
        ep_tensor = episodes.tensor(flatten=True)
        for _ in trange(epochs_per_episode, leave=False, colour="yellow", desc="Backprop"):
            batches = ep_tensor.prioritized_sampling(alpha=0.5, eps=0.01).batch(batch_size)
            # batches = ep_tensor.shuffle().batch(batch_size)
            for batch in batches:
                loss = agent.loss(batch.to(device), logger)
                logger.append(loss=loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters, 1)
                optimizer.step()
            scheduler.step()

        # Logging
        logger.log(
            commit=True,
            episode=step,
            lr=scheduler.get_last_lr(),
            **reward_metrics(episodes),
        )

    logger.finish()
    return agent


def validate(agent, venv, device, buffer_size, logger, max_ep_len):
    """Evaluate an agent on a fixed environment"""
    agent.eval()
    state = venv.callmethod("get_state")
    episodes = venv(agent, n_steps=buffer_size, use_tqdm=True)
    logger.log(**win_metrics(episodes, key_prefix="val_", max_ep_len=max_ep_len))
    logger.log(**action_metrics(episodes, key_prefix="val_"))

    venv.callmethod("set_state", state)
    agent.train()
