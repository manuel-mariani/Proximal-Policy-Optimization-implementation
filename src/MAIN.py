import warnings

import torch

from agents.ppo import PPOAgent
from agents.reinforce import ReinforceAgent
from environment import CoinRunEnv
from logger import Logger, WandbLogger
from trainer import train
from utils import render_trajectory

# ======================================================================
#                               CONSTANTS
# ======================================================================

AGENT = "ppo"  # "ppo" or "reinforce"
TRAIN = True  # True runs training, False runs loaded model + trajectory rendering
LOGGING = True  # Enables Wandb logging (must have an account)
AUTOSAVE_MODEL = True
MODEL_LOAD_PATH = "../trained_models/PPOAgent-0623-1635.pt"

# Discounted returns & advantage parameters
GAMMA = 0.9
LAMBDA = 0.95

# Training/validation parameters
N_EPISODES = 100
BATCH_SIZE = 2048
LR = 1e-4
N_PARALLEL_TRAIN = 4  # Number of parallel agents to run training steps on
N_PARALLEL_VALID = 2  # Number of parallel agents to run validation steps on
BUFFER_SIZE = 2048  # Maximum number of steps, per parallel agent
EPOCHS_PER_EPISODE = 4  # No. of backward passes per training episode
EPSILON = 0.0  # Eps-greedy used in training
VAL_EPSILON = 1.0  # Eps-greedy used in validation (to avoid looping / stuck states). 0 -> deterministic policy.

# PPO Specific
PPO_CLIP_EPS = 0.2  # PPO clipping epsilon


# ======================================================================


def main():
    # Set agent
    if AGENT.lower() == "ppo":
        agent = PPOAgent(4, epsilon=EPSILON, val_epsilon=VAL_EPSILON, clip_eps=PPO_CLIP_EPS)
    elif AGENT.lower() == "reinforce":
        agent = ReinforceAgent(4, epsilon=EPSILON, val_epsilon=VAL_EPSILON)
    else:
        raise Exception(f"Agent {AGENT} not found. Valid options: 'PPO', 'REINFORCE'")

    # Logging
    logger = Logger()
    if LOGGING and TRAIN:
        logger = WandbLogger(AGENT, train_kwargs)

    # Training
    if TRAIN:
        train(agent, logger, **train_kwargs)
        save = True if AUTOSAVE_MODEL else input("Save model? [Y/n]").strip() != "n"
        if save:
            agent.save()
    # Evaluation
    else:
        agent.load(MODEL_LOAD_PATH)
        agent.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        venv = CoinRunEnv(N_PARALLEL_VALID, seed=123)
        episodes = venv(agent, device, n_steps=BUFFER_SIZE, use_tqdm=True)
        render_trajectory(episodes)


train_kwargs = dict(
    n_episodes=N_EPISODES,
    n_parallel=N_PARALLEL_TRAIN,
    validation_n_parallel=N_PARALLEL_VALID,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    epochs_per_episode=EPOCHS_PER_EPISODE,
    lr=LR,
    gamma=GAMMA,
    _lambda=LAMBDA,
)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
