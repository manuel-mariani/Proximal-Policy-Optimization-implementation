import torch

from agents.ppo import PPOAgent
from agents.reinforce import ReinforceAgent
from environment import CoinRunEnv
from trainer import train, validate

# ======================================================================
#                               CONSTANTS
# ======================================================================
from utils import render_trajectory

AGENT = "ppo"  # "ppo" or "reinforce"
TRAIN = True  # If to run training or just use trained model to display some output. TODO
LOGGING = True
AUTOSAVE_MODEL = True
MODEL_LOAD_PATH = "../trained_models/PPOAgent-0606-1639.pt"

# Discounted returns parameters
GAMMA = 0.99
LAMBDA = 0.95

# Training/validation parameters
N_EPISODES = 1
BATCH_SIZE = 256
LR = 3e-4
N_PARALLEL_TRAIN = 8  # Number of parallel agents to run training steps on
N_PARALLEL_VALID = 4  # Number of parallel agents to run validation steps on
BUFFER_SIZE = 5000  # Maximum number of steps, per parallel agent
EPOCHS_PER_EPISODE = 2  # No. of backward passes per training episode
EPSILON = 0.05  # Eps-greedy

# PPO Specific
PPO_CLIP_EPS = 0.25  # PPO clipping epsilon


# ======================================================================


def main():
    # Set agent
    if AGENT.lower() == "ppo":
        agent = PPOAgent(4, epsilon=EPSILON, clip_eps=PPO_CLIP_EPS)
    elif AGENT.lower() == "reinforce":
        agent = ReinforceAgent(4)
    else:
        raise Exception(f"Agent {AGENT} not found. Valid options: 'PPO', 'REINFORCE'")

    # Training
    if TRAIN:
        train(agent, **train_kwargs)
        save = True if AUTOSAVE_MODEL else input("Save model? [Y/n]").strip() != "n"
        if save:
            agent.save()
    # Evaluation
    else:
        agent.load(MODEL_LOAD_PATH)
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
    main()
