from agents.ppo import PPOAgent
from agents.reinforce import ReinforceAgent
from trainer import train

# ======================================================================
#                               CONSTANTS
# ======================================================================

AGENT = "ppo"  # "ppo" or "reinforce"
TRAIN = True  # If to run training or just use trained model to display some output. TODO
AUTOSAVE_MODEL = True
LOGGING = True

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

    # Check training
    if TRAIN:
        train(agent, **train_kwargs)
        save = True if AUTOSAVE_MODEL else input("Save model? [Y/n]").strip() != "n"
        if save:
            agent.save()
    else:
        raise NotImplementedError


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
