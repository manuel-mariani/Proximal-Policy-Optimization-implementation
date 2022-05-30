import torch
from torch import autocast, nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from agents.agent import Agent
from models.action_net import ActionNet
from models.feature_extractor import FeatureExtractor
from replay_buffer import ReplayBuffer, VectorizedReplayBuffer
from utils import generate_environment, generate_vec_environment, obs_to_tensor


# ======================================================================
class ReinforceAgent(Agent):
    def __init__(self, obs_space_shape: int, act_space_size: int):
        super().__init__(obs_space_shape, act_space_size)
        self.is_training = True

        # Initialize the network
        n_features = 256
        self.feature_extractor = FeatureExtractor(n_features)
        self.action_net = ActionNet(n_features, act_space_size)
        self.model = nn.Sequential(self.feature_extractor, self.action_net)

    def act(self, obs: torch.Tensor):
        with autocast("cpu" if obs.device == "cpu" else "cuda"):
            dist = self.model(obs)
        return Categorical(dist)

    def reset(self):
        self.action_net.reset()

    def train(self):
        self.is_training = True
        self.model.train()

    def eval(self):
        self.is_training = False
        self.model.eval()


# ======================================================================


def train(n_episodes=3, n_parallel=64, buffer_size=1000):
    print("Initializing")
    venv = generate_vec_environment(n_parallel, "coinrun")
    replay_buffer = VectorizedReplayBuffer(buffer_size)
    agent = ReinforceAgent(-1, 15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(params=agent.model.parameters(), lr=2e-3)
    agent.model.to(device)
    agent.train()

    for episode in range(n_episodes):
        agent.reset()
        replay_buffer(venv, agent, device, progress_bar=True)


def _train(n_episodes=3):
    print("Initializing")
    env = generate_environment()
    replay_buffer = ReplayBuffer(buffer_size=1000, n_episodes=1)
    agent = ReinforceAgent(env.observation_space, env.action_space)
    history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    optimizer = torch.optim.Adam(params=agent.model.parameters(), lr=2e-3)
    agent.model.to(device)
    agent.train()

    for episode in tqdm(range(n_episodes)):
        agent.reset()
        obs = env.reset()

        # Generate the episode
        actions = []
        rewards = []

        while True:
            obs = obs_to_tensor(obs).to(device)
            action = agent.act(obs)
            chosen_action = action.sample()
            next_obs, reward, done, _ = env.step(chosen_action.detach().item())
            # chosen_actions.append(action)
            # rewards.append(reward)
            obs = next_obs

            loss = -action.log_prob(chosen_action) * reward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if done:
                break

        # Compute the discounted returns

        # rb = replay_buffer(env, agent).as_tensors()

        # Calculate G
        # rew = rb['reward_buffer']
        # G =
    return agent


def eval(agent, n_episodes=10):
    agent.eval()
    env = generate_environment(render="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    with torch.no_grad():
        for _ in tqdm(range(n_episodes)):
            agent.reset()
            obs = env.reset()
            while True:
                obs = obs_to_tensor(obs).to(device)
                action = agent.act(obs)
                chosen_action = action.item()
                obs, reward, done, _ = env.step(chosen_action)
                if done:
                    break


if __name__ == "__main__":
    agent = train()
    eval(agent)
