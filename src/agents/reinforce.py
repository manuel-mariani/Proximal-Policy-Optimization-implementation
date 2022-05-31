import numpy as np
import torch
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.distributions import Categorical
from tqdm.auto import tqdm

from agents.agent import Agent
from models.action_net import ActionNet
from models.feature_extractor import FeatureExtractor
from replay_buffer import VectorizedReplayBuffer
from utils import discount, generate_environment, generate_vec_environment, obs_to_tensor


# ======================================================================
class ReinforceAgent(Agent):
    def __init__(self, obs_space_shape: int, act_space_size: int):
        super().__init__(obs_space_shape, act_space_size)
        self.is_training = True

        # Initialize the network
        n_features = 128
        self.feature_extractor = FeatureExtractor(n_features, downsampling=1)
        self.action_net = ActionNet(n_features, act_space_size)
        self.model = nn.Sequential(self.feature_extractor, self.action_net)

    def act(self, obs: torch.Tensor, add_rand=True):
        with autocast("cpu" if obs.device == "cpu" else "cuda"):
            dist = self.model(obs)
        if not self.is_training:
            onehot = torch.zeros(dist.size())
            best_a = torch.argmax(dist, dim=-1)
            onehot[:, best_a] = 1
            return Categorical(onehot)
        if add_rand:
            dist = dist + torch.rand_like(dist).detach() * 0.1
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


def train(n_episodes=30, n_parallel=128, buffer_size=500):
    print("Initializing")
    venv = generate_vec_environment(n_parallel)
    replay_buffer = VectorizedReplayBuffer(buffer_size)
    agent = ReinforceAgent(-1, 15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(params=agent.model.parameters(), lr=5e-3)
    # scaler = GradScaler()
    agent.model.to(device)
    agent.train()

    for episode in range(n_episodes):
        # Generate the episodes
        agent.reset()
        replay_buffer(venv, agent, device, progress_bar=True)
        episode_dict = replay_buffer.to_single_episodes()

        # Take the *episodes* obs, rewards and chosen actions
        ep_chosen_actions = episode_dict["chosen_actions"]
        ep_rewards = episode_dict["rewards"]
        ep_obs = episode_dict["obs"]

        # Compute the discounted rewards
        ep_disc_rewards = discount(ep_rewards, gamma=0.9999)
        losses = []

        # Using the replay buffer, compute gradients and do backwards passes
        for ep in range(len(ep_rewards)):
            obs = ep_obs[ep].to(device)
            ch_a = ep_chosen_actions[ep].to(device)
            drew = ep_disc_rewards[ep].to(device)

            dist = agent.act(obs, add_rand=False)
            log_prob = dist.log_prob(ch_a)
            loss = - torch.sum(log_prob * drew)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_reward = torch.mean(replay_buffer.rewards).item()
        mean_discounted = torch.mean(torch.cat(ep_disc_rewards)).item()
        print()
        print("Mean reward", mean_reward)
        print("Mean discounted reward", mean_discounted)
        print("Loss", np.mean(losses))
        print("Episode", episode)
    return agent


def eval(agent, n_episodes=10):
    agent.eval()
    env = generate_environment(render="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for _ in tqdm(range(n_episodes)):
            agent.reset()
            obs = env.reset()
            while True:
                obs = obs_to_tensor(obs).to(device)
                action = agent.act(obs)
                chosen_action = action.sample().item()
                obs, reward, done, _ = env.step(chosen_action)
                if done:
                    break


if __name__ == "__main__":
    agent = train()
    compiled_model = torch.jit.script(agent.model)
    compiled_model.save("reinforce.pt")
    input("Waiting for input")
    eval(agent)

