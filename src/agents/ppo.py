import numpy as np
import torch
from torch import autocast, nn
from torch.distributions import Categorical
from torchinfo import summary
from tqdm.auto import tqdm

from agents.agent import Agent
from models.action_net import ActionNet
from models.feature_extractor import FeatureExtractor
from replay_buffer import ReplayBuffer
from utils import discount, generate_environment, generate_vec_environment, obs_to_tensor, onehot


class PPONet(nn.Module):
    def __init__(self, n_features, downsampling, n_actions):
        super().__init__()
        self.feature_extractor = FeatureExtractor(n_features, downsampling)
        self.action_net = ActionNet(n_features, n_actions)
        self.value_net = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(n_features // 2, n_features // 4),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(n_features // 4, 1),
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        actions = self.action_net(feats)
        values = self.action_net(feats)
        return actions, values


class PPOAgent(Agent):
    def __init__(self, act_space_size, eps=0.2):
        super().__init__(act_space_size)
        self.is_training = True
        self.eps = eps
        self.model = PPONet(256, 1, act_space_size)

    def act(self, obs: torch.Tensor, add_rand=True):
        # Forward
        with autocast("cpu" if obs.device == "cpu" else "cuda"):
            actions, values = self.model(obs)
        # If evaluating, return just the argmax (one-hotted)
        if not self.is_training:
            actions = onehot(torch.argmax(actions, dim=-1), actions.size())
            return Categorical(actions, validate_args=False)
        # If eps-greedy, return random onehot distribution
        if add_rand and self.rng.uniform() < self.eps:
            actions = onehot(self.rng.integers(0, self.act_space_size), actions.size())
        return Categorical(actions), values

    def compile(self, device, sample_input=torch.rand(32, 3, 64, 64)):
        if self.model == torch.jit.ScriptModule:
            return
        self.act(sample_input.to(device))
        self.model = torch.jit.script(self.model)
        print(
            summary(
                self.model,
                input_size=sample_input.size(),
                mode="train",
                depth=7,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                ],
            )
        )

    def save(self, path):
        self.model.save(path)

    def train(self):
        self.is_training = True
        self.model.train()

    def eval(self):
        self.is_training = False
        self.model.eval()

    @staticmethod
    def load(self, act_space_size: int, path):
        agent = PPOAgent(act_space_size)
        agent.model = torch.jit.load(path)


# ======================================================================
def train(n_episodes=50, n_parallel=32, buffer_size=1000, batch_size=32):
    print("Initializing")
    venv = generate_vec_environment(n_parallel)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = PPOAgent(15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.RMSprop(params=agent.model.parameters(), lr=5e-4)
    # scaler = GradScaler()
    agent.model.to(device)
    agent.compile(device)
    agent.train()

    for episode in range(n_episodes):
        # Generate the episodes
        agent.reset()
        episodes = replay_buffer.generate_vectorized(venv, agent, device, progress_bar=True).episodic()

        # Compute the discounted rewards
        mean_reward = torch.mean(torch.cat(episodes.rewards)).item()
        episodes.rewards = discount(episodes.rewards, gamma=0.99)
        # Using the replay buffer, compute gradients and do backwards passes

        losses = []
        batches = episodes.tensor().batch(batch_size)
        for obs, action, reward, _, values in batches:
            obs = obs.to(device)
            act = action.to(device)
            rew = reward.to(device)

            dist = agent.act(obs, add_rand=False)
            log_prob = dist.log_prob(act)
            loss = -torch.sum(log_prob * rew)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_discounted = torch.mean(torch.cat(batches.rewards)).item()
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
    # compiled_model = torch.jit.script(agent.model)
    # compiled_model.save("reinforce.pt")
    agent.save("reinforce.pt")
    input("Waiting for input")
    eval(agent)