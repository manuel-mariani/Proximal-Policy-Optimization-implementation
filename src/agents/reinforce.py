import numpy as np
import torch
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.distributions import Categorical
from torchinfo import summary
from tqdm.auto import tqdm

from agents.agent import Agent
from models.action_net import ActionNet
from models.feature_extractor import FeatureExtractor
from replay_buffer import ReplayBuffer
from utils import discount, generate_environment, generate_vec_environment, obs_to_tensor, onehot


# ======================================================================
class ReinforceAgent(Agent):
    def __init__(self, act_space_size: int, eps=0.2):
        super().__init__(act_space_size)
        self.is_training = True
        self.eps = eps

        # Initialize the network
        n_features = 256
        self.feature_extractor = FeatureExtractor(n_features, downsampling=2)
        self.action_net = ActionNet(n_features, act_space_size)
        self.model = nn.Sequential(self.feature_extractor, self.action_net)

    def act(self, obs: torch.Tensor, add_rand=True):
        with autocast("cpu" if obs.device == "cpu" else "cuda"):
            dist = self.model(obs)
        if not self.is_training:
            dist = onehot(torch.argmax(dist, dim=-1), dist.size())
        if add_rand and self.rng.uniform() < self.eps:
            dist = onehot(self.rng.integers(0, self.act_space_size), dist.size())
        return Categorical(dist, validate_args=False)

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

    def reset(self):
        self.action_net.reset()

    def train(self):
        self.is_training = True
        self.model.train()

    def eval(self):
        self.is_training = False
        self.model.eval()

    @staticmethod
    def load(self, act_space_size: int, path):
        agent = ReinforceAgent(act_space_size)
        agent.model = torch.jit.load(path)


# ======================================================================
def train(n_episodes=50, n_parallel=32, buffer_size=1000, batch_size=32):
    print("Initializing")
    venv = generate_vec_environment(n_parallel, "coinrush")
    replay_buffer = ReplayBuffer(buffer_size)
    agent = ReinforceAgent(15)

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
        for obs, action, reward, _ in batches:
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
    env = generate_environment("coinrush", render="human")
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
