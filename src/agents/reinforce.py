import torch
from torch import autocast, nn
from torch.cuda.amp import GradScaler
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
        # with autocast("cpu" if obs.device == "cpu" else "cuda"):
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


def train(n_episodes=100, n_parallel=128, buffer_size=200):
    print("Initializing")
    venv = generate_vec_environment(n_parallel, "bossrush")
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

        # Take the *episodes* rewards and actions probabilities (log)
        ep_rewards = episode_dict['rewards']
        ep_log_probs = episode_dict['log_probs']

        # Compute the discounted rewards
        discounted_rewards = []
        gamma = 0.9999

        for er in ep_rewards:
            er = torch.flip(er, (0,))
            dr = [er[0]]
            for r in er[1:]:
                dr.append(r + gamma * dr[-1])
            dr = torch.flip(torch.tensor(dr), (0,))
            discounted_rewards.append(dr)

        # Compute the loss
        losses = []
        for episode in range(len(ep_rewards)):
            disc = discounted_rewards[episode]
            logs = ep_log_probs[episode]

            loss = - torch.sum(logs * disc)
            losses.append(loss)
        loss = torch.sum(torch.stack(losses)).to(device)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        mean_reward = torch.mean(replay_buffer.rewards).item()
        print()
        print("Mean reward", mean_reward)
        print("Loss", loss.item())
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
    # eval(agent)
