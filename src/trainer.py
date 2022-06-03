import numpy as np
import torch
from tqdm.auto import trange

from agents.agent import TrainableAgent
from replay_buffer import ReplayBuffer
from utils import discount, generate_environment, generate_vec_environment, obs_to_tensor


def train(
        agent: TrainableAgent,
        env_name,
        n_episodes=10,
        n_parallel=128,
        buffer_size=512,
        batch_size=128,
):
    # Initialize the environment
    venv = generate_vec_environment(n_parallel, env_name)
    replay_buffer = ReplayBuffer(buffer_size)

    # Initialize torch stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.RAdam(agent.parameters, lr=1e-5, weight_decay=0.01)
    agent.compile(device)
    agent.train()

    for episode in range(n_episodes):
        # Generate the episodes
        agent.reset()
        episodes = replay_buffer.generate_vectorized(venv, agent, device, progress_bar=True).episodic()

        # Compute discount rewards
        mean_reward: torch.Tensor = torch.mean(torch.cat(episodes.rewards)).item()
        episodes.rewards = discount(episodes.rewards, gamma=0.99)
        mean_discounted = torch.mean(torch.cat(episodes.rewards)).item()

        # Do a training step
        n_losses = 0
        sum_loss = 0
        losses = []
        batches = episodes.tensor().batch(batch_size)
        for batch in batches:
            loss = agent.loss(batch.to(device))
            # sum_loss = sum_loss + loss
            # n_losses += 1
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print()
        print("Mean reward", mean_reward)
        print("Mean discounted reward", mean_discounted)
        print("Loss", np.mean(losses))
        print("Episode", episode)
    return agent


def evaluate(agent, env_name, n_episodes=10):
    agent.eval()
    env = generate_environment(env_name=env_name, render="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for _ in trange(n_episodes):
            agent.reset()
            obs = env.reset()
            while True:
                obs = obs_to_tensor(obs).to(device)
                action = agent.act(obs)
                chosen_action = action.sample().item()
                obs, reward, done, _ = env.step(chosen_action)
                if done:
                    break


def train_eval(agent: TrainableAgent, save_path, env_name="coinrun"):
    agent = train(agent, env_name=env_name)
    agent.save(save_path)
    input("Waiting for input")
    evaluate(agent, env_name=env_name)