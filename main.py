import gym
from procgen.env import ENV_NAMES

from src.agent import Agent
from src.replay_buffer import ReplayBuffer
from src.utils import render_buffer


def main():
    # Environment init
    print("Possible environments:", ENV_NAMES)
    env_name = "bossfight"

    env: gym.Env = gym.make(
        f"procgen:procgen-{env_name}-v0",
        # render="human",
        start_level=0,
        distribution_mode="hard",  # "easy", "hard", "extreme", "memory", "exploration"
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=False,
    )

    # Agent init
    agent = Agent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        train=True,
        eps=0.5,
    )

    # Create a sample ReplayBuffer
    rb = ReplayBuffer(1000, n_episodes=1)
    rb(env, agent)
    print(len(rb))
    render_buffer(rb)


if __name__ == "__main__":
    main()
