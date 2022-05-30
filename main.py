def main():
    pass
    # # Environment init
    # print("Possible environments:", ENV_NAMES)
    # env_name = "bossfight"
    #
    # env: gym.Env = generate_environment(env_name, "human")
    #
    # # Agent init
    # agent = RandomAgent(
    #     obs_space_shape=env.observation_space,
    #     act_space_size=env.action_space,
    # )
    #
    # # Create a sample ReplayBuffer
    # rb = ReplayBuffer(1000, n_episodes=1)
    # rb(env, agent)
    # print(len(rb))
    # render_buffer(rb)


if __name__ == "__main__":
    main()
