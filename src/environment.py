from typing import Any, List, Sequence, Tuple

import gym
import gym3
import numpy as np
import procgen
import torch
from gym.spaces import Discrete
from tqdm.auto import trange

from agents.agent import Agent
from trajectories import ListTrajectory


def generate_environment(env_name="coinrun", render=None, difficulty="easy") -> gym.Env:
    return gym.make(
        f"procgen:procgen-{env_name}-v0",
        render=render,
        start_level=0,
        num_levels=0,
        distribution_mode=difficulty,  # "easy", "hard", "extreme", "memory", "exploration"
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=False,
    )


def generate_vec_environment(n_parallel, env_name="coinrun", seed=42, difficulty="easy") -> gym3.Env:
    return procgen.ProcgenGym3Env(
        num=n_parallel,
        env_name=env_name,
        start_level=0,
        num_levels=0,
        rand_seed=seed,
        distribution_mode=difficulty,  # "easy", "hard", "extreme", "memory", "exploration"
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=False,
    )


class CoinRunEnv:
    """Wrapper for the coinrun vectorized procgen environment, using a smaller and less redundant action space"""
    def __init__(self, n_parallel, render=None, seed=42, difficulty="easy"):
        self.action_space = Discrete(4)
        self.env = generate_vec_environment(n_parallel, env_name="coinrun", seed=seed, difficulty=difficulty)

        # Remapping of the action space. Agent action -> Procgen action
        self.action_mapping = {
            0: 1,  # 0 -> LEFT
            1: 7,  # 1 -> RIGHT
            2: 5,  # 2 -> UP
            3: 4,  # 3 -> Nothing
        }
        self._translate_action = np.vectorize(self.action_mapping.get)

    def act(self, ac: Any) -> None:
        """Perform an action, converting it to the appropriate procgen encoding"""
        action = self._translate_action(ac)
        self.env.act(action)

    def callmethod(self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]) -> List[Any]:
        """Calls an internam method of the procgen environment. Mostly used to get/set the internal state"""
        return self.env.callmethod(method, *args, **kwargs)

    def observe(self) -> Tuple[Any, Any, Any]:
        """Return the agent's observation of the environment"""
        return self.env.observe()

    def __call__(self, agent: Agent, device, n_steps, use_tqdm=False) -> "ListTrajectory":
        """
        Generate the trajectory of the episodes. The number of total steps is n_parallel * n_steps
        :param agent: agent performing the actions
        :param device: device the agent runs on
        :param n_steps: number of steps per parallel environment
        :param use_tqdm: if showing the tqdm progress bar
        :return: ListTrajectory containing a list of trajectories, one for each episode
        """
        trajectory = ListTrajectory.empty()
        steps = trange(n_steps, leave=False, colour='blue', desc="Trajectory generation") if use_tqdm else range(n_steps)

        with torch.no_grad():
            rew, obs, first = self.observe()

            for _ in steps:
                # Convert obs tensor [N, W, H, C] to [N, C, W, H]
                obs: torch.Tensor = torch.from_numpy(obs["rgb"]).float().to(device)
                obs = obs.permute((0, 3, 1, 2)) / 255

                # Get the action from the agent. If output is tuple, then it is (action, state value)
                agent_output = agent.act(obs)
                if isinstance(agent_output, tuple):
                    action_dist, values = agent_output
                    values = values.to("cpu")
                else:
                    action_dist, values = agent_output, None

                # Sample the actions and act
                chosen_actions = agent.sampling_strategy(action_dist)
                self.act(chosen_actions.cpu().detach().numpy())

                # Observe the effects of the actions.
                # IMPORTANT: the reward we add to the trajectory is not the one of the current state,
                # but the one of the state following the action
                next_rew, next_obs, next_first = self.observe()
                rew = next_rew

                # Remove tensors from GPU (no effect if using CPU) and append to trajectory
                trajectory.append(
                    obs=obs.to("cpu"),
                    actions=chosen_actions.to("cpu"),
                    rewards=torch.from_numpy(next_rew).to("cpu"),
                    is_first=torch.from_numpy(first).bool().to("cpu"),
                    probs=action_dist.probs.to("cpu"),
                    values=values,
                )

                # Step
                obs, first = next_obs, next_first
        return trajectory.tensor().episodic()
