from typing import Any, List, Sequence, Tuple

import gym
import gym3
import numpy as np
import procgen
from gym.core import ActType, ObsType
from gym.spaces import Discrete


def generate_environment(env_name="coinrun", render=None) -> gym.Env:
    return gym.make(
        f"procgen:procgen-{env_name}-v0",
        render=render,
        start_level=0,
        num_levels=0,
        distribution_mode="hard",  # "easy", "hard", "extreme", "memory", "exploration"
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=False,
    )


def generate_vec_environment(n_parallel, env_name="coinrun") -> gym3.Env:
    return procgen.ProcgenGym3Env(
        num=n_parallel,
        env_name=env_name,
        start_level=0,
        num_levels=0,
        rand_seed=42,
        distribution_mode="hard",  # "easy", "hard", "extreme", "memory", "exploration"
        restrict_themes=True,
        use_backgrounds=False,
        use_monochrome_assets=False,
    )

class CoinRunEnv:
    def __init__(self, n_parallel, render=None):
        self.action_space = Discrete(4)
        self.env = generate_vec_environment(n_parallel, env_name="coinrun")

        self.action_mapping = {
            0: 1,  # 0 -> LEFT
            1: 7,  # 1 -> RIGHT
            2: 5,  # 2 -> UP
            3: 4,  # 3 -> Nothing
        }
        self._translate_action = np.vectorize(self.action_mapping.get)

    def act(self, ac: Any) -> None:
        action = self._translate_action(ac)
        self.env.act(action)

    def callmethod(self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]) -> List[Any]:
        return self.env.callmethod(method, args, kwargs)

    def observe(self) -> Tuple[Any, Any, Any]:
        return self.env.observe()