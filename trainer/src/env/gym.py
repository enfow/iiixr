from typing import Any, Dict, Optional, Type

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class CustomBipedalWalkerWrapper(gym.Wrapper):
    """
    A custom wrapper for BipedalWalker environments that adds a penalty for falling.
    """

    def __init__(self, env: gym.Env, config: Dict[str, Any] = None):
        super().__init__(env)
        self.config = config if config is not None else {}
        self.fall_penalty = self.config.get("penalty", -100.0)
        print(
            f"-> CustomBipedalWalkerWrapper applied with fall_penalty: {self.fall_penalty}"
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and reward < 0:
            reward += self.fall_penalty
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class CurriculumBipedalWalkerWrapper(gym.Wrapper):
    is_curriculum = True
    _default_curriculum_threshold = 200.0

    def __init__(self, env: gym.Env, config: Dict[str, Any] = None):
        super().__init__(env)
        self.is_hardcore = False
        self.curriculum_threshold = config.get("curriculum_threshold", None)
        if self.curriculum_threshold is None:
            self.curriculum_threshold = self._default_curriculum_threshold
            print(
                f"CurriculumBipedalWalkerWrapper applied with default curriculum_threshold: {self.curriculum_threshold}"
            )
        else:
            print(
                f"CurriculumBipedalWalkerWrapper applied with curriculum_threshold: {self.curriculum_threshold}"
            )

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def set_hardcore(self, score: float):
        """Sets the environment to hardcore mode."""
        if not self.is_hardcore and score > self.curriculum_threshold:
            print("\n" + "=" * 40)
            print(
                f"CHANGING ENVIRONMENT TO HARDCORE: {score:.2f} > {self.curriculum_threshold:.2f}"
            )
            print("=" * 40 + "\n")
            self.env.unwrapped.hardcore = True
            self.is_hardcore = True


class CustomEnv(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    env_name: str
    kwargs: dict = {}
    wrapper: Optional[Type[gym.Wrapper]] = None
    wrapper_config: dict = {}

    @field_validator("wrapper")
    @classmethod
    def validate_wrapper(cls, v):
        if v and not (isinstance(v, type) and issubclass(v, gym.Wrapper)):
            raise ValueError("Wrapper must be a subclass of gym.Wrapper")
        return v


class TestGymEnv(gym.Env):
    def __init__(self, env_name: str, **kwargs):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

    def close(self):
        pass


CUSTOM_ENVS = {
    "BipedalWalkerHardcore-v3": CustomEnv(
        env_name="BipedalWalker-v3", kwargs={"hardcore": True}
    ),
    "CustomBipedalWalkerHardcore-v3": CustomEnv(
        env_name="BipedalWalker-v3",
        kwargs={"hardcore": True},
        wrapper=CustomBipedalWalkerWrapper,
        wrapper_config={"penalty": -100.0},
    ),
    "CurriculumBipedalWalker-v3": CustomEnv(
        env_name="BipedalWalker-v3",
        wrapper=CurriculumBipedalWalkerWrapper,
        wrapper_config={},
    ),
}


class GymEnvFactory:
    """
    Factory for creating single or vectorized gym environments.
    """

    _registered_envs = list(gym.envs.registry.keys())

    @staticmethod
    def _generate_env(env_name: str, **kwargs) -> gym.Env:
        """Creates a single environment instance. Corrected to be a staticmethod."""
        if env_name in CUSTOM_ENVS:
            custom_def = CUSTOM_ENVS[env_name]
            if env_name == "TestGymEnv":
                return TestGymEnv(env_name, **custom_def.kwargs)

            # Create the base environment
            base_env = gym.make(custom_def.env_name, **custom_def.kwargs)

            # Apply wrapper if defined
            if custom_def.wrapper:
                _base_config = custom_def.wrapper_config
                if "curriculum_threshold" in kwargs:
                    _base_config["curriculum_threshold"] = kwargs[
                        "curriculum_threshold"
                    ]
                return custom_def.wrapper(base_env, config=_base_config)
            return base_env

        elif env_name in GymEnvFactory._registered_envs:
            return gym.make(env_name, **kwargs)
        else:
            raise ValueError(f"Invalid environment name: {env_name}")

    def __new__(cls, env_name: str, n_envs: int = 1, **kwargs):
        """Creates a single or vectorized environment."""
        if n_envs < 1:
            raise ValueError(f"Number of environments must be at least 1, got {n_envs}")

        if n_envs == 1:
            return cls._generate_env(env_name, **kwargs)
        else:
            print(f"Creating {n_envs} vectorized environments")
            # Create a vectorized environment
            env_fns = [
                lambda: cls._generate_env(env_name, **kwargs) for _ in range(n_envs)
            ]
            return gym.vector.AsyncVectorEnv(env_fns)

    @classmethod
    def valid_envs(cls):
        return cls._registered_envs + list(CUSTOM_ENVS.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("### TEST 1: Creating a single, standard gym environment ###")
    env1 = GymEnvFactory("CartPole-v1")
    print(f"Successfully created: {env1}")
    obs, info = env1.reset()
    print(f"Observation shape: {obs.shape}")
    action = env1.action_space.sample()
    obs, reward, term, trunc, info = env1.step(action)
    print(f"Step successful. Reward: {reward}")
    env1.close()
    print("Test 1 PASSED")
    print("=" * 60)

    print("\n### TEST 2: Creating a single, custom environment WITHOUT wrapper ###")
    env2 = GymEnvFactory("BipedalWalkerHardcore-v3")
    print(f"Successfully created: {env2}")
    obs, info = env2.reset()
    print(f"Observation shape: {obs.shape}")
    env2.close()
    print("Test 2 PASSED")
    print("=" * 60)

    print("\n### TEST 3: Creating a single, custom environment WITH wrapper ###")
    env3 = GymEnvFactory("CustomBipedalWalkerHardcore-v3")
    print(f"Successfully created: {env3}")
    obs, info = env3.reset()
    print(f"Observation shape: {obs.shape}")
    env3.close()
    print("Test 3 PASSED")
    print("=" * 60)

    print("\n### TEST 4: Creating a PARALLEL, custom-wrapped environment ###")
    N_ENVS = 4
    env4 = GymEnvFactory("CustomBipedalWalkerHardcore-v3", n_envs=N_ENVS)
    print(f"Successfully created: {env4}")
    obs, info = env4.reset()
    print(f"Vectorized observation shape: {obs.shape}")
    assert obs.shape == (N_ENVS, 24), "Vectorized observation shape is incorrect!"
    actions = env4.action_space.sample()
    print(f"Vectorized action shape: {actions.shape}")
    assert actions.shape == (N_ENVS, 4), "Vectorized action shape is incorrect!"
    obs, rewards, terms, truncs, infos = env4.step(actions)
    print(f"Vectorized step successful. Rewards shape: {rewards.shape}")
    assert rewards.shape == (N_ENVS,), "Vectorized rewards shape is incorrect!"
    env4.close()
    print("Test 4 PASSED")
    print("=" * 60)
