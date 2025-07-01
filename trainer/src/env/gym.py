from typing import Optional, Type

import gymnasium as gym
from pydantic import BaseModel, ConfigDict, field_validator

from env.custom_env import CustomBipedalWalkerWrapper


class CustomEnv(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_name: str
    kwargs: dict
    wrapper: Optional[Type[gym.Wrapper]] = None

    @field_validator("wrapper")
    @classmethod
    def validate_wrapper(cls, v):
        if v is not None:
            if not (isinstance(v, type) and issubclass(v, gym.Wrapper)):
                raise ValueError("Must be a subclass of gym.Wrapper")
        return v


class TestGymEnv(gym.Env):
    def __init__(self, env_name: str, **kwargs):
        self.env_name = env_name
        self.kwargs = kwargs
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

    def close(self):
        return


CUSTOM_ENVS = {
    "BipedalWalkerHardcore-v3": CustomEnv(
        env_name="BipedalWalker-v3", kwargs={"hardcore": True}
    ),
    "CustomBipedalWalkerHardcore-v3": CustomEnv(
        env_name="BipedalWalker-v3",
        kwargs={"hardcore": True},
        wrapper=CustomBipedalWalkerWrapper,
    ),
    "TestGymEnv": CustomEnv(env_name="TestGymEnv", kwargs={}),
}


class GymEnvFactory:
    """
    Factory class for gym environments.
    """

    _registered_envs = list(gym.envs.registry.keys())

    def __new__(cls, env_name: str, **kwargs):
        if env_name in CUSTOM_ENVS:
            if env_name == "TestGymEnv":
                return TestGymEnv(env_name, **kwargs)
            else:
                if CUSTOM_ENVS[env_name].wrapper:
                    return CUSTOM_ENVS[env_name].wrapper(
                        gym.make(
                            CUSTOM_ENVS[env_name].env_name,
                            **CUSTOM_ENVS[env_name].kwargs,
                        )
                    )
                else:
                    return gym.make(
                        CUSTOM_ENVS[env_name].env_name, **CUSTOM_ENVS[env_name].kwargs
                    )
        elif env_name in cls._registered_envs:
            return gym.make(env_name, **kwargs)
        else:
            raise ValueError(f"Invalid environment name: {env_name}")

    @classmethod
    def valid_envs(cls):
        return cls._registered_envs + list(CUSTOM_ENVS.keys())


if __name__ == "__main__":
    print(GymEnvFactory.valid_envs())
    env = GymEnvFactory("CustomBipedalWalkerHardcore-v3")
    print(env)
