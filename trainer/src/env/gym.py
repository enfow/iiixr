import gymnasium as gym
from pydantic import BaseModel


class CustomEnv(BaseModel):
    env_name: str
    kwargs: dict


CUSTOM_ENVS = {
    "BipedalWalkerHardcore-v3": CustomEnv(
        env_name="BipedalWalker-v3", kwargs={"hardcore": True}
    ),
}


class GymEnvFactory:
    """
    Factory class for gym environments.
    """

    _registered_envs = list(gym.envs.registry.keys())

    def __new__(cls, env_name: str, **kwargs):
        if env_name in CUSTOM_ENVS:
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
    env = GymEnvFactory("BipedalWalkerHardcore-v3")
    print(env)
