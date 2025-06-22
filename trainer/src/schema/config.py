import json

import torch
from pydantic import BaseModel


class BaseConfig(BaseModel):
    model: str = None
    seed: int = 42
    episodes: int = 1000
    max_steps: int = 1000
    lr: float = 3e-4
    gamma: float = 0.99
    hidden_dim: int = 256
    buffer_size: int = 1000000
    batch_size: int = 256
    # env
    env_name: str = None
    state_dim: int = None
    action_dim: int = None
    is_discrete: bool = None
    # device
    device: str = "cpu"
    # evaluation
    eval: bool = True
    eval_period: int = 10
    eval_episodes: int = 10

    @classmethod
    def from_dict(cls, config: dict):
        cls._check_device(config)
        print(config)
        return cls(**config)

    @staticmethod
    def _check_device(config: dict):
        if "device" not in config:
            return
        if config["device"] not in ["cuda", "cpu"]:
            raise ValueError("Invalid device")
        elif config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        elif config["device"] == "cpu" and torch.cuda.is_available():
            raise ValueError("CUDA is available but device is set to CPU")

    def to_dict(self):
        return self.model_dump()

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


class PPOConfig(BaseConfig):
    ppo_epochs: int = 4
    clip_eps: float = 0.2


class SACConfig(BaseConfig):
    tau: float = 0.005
    entropy_coef: float = 1.0
    start_steps: int = 1000


class RainbowDQNConfig(BaseConfig):
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000
    target_update: int = 10
