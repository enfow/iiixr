import json
import time

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
    n_layers: int = 3
    buffer_size: int = 1000000
    batch_size: int = 256
    # env
    env: str = None
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
    n_transactions: int = 1000
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    normalize_advantages: bool = False
    entropy_coef: float = 0.01


class SACConfig(BaseConfig):
    tau: float = 0.005
    entropy_coef: float = 1.0
    start_steps: int = 1000


class RainbowDQNConfig(BaseConfig):
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000
    target_update: int = 10


class TD3Config(BaseConfig):
    tau: float = 0.005
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    start_steps: int = 25000


class EvalConfig:
    eval_episodes: int = 10
    eval_render: bool = False
    eval_save_dir: str = None

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)


class TrainingConfigFactory:
    def __new__(cls, config_dict: dict):
        model_name = config_dict["model"]
        if model_name == "ppo":
            return PPOConfig.from_dict(config_dict)
        elif model_name in ["sac", "discrete_sac"]:
            return SACConfig.from_dict(config_dict)
        elif model_name == "rainbow_dqn":
            return RainbowDQNConfig.from_dict(config_dict)
        elif model_name == "td3":
            return TD3Config.from_dict(config_dict)
        else:
            raise ValueError(f"Unknown model for configuration: {model_name}")
