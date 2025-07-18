import json
from enum import Enum
from typing import Optional

import torch
from pydantic import BaseModel


class ModelEmbeddingType(str, Enum):
    FC = "fc"
    TRANSFORMER = "transformer"
    LSTM = "lstm"


class BufferType(str, Enum):
    DEFAULT = "default"  # simple replay buffer
    SEQUENTIAL = "sequential"  # sequence replay buffer
    PER = "per"  # prioritized experience replay
    PER_SEQUENTIAL = (
        "per_sequential"  # prioritized experience replay for sequential data
    )


class ModelConfig(BaseModel):
    # TODO: change it to model_name
    model: str = None
    hidden_dim: int = 256
    n_layers: int = 3
    embedding_type: ModelEmbeddingType = ModelEmbeddingType.FC
    seq_len: int = 1
    seq_stride: int = 1
    use_layernorm: bool = False

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)


class BufferConfig(BaseModel):
    buffer_size: int = 1000000
    buffer_type: BufferType = BufferType.DEFAULT
    # Prioritized Experience Replay
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000
    # Multi-step learning
    per_n_steps: int = 3

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)


class BaseConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    buffer: BufferConfig = BufferConfig()
    seed: int = 42
    episodes: int = 1000
    max_steps: int = 1000
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    # env
    env: str = None
    eval_env: str = None
    n_envs: int = 1  # if use multi-env
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    is_discrete: Optional[bool] = None
    # device
    device: str = "cpu"
    # evaluation
    eval: bool = True
    eval_period: int = 10
    eval_episodes: int = 10
    # curriculum
    curriculum_threshold: Optional[float] = None
    # exploration noise
    start_exploration_noise: float = 0.2
    end_exploration_noise: float = 0.2  # default is no decay
    exploration_noise_decay_episodes: int = 1000

    @classmethod
    def from_dict(cls, config: dict):
        # Handle nested model config
        if "model" in config and isinstance(config["model"], dict):
            # If model is a dict, it contains model-specific config
            model_config = config["model"]
            # Extract model name if present
            model_name = model_config.get("model")
            if model_name:
                config["model"] = model_name
            # Create ModelConfig instance
            config["model"] = ModelConfig.from_dict(model_config)
        elif "model" in config and isinstance(config["model"], str):
            # If model is a string, create ModelConfig with just the model name
            model_name = config["model"]
            config["model"] = ModelConfig(model=model_name)

        if "buffer" in config and isinstance(config["buffer"], dict):
            buffer_config = config["buffer"]
            config["buffer"] = BufferConfig.from_dict(buffer_config)
        elif "buffer" in config and isinstance(config["buffer"], str):
            config["buffer"] = BufferConfig(
                buffer_type=BufferType.DEFAULT,
                buffer_size=config.get("buffer_size", 1000000),
                alpha=config.get("alpha", 0.6),
                beta_start=config.get("beta_start", 0.4),
                beta_frames=config.get("beta_frames", 100000),
            )

        if "start_exploration_noise" not in config and "exploration_noise" in config:
            config["start_exploration_noise"] = config["exploration_noise"]
            config["end_exploration_noise"] = config["exploration_noise"]
            config["exploration_noise_decay_episodes"] = config["episodes"]
            del config["exploration_noise"]

        cls._check_validity(config)
        cls._check_device(config)
        return cls(**config)

    @staticmethod
    def _check_validity(config: dict):
        model_config = config["model"]
        buffer_config = config.get("buffer", BufferConfig())

        # Ensure we have proper config objects
        if not isinstance(model_config, ModelConfig):
            raise ValueError("Model config must be a ModelConfig instance")
        if not isinstance(buffer_config, BufferConfig):
            raise ValueError("Buffer config must be a BufferConfig instance")

        if buffer_config.buffer_type == BufferType.SEQUENTIAL:
            if model_config.seq_len == 1:
                raise ValueError("seq_len must be greater than 1 for sequential buffer")

        if (
            model_config.model not in ["ppo", "td3", "td3_seq", "ppo_seq"]
            and model_config.use_layernorm
        ):
            raise ValueError(
                "currently, use_layernorm is only supported for ppo, td3, td3_seq, and ppo_seq"
            )
        if model_config.model == "td3_seq":
            if buffer_config.buffer_type == BufferType.PER:
                raise ValueError("PER is not supported for td3_seq")
            if model_config.seq_len == 1:
                raise ValueError("seq_len must be greater than 1 for td3_seq")
            if model_config.embedding_type == ModelEmbeddingType.FC:
                raise ValueError("fc is not supported for td3_seq")

        if model_config.model == "ppo_seq":
            if model_config.embedding_type == ModelEmbeddingType.FC:
                raise ValueError("fc is not supported for ppo_seq")
            if buffer_config.buffer_type == BufferType.PER:
                raise ValueError("PER is not supported for ppo_seq")
            if model_config.seq_len == 1:
                raise ValueError("seq_len must be greater than 1 for ppo_seq")

    @staticmethod
    def _check_device(config: dict):
        if "device" not in config:
            return
        if config["device"] not in ["cuda", "cpu"]:
            raise ValueError("Invalid device")
        elif config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        elif config["device"] == "cpu" and torch.cuda.is_available():
            print("CUDA is available but device is set to CPU")

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
    gae: bool = False
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5


class SACConfig(BaseConfig):
    tau: float = 0.005
    entropy_coef: float = 1.0
    start_steps: int = 10000
    policy_update_interval: int = 2


class RainbowDQNConfig(BaseConfig):
    target_update_interval: int = 10000
    # Multi-step Learning
    n_steps: int = 3
    # Distributional RL (Categorical DQN)
    n_atoms: int = 51
    v_min: float = -10.0  # should be set up to env (min return)
    v_max: float = 10.0  # should be set up to env (max return)
    start_steps: int = 10000


class DoubleDQNConfig(BaseConfig):
    target_update_interval: int = 10000
    start_steps: int = 10000
    # Epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 50000


class C51Config(BaseConfig):
    target_update_interval: int = 10000
    start_steps: int = 10000
    n_atoms: int = 51
    v_min: float = -10.0  # should be set up to env (min return)
    v_max: float = 10.0  # should be set up to env (max return)
    n_steps: int = 3  # simple 1-step return
    # Epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 50000


class TD3Config(BaseConfig):
    tau: float = 0.005
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    start_steps: int = 10000


class TD3ForkConfig(TD3Config):
    fork_alpha: float = 0.5
    fork_hidden_dim: int = 256
    fork_n_layers: int = 2


class EvalConfig:
    eval_episodes: int = 10
    eval_render: bool = False
    eval_save_dir: str = None

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)
