import json
import os
from dataclasses import asdict, dataclass, fields

import gymnasium as gym
import numpy as np
import torch


@dataclass
class BaseConfig:
    model: str = None
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
    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_dict(cls, config: dict, env: gym.Env):
        if env is not None:
            config["env_name"] = env.__class__.__name__
            config["state_dim"] = env.observation_space.shape[0]
            config["action_dim"] = int(
                env.action_space.n
                if isinstance(env.action_space, gym.spaces.Discrete)
                else env.action_space.shape[0]
            )
        valid_keys = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


class BaseTrainer:
    def __init__(self, env, config, save_dir):
        self.env = env
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = (
            env.action_space.n if self.is_discrete else env.action_space.shape[0]
        )

        self.best_score = -np.inf
        self.scores = []
        self.losses = []
        self.episode_losses = []
        self.log_file = os.path.join(self.save_dir, "metrics.jsonl")
        self.model_file = os.path.join(self.save_dir, "best_model.pth")
        self.config_file = os.path.join(self.save_dir, "config.json")

        self._log_config()
        self._init_models()

    def _init_models(self):
        raise NotImplementedError("Subclasses must implement this method")

    def select_action(self, state):
        raise NotImplementedError("Subclasses must implement this method")

    def update(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self, episodes, max_steps):
        raise NotImplementedError("Subclasses must implement this method")

    def _log_metrics(self):
        with open(self.log_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "episode": len(self.scores),
                        "return": float(self.scores[-1]),
                        "loss": float(self.losses[-1]),
                    }
                )
                + "\n"
            )
        print(
            f"Episode {len(self.scores)}: Return={self.scores[-1]:.2f}, Loss={self.losses[-1]:.4f}"
        )

    def _log_config(self):
        with open(self.config_file, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def save_model(self):
        pass

    def load_model(self):
        pass

    def ready_to_evaluate(self):
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate(self, episodes=10):
        self.ready_to_evaluate()

        scores = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            scores.append(total_reward)

        return scores
