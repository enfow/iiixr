import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from typing import Union

import gymnasium as gym
import numpy as np
import torch

from schema.train_result import EvalResult, TrainResult
from util.file import log_result, save_json


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class BaseConfig:
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
    eval: bool = False
    eval_period: int = 10
    eval_episodes: int = 10

    @classmethod
    def from_dict(cls, config: dict):
        cls._check_device(config)
        valid_keys = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

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


class BaseTrainer:
    def __init__(self, env, config, save_dir):
        set_seed(config.seed)

        self.env = env
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = int(
            env.action_space.n
            if isinstance(env.action_space, gym.spaces.Discrete)
            else env.action_space.shape[0]
        )
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        self.best_score = -np.inf
        self.total_steps = 0
        self.scores = []
        self.losses = []
        self.episode_elapsed_times = []
        self.best_results: EvalResult = None
        self.log_file = os.path.join(self.save_dir, "metrics.jsonl")
        self.model_file = os.path.join(self.save_dir, "best_model.pth")
        self.config_file = os.path.join(self.save_dir, "config.json")

        self._log_config()
        self._init_models()

    def _init_models(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train_episode(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self):
        for ep in range(self.config.episodes):
            start_time = time.time()
            result = self.train_episode()
            elapsed_time = time.time() - start_time

            if "total_reward" not in result or "losses" not in result:
                raise ValueError("Results must have total_reward and losses")

            avg_loss = np.mean(result["losses"]) if result["losses"] else 0.0

            self.scores.append(result["total_reward"])
            self.losses.append(avg_loss)
            self.episode_elapsed_times.append(elapsed_time)

            # log train result
            train_result = TrainResult.from_train_results(
                self.scores, self.losses, self.episode_elapsed_times
            )
            print(train_result)
            log_result(train_result, self.log_file)

            # evaluate
            if self.config.eval and ep % self.config.eval_period == 0:
                eval_result = self.evaluate(self.config.eval_episodes)
                if self.best_results is None or eval_result > self.best_results:
                    self.best_results = eval_result
                    print("New best results:")
                print(eval_result)
                log_result(eval_result, self.log_file)
                self.save_model()

    def evaluate(self, episodes=10):
        self.eval_mode_on()

        scores = []
        steps = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                action = self.select_action(state)["action"]
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

                step += 1
                if step >= self.config.max_steps:
                    break

            scores.append(total_reward)
            steps.append(step)

        self.eval_mode_off()

        return EvalResult.from_eval_results(scores, steps)

    def select_action(self, state) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    def update(self):
        raise NotImplementedError("Subclasses must implement this method")

    def save_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def eval_mode_on(self):
        raise NotImplementedError("Subclasses must implement this method")

    def eval_mode_off(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _log_config(self):
        save_json(self.config.to_dict(), self.config_file)
