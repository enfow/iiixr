import json
import os

import gymnasium as gym
import numpy as np
import torch


class BaseTrainer:
    def __init__(self, env, config, save_dir):
        self.env = env
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self._init_models(config)

    def _init_models(self, config):
        pass

    def select_action(self, state):
        pass

    def update(self):
        pass

    def train(self, episodes, max_steps):
        pass

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
            json.dump(self.config, f, indent=2)
