import os

import gymnasium as gym
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

        self._init_models(config)

    def _init_models(self, config):
        pass

    def select_action(self, state):
        pass

    def update(self):
        pass

    def train(self, episodes, max_steps):
        pass
