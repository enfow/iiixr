import gymnasium as gym

from schema.config import PPOConfig
from trainer.continuous_ppo_trainer import ContinuousPPOTrainer
from trainer.discrete_ppo_trainer import DiscretePPOTrainer
from util.gym_env import is_discrete_action_space


class PPOTrainer:
    def __new__(cls, env: gym.Env, config: PPOConfig, save_dir: str):
        if is_discrete_action_space(env):
            print("Discrete action space")
            return DiscretePPOTrainer(env, config, save_dir)
        print("Continuous action space")
        return ContinuousPPOTrainer(env, config, save_dir)
