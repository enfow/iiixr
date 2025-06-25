import gymnasium as gym

from schema.config import PPOConfig
from trainer.discrete_ppo_trainer import DiscretePPOTrainer
from trainer.ppo_trainer import PPOTrainer
from util.gym_env import is_discrete_action_space


class PPOTrainerFactory:
    # factory pattern
    def __new__(cls, env: gym.Env, config: PPOConfig, save_dir: str):
        if is_discrete_action_space(env):
            print("Discrete action space")
            return DiscretePPOTrainer(env, config, save_dir)
        print("Continuous action space")
        return PPOTrainer(env, config, save_dir)
