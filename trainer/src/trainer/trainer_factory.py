import gymnasium as gym

from schema.config import BaseConfig, PPOConfig
from trainer.discrete_ppo_trainer import DiscretePPOTrainer
from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer
from trainer.td3_trainer import TD3Trainer
from util.gym_env import is_discrete_action_space


class PPOTrainerFactory:
    # factory pattern
    def __new__(cls, env: gym.Env, config: PPOConfig, save_dir: str):
        if is_discrete_action_space(env):
            print("Discrete action space")
            return DiscretePPOTrainer(env, config, save_dir)
        print("Continuous action space")
        return PPOTrainer(env, config, save_dir)


class TrainerFactory:
    selectable_models = ["ppo", "sac", "rainbow_dqn", "discrete_sac", "td3"]

    def __new__(cls, env: gym.Env, config: BaseConfig, save_dir: str):
        model_name = config["model"]
        if model_name == "ppo":
            return PPOTrainerFactory(env, config, save_dir)
        elif model_name == "sac":
            return SACTrainer(env, config, save_dir)
        elif model_name == "rainbow_dqn":
            return RainbowDQNTrainer(env, config, save_dir)
        elif model_name == "discrete_sac":
            return DiscreteSACTrainer(env, config, save_dir)
        elif model_name == "td3":
            return TD3Trainer(env, config, save_dir)
        else:
            raise ValueError(f"Unknown model: {model_name}")
