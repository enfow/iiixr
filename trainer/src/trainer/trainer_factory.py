import gymnasium as gym

from schema.config import (BaseConfig, PPOConfig, RainbowDQNConfig, SACConfig,
                           TD3Config)
from trainer.discrete_ppo_trainer import DiscretePPOTrainer
from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer
from trainer.td3_trainer import TD3Trainer
from util.gym_env import is_discrete_action_space


class PPOTrainerFactory:
    # factory pattern
    def __new__(cls, env: gym.Env, config: BaseConfig, save_dir: str):
        if is_discrete_action_space(env):
            print("Discrete action space")
            return DiscretePPOTrainer(env, config, save_dir)
        print("Continuous action space")
        return PPOTrainer(env, config, save_dir)


class TrainerFactory:
    selectable_models = ["ppo", "sac", "rainbow_dqn", "discrete_sac", "td3"]

    def __new__(cls, env_name: str, config_dict: dict, save_dir: str):
        model_name = config_dict["model"]
        if model_name == "ppo":
            config = PPOConfig.from_dict(config_dict)
            return PPOTrainerFactory(env_name, config, save_dir)
        elif model_name == "sac":
            config = SACConfig.from_dict(config_dict)
            return SACTrainer(env_name, config, save_dir)
        elif model_name == "rainbow_dqn":
            config = RainbowDQNConfig.from_dict(config_dict)
            return RainbowDQNTrainer(env_name, config, save_dir)
        elif model_name == "discrete_sac":
            config = SACConfig.from_dict(config_dict)
            return DiscreteSACTrainer(env_name, config, save_dir)
        elif model_name == "td3":
            config = TD3Config.from_dict(config_dict)
            return TD3Trainer(env_name, config, save_dir)
        else:
            raise ValueError(f"Unknown model: {model_name}")
