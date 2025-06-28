import gymnasium as gym

from env.gym import GymEnvFactory
from trainer.ddqn_trainer import DDQNTrainer
from trainer.discrete_ppo_trainer import DiscretePPOTrainer
from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer
from trainer.sac_v2_trainer import SACV2Trainer
from trainer.td3_sequential_trainer import TD3SequentialTrainer
from trainer.td3_trainer import TD3Trainer
from util.gym_env import is_discrete_action_space

SELECTABLE_MODELS = [
    PPOTrainer.name,
    RainbowDQNTrainer.name,
    TD3Trainer.name,
    SACTrainer.name,
    SACV2Trainer.name,
    DiscreteSACTrainer.name,
    TD3SequentialTrainer.name,
    DDQNTrainer.name,
]


class PPOTrainerFactory:
    # factory pattern
    def __new__(cls, env_name: str, config_dict: dict, save_dir: str):
        env = GymEnvFactory(env_name)
        is_discrete = is_discrete_action_space(env)
        del env
        if is_discrete:
            print("Discrete action space")
            return DiscretePPOTrainer(env_name, config_dict, save_dir)
        print("Continuous action space")
        return PPOTrainer(env_name, config_dict, save_dir)


class TrainerFactory:
    selectable_models = SELECTABLE_MODELS

    def __new__(cls, env_name: str, config_dict: dict, save_dir: str):
        model_name = config_dict["model"]["model"]
        if model_name == PPOTrainer.name:
            return PPOTrainerFactory(env_name, config_dict, save_dir)
        elif model_name == SACTrainer.name:
            return SACTrainer(env_name, config_dict, save_dir)
        elif model_name == SACV2Trainer.name:
            return SACV2Trainer(env_name, config_dict, save_dir)
        elif model_name == RainbowDQNTrainer.name:
            return RainbowDQNTrainer(env_name, config_dict, save_dir)
        elif model_name == DiscreteSACTrainer.name:
            return DiscreteSACTrainer(env_name, config_dict, save_dir)
        elif model_name == TD3Trainer.name:
            return TD3Trainer(env_name, config_dict, save_dir)
        elif model_name == TD3SequentialTrainer.name:
            return TD3SequentialTrainer(env_name, config_dict, save_dir)
        elif model_name == DDQNTrainer.name:
            return DDQNTrainer(env_name, config_dict, save_dir)
        else:
            raise ValueError(f"Unknown model: {model_name}")
