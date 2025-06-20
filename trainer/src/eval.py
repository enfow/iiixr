import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym

from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer


@dataclass
class EvalConfig:
    results_path: str = None
    eval_episodes: int = 10
    eval_render: bool = False
    eval_save_dir: str = None

    @classmethod
    def from_dict(cls, args):
        return cls(
            results_path=args.results_path,
            eval_episodes=args.episodes,
            eval_render=args.render,
            eval_save_dir=f"{args.results_path}/eval.json",
        )


def load_config_from_results(results_path):
    """Load configuration from the results directory."""
    config_file = os.path.join(results_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {config_file}")


def create_trainer_from_results(env, train_config, eval_config):
    """Create a trainer instance based on the results directory structure."""

    model_type = train_config["model"]

    if model_type == "ppo":
        trainer = PPOTrainer(env, train_config, eval_config.results_path)
    elif model_type == "sac":
        trainer = SACTrainer(env, train_config, eval_config.results_path)
    elif model_type == "rainbow_dqn":
        trainer = RainbowDQNTrainer(env, train_config, eval_config.results_path)
    elif model_type == "discrete_sac":
        trainer = DiscreteSACTrainer(env, train_config, eval_config.results_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return trainer


def evaluate_model(eval_config):
    """Evaluate a trained model."""
    train_config = load_config_from_results(eval_config.results_path)

    print(f"Evaluating model from: {eval_config.results_path}")
    print(f"Environment: {train_config['env_name']}")
    print(f"Episodes: {eval_config.eval_episodes}")

    # Create environment
    env = gym.make(
        train_config["env_name"],
        render_mode="human" if eval_config.eval_render else None,
    )

    trainer = create_trainer_from_results(env, train_config, eval_config)

    # Run evaluation
    trainer.load_model()
    eval_results = trainer.evaluate(episodes=eval_config.eval_episodes)

    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained reinforcement learning models"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the results directory containing the trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation (default: 10)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )

    args = parser.parse_args()

    eval_config = EvalConfig.from_dict(args)

    # Run evaluation
    results = evaluate_model(eval_config)

    # Save results if output file specified
    if eval_config.eval_save_dir and results:
        with open(eval_config.eval_save_dir, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to: {eval_config.eval_save_dir}")


if __name__ == "__main__":
    main()
