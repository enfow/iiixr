import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer


@dataclass
class EvalConfig:
    eval_episodes: int = 10
    eval_render: bool = False
    eval_save_dir: str = "results/eval"


def load_config_from_results(results_path):
    """Load configuration from the results directory."""
    config_file = os.path.join(results_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {config_file}")


def create_trainer_from_results(results_path, env):
    """Create a trainer instance based on the results directory structure."""
    results_path = Path(results_path)
    trainer_type = results_path.name

    config = load_config_from_results(results_path)

    if trainer_type == "ppo":
        trainer = PPOTrainer(env, config, save_dir=str(results_path))
    elif trainer_type == "sac":
        trainer = SACTrainer(env, config, save_dir=str(results_path))
    elif trainer_type == "rainbow_dqn":
        trainer = RainbowDQNTrainer(env, config, save_dir=str(results_path))
    elif trainer_type == "discrete_sac":
        trainer = DiscreteSACTrainer(env, config, save_dir=str(results_path))
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

    return trainer


def evaluate_model(results_path, env_name, episodes=10, render=False):
    """Evaluate a trained model."""
    print(f"Evaluating model from: {results_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {episodes}")

    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)

    trainer = create_trainer_from_results(results_path, env)

    # Run evaluation
    trainer.load_model()
    eval_results = trainer.evaluate(episodes=episodes)

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
        "--env",
        type=str,
        required=True,
        help="Gymnasium environment name (e.g., 'LunarLander-v2', 'CartPole-v1')",
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save evaluation results (JSON format)",
    )

    args = parser.parse_args()

    # Validate results path
    if not os.path.exists(args.results_path):
        print(f"Error: Results path does not exist: {args.results_path}")
        return

    # Run evaluation
    results = evaluate_model(
        results_path=args.results_path,
        env_name=args.env,
        episodes=args.episodes,
        render=args.render,
    )

    # Save results if output file specified
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
