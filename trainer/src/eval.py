import argparse
import json
import os

import gymnasium as gym

from schema.config import EvalConfig
from trainer.trainer_factory import TrainerFactory


def load_config_from_result(result_path):
    """Load configuration from the results directory."""
    config_file = os.path.join(result_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {config_file}")


def evaluate_model(eval_config):
    """Evaluate a trained model."""
    train_config = load_config_from_result(eval_config.result_path)

    print(f"Evaluating model from: {eval_config.result_path}")
    print(f"Environment: {train_config['env_name']}")
    print(f"Episodes: {eval_config.eval_episodes}")

    # Create environment
    env = gym.make(
        train_config["env_name"],
        render_mode="human" if eval_config.eval_render else None,
    )

    trainer = TrainerFactory(env, train_config, eval_config.result_path)

    # Run evaluation
    trainer.load_model()
    eval_results = trainer.evaluate(episodes=eval_config.eval_episodes)

    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained reinforcement learning models"
    )
    parser.add_argument(
        "--result_path",
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
