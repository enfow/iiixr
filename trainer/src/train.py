import argparse
import os
import random
import time

from result_parser.plot import ResultParser
from schema.config import ModelEmbeddingType
from trainer.trainer_factory import TrainerFactory
from util.config import load_config_from_yaml, merge_configs


def main():
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model."
    )

    # Add config file argument
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to YAML configuration file",
    )

    # Keep all existing arguments without default values
    parser.add_argument("--env", type=str, help="Gymnasium environment name")
    parser.add_argument(
        "--model",
        type=str,
        choices=TrainerFactory.selectable_models,
        help="Model to train",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=ModelEmbeddingType,
        help="Embedding type",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, help="Maximum steps per episode")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension for networks")
    parser.add_argument("--n_layers", type=int, help="Number of layers for networks")
    parser.add_argument("--buffer_size", type=int, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--save_dir", type=str, help="Directory to save results")
    parser.add_argument("--eval", type=bool, help="Whether to evaluate the model")
    parser.add_argument("--eval_period", type=int, help="Evaluation period")
    parser.add_argument(
        "--eval_episodes", type=int, help="Number of episodes to evaluate"
    )
    parser.add_argument("--device", type=str, help="Device to use (cpu/cuda)")
    parser.add_argument("--alpha", type=float, help="Alpha parameter for Rainbow DQN")
    parser.add_argument("--beta_start", type=float, help="Beta start for Rainbow DQN")
    parser.add_argument("--beta_frames", type=int, help="Beta frames for Rainbow DQN")
    parser.add_argument(
        "--target_update_interval",
        type=int,
        help="Target network update interval for Rainbow DQN",
    )
    parser.add_argument(
        "--n_steps", type=int, help="Multi-step learning for Rainbow DQN"
    )
    parser.add_argument(
        "--n_atoms",
        type=int,
        help="Number of atoms for distributional RL in Rainbow DQN",
    )
    parser.add_argument(
        "--v_min",
        type=float,
        help="Minimum value for distributional RL in Rainbow DQN",
    )
    parser.add_argument(
        "--v_max",
        type=float,
        help="Maximum value for distributional RL in Rainbow DQN",
    )
    parser.add_argument("--tau", type=float, help="Tau parameter for SAC/TD3")
    parser.add_argument(
        "--entropy_coef", type=float, help="Entropy coefficient for SAC"
    )
    parser.add_argument("--start_steps", type=int, help="Start steps for SAC/TD3")
    parser.add_argument("--ppo_epochs", type=int, help="PPO epochs")
    parser.add_argument("--clip_eps", type=float, help="PPO clip epsilon")
    parser.add_argument(
        "--n_transactions",
        type=int,
        help="Number of transactions for PPO",
    )
    parser.add_argument(
        "--normalize_advantages",
        type=lambda x: x.lower() == "true",
        help="Whether to normalize advantages in PPO",
    )
    # TD3 specific parameters
    parser.add_argument("--policy_delay", type=int, help="Policy delay for TD3")
    parser.add_argument("--policy_noise", type=float, help="Policy noise for TD3")
    parser.add_argument("--noise_clip", type=float, help="Noise clip for TD3")
    parser.add_argument(
        "--exploration_noise", type=float, help="Exploration noise for TD3"
    )

    args = parser.parse_args()

    # Convert CLI args to dict
    cli_args = vars(args)

    model_config_from_cli = {
        "model": args.model,
        "embedding_type": args.embedding_type,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
    }

    cli_args["model"] = model_config_from_cli

    # Load YAML config if file exists
    yaml_config = {}
    if os.path.exists(args.config):
        try:
            yaml_config = load_config_from_yaml(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Warning: Failed to load config file {args.config}: {e}")
            print("Using default values.")
    else:
        print(f"Config file {args.config} not found. Using default values.")

    # Merge configs with proper precedence: CLI args > YAML config
    config = merge_configs(primary_config=cli_args, secondary_config=yaml_config)

    # Restructure config to handle nested model config
    # Move model-specific parameters to nested structure
    model_params = config.get("model", {})

    config["model"] = model_params

    # Create save directory path
    save_dir = f"{config['save_dir']}/{config['env']}/{config['model']['model']}/{config['model']['embedding_type']}/{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"Training configuration:")
    print(f"  Environment: {config['env']}")
    print(f"  Model: {config['model']['model']}")
    print(f"  Episodes: {config.get('episodes', 'default')}")
    print(f"  Device: {config['device']}")
    print(f"  Save directory: {save_dir}")

    trainer = TrainerFactory(config["env"], config, save_dir=save_dir)
    trainer.train()

    print("Plotting results...")
    result_parser = ResultParser(save_dir)
    result_parser.plot_results()


if __name__ == "__main__":
    main()
