import argparse
import os
import time
from typing import Any, Dict, List

import gymnasium as gym
import yaml

from hpo.optimizer import run_optuna_optimization
from trainer.trainer_factory import TrainerFactory


def load_hpo_config(config_path: str = "hpo_config.yaml") -> Dict[str, Any]:
    """Load HPO configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"HPO config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def parse_categorical_arg(value_str: str) -> List:
    """Parse categorical argument from comma-separated string"""
    if not value_str:
        return []

    result = []
    for x in value_str.split(","):
        x = x.strip().lower()
        if x in ["true", "false"]:
            result.append(x == "true")
        elif x.isdigit():
            result.append(int(x))
        else:
            result.append(float(x))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for RL training with Optuna."
    )

    # HPO configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="hpo_config.yaml",
        help="Path to HPO configuration YAML file",
    )

    # Environment and device settings
    parser.add_argument(
        "--env",
        type=str,
        default="LunarLander-v3",
        help="Gymnasium environment name (overrides config file)",
    )
    parser.add_argument(
        "--model", type=str, default="ppo", help="Model to use (overrides config file)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda) (overrides config file)",
    )

    # Training parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes for training (overrides config file)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps per episode (overrides config file)",
    )

    # HPO-specific arguments
    parser.add_argument(
        "--hpo_n_trials",
        type=int,
        default=2,
        help="Number of Optuna trials (overrides config file)",
    )
    parser.add_argument(
        "--hpo_study_name",
        type=str,
        required=True,
        help="Name for the Optuna study (overrides config file)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the entire optimization process",
    )

    # Searchable hyperparameters (categorical)
    parser.add_argument(
        "--batch_size",
        type=str,
        default="",
        help="Comma-separated list of batch sizes to search (e.g., '64,128,256,512')",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="",
        help="Comma-separated list of gamma values to search (e.g., '0.95,0.97,0.99,0.995')",
    )
    parser.add_argument(
        "--hidden_dim",
        type=str,
        default="",
        help="Comma-separated list of hidden dimensions to search (e.g., '64,128,256,512')",
    )
    parser.add_argument(
        "--n_layers",
        type=str,
        default="",
        help="Comma-separated list of layer counts to search (e.g., '1,2,3,4')",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=str,
        default="",
        help="Comma-separated list of PPO epochs to search (e.g., '4,6,8,10,12')",
    )
    parser.add_argument(
        "--clip_eps",
        type=str,
        default="",
        help="Comma-separated list of clip epsilon values to search (e.g., '0.1,0.15,0.2,0.25')",
    )
    parser.add_argument(
        "--start_steps",
        type=str,
        default="",
        help="Comma-separated list of start steps to search (e.g., '500,1000,2000')",
    )
    parser.add_argument(
        "--entropy_coef",
        type=str,
        default="",
        help="Comma-separated list of entropy coefficients to search (e.g., '0.01,0.1,0.5,1.0')",
    )
    parser.add_argument(
        "--target_update",
        type=str,
        default="",
        help="Comma-separated list of target update frequencies to search (e.g., '5,10,20,50')",
    )
    parser.add_argument(
        "--n_transactions",
        type=str,
        default="",
        help="Comma-separated list of n_transactions values to search (e.g., '500,1000,2000')",
    )
    parser.add_argument(
        "--normalize_advantages",
        type=str,
        default="",
        help="Comma-separated list of normalize_advantages values to search (e.g., 'true,false')",
    )

    args = parser.parse_args()

    # Load HPO configuration from YAML file
    hpo_config = load_hpo_config(args.config)

    # Override config with command line arguments
    if args.env is not None:
        hpo_config["env"] = args.env
    if args.device is not None:
        hpo_config["device"] = args.device
    if args.model is not None:
        hpo_config["model"] = args.model
    if args.episodes is not None:
        hpo_config["episodes"] = args.episodes
    if args.max_steps is not None:
        hpo_config["max_steps"] = args.max_steps
    if args.hpo_n_trials is not None:
        hpo_config["hpo_n_trials"] = args.hpo_n_trials
    if args.hpo_study_name is not None:
        hpo_config["hpo_study_name"] = args.hpo_study_name

    # Get default search spaces from config file
    config_searchable = hpo_config.get("searchable_hyperparameters", {})

    # Build searchable parameters dictionary (only include if user provided CLI value)
    searchable_params = {}
    if args.batch_size:
        searchable_params["batch_size"] = parse_categorical_arg(args.batch_size)
    elif "batch_size" in config_searchable:
        searchable_params["batch_size"] = config_searchable["batch_size"]

    if args.gamma:
        searchable_params["gamma"] = parse_categorical_arg(args.gamma)
    elif "gamma" in config_searchable:
        searchable_params["gamma"] = config_searchable["gamma"]

    if args.hidden_dim:
        searchable_params["hidden_dim"] = parse_categorical_arg(args.hidden_dim)
    elif "hidden_dim" in config_searchable:
        searchable_params["hidden_dim"] = config_searchable["hidden_dim"]

    if args.n_layers:
        searchable_params["n_layers"] = parse_categorical_arg(args.n_layers)
    elif "n_layers" in config_searchable:
        searchable_params["n_layers"] = config_searchable["n_layers"]

    if args.ppo_epochs:
        searchable_params["ppo_epochs"] = parse_categorical_arg(args.ppo_epochs)
    elif "ppo_epochs" in config_searchable:
        searchable_params["ppo_epochs"] = config_searchable["ppo_epochs"]

    if args.clip_eps:
        searchable_params["clip_eps"] = parse_categorical_arg(args.clip_eps)
    elif "clip_eps" in config_searchable:
        searchable_params["clip_eps"] = config_searchable["clip_eps"]

    if args.start_steps:
        searchable_params["start_steps"] = parse_categorical_arg(args.start_steps)
    elif "start_steps" in config_searchable:
        searchable_params["start_steps"] = config_searchable["start_steps"]

    if args.entropy_coef:
        searchable_params["entropy_coef"] = parse_categorical_arg(args.entropy_coef)
    elif "entropy_coef" in config_searchable:
        searchable_params["entropy_coef"] = config_searchable["entropy_coef"]

    if args.target_update:
        searchable_params["target_update"] = parse_categorical_arg(args.target_update)
    elif "target_update" in config_searchable:
        searchable_params["target_update"] = config_searchable["target_update"]

    if args.n_transactions:
        searchable_params["n_transactions"] = parse_categorical_arg(args.n_transactions)
    elif "n_transactions" in config_searchable:
        searchable_params["n_transactions"] = config_searchable["n_transactions"]

    if args.normalize_advantages:
        searchable_params["normalize_advantages"] = parse_categorical_arg(
            args.normalize_advantages
        )
    elif "normalize_advantages" in config_searchable:
        searchable_params["normalize_advantages"] = config_searchable[
            "normalize_advantages"
        ]

    n_possible_combinations = 1
    for param_name, param_values in searchable_params.items():
        n_possible_combinations *= len(param_values)

    print("HPO Config:")
    print(hpo_config)
    print("Searchable Parameters:")
    print(searchable_params)
    print("Number of possible combinations:", n_possible_combinations)
    hpo_config["hpo_n_trials"] = min(
        hpo_config["hpo_n_trials"], n_possible_combinations
    )
    print("Number of trials:", hpo_config["hpo_n_trials"])

    print("Study name:", hpo_config["hpo_study_name"])
    print("=" * 50)
    proceed = input("Proceed? (y/n): ")
    if proceed != "y":
        print("Exiting...")
        exit()

    save_dir = f"results/hpo/{hpo_config['env']}/{hpo_config['model']}/{hpo_config['hpo_study_name']}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Run optimization
    study = run_optuna_optimization(
        hpo_config=hpo_config,
        searchable_params=searchable_params,
        n_trials=hpo_config["hpo_n_trials"],
        timeout=args.timeout,
        study_name=hpo_config["hpo_study_name"],
        save_dir=save_dir,
    )

    # Train final model with best parameters
    print("\n" + "=" * 50)
    print("Training final model with best parameters...")

    final_config = hpo_config.copy()
    final_config.update(study.best_params)
    final_config["episodes"] = round(
        hpo_config["episodes"] * 1.2
    )  # Train longer for final model

    # Create final save directory
    final_save_dir = f"{save_dir}/final_model"

    if final_config["env"] == "BipedalWalker-v3":
        env = gym.make(final_config["env"], hardcore=True)
    else:
        env = gym.make(final_config["env"])

    # Train final model
    trainer = TrainerFactory(env, final_config, save_dir=final_save_dir)

    trainer.train()
    print(f"Final model saved to: {final_save_dir}")


if __name__ == "__main__":
    main()
