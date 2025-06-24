import argparse
import gc
import os
import pickle
import time
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import optuna
import torch
import yaml

from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer


def load_hpo_config(config_path: str = "hpo_config.yaml") -> Dict[str, Any]:
    """Load HPO configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"HPO config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class OptunaRLOptimizer:
    def __init__(
        self,
        hpo_config: Dict[str, Any],
        searchable_params: Dict[str, List],
        save_dir: str,
        n_eval_episodes: int = 10,
    ):
        self.hpo_config = hpo_config
        self.searchable_params = searchable_params
        self.save_dir = save_dir
        self.n_eval_episodes = n_eval_episodes
        self.best_score = float("-inf")
        self.best_trial_info = None

    def objective(self, trial):
        """
        Optuna objective function for RL hyperparameter optimization
        """
        # Use model from config (fixed, not searched)
        model = self.hpo_config["model"]

        # Suggest searchable hyperparameters
        suggested_params = {}
        for param_name, param_values in self.searchable_params.items():
            suggested_params[param_name] = trial.suggest_categorical(
                param_name, param_values
            )

        # Create config for this trial
        trial_config = self.hpo_config.copy()
        trial_config.update({"model": model, **suggested_params})

        # Create unique save directory for this trial
        trial_save_dir = f"{self.save_dir}/trial_{trial.number}_{model}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(trial_save_dir, exist_ok=True)

        try:
            # Train the model
            score = self.train_and_evaluate(trial_config, trial_save_dir, trial)

            # Update best score if this trial is better
            if score > self.best_score:
                self.best_score = score
                self.best_trial_info = {
                    "trial_number": trial.number,
                    "score": score,
                    "model": model,
                    "parameters": suggested_params,
                    "save_dir": trial_save_dir,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                }
                print(f"New best score: {score:.4f} with trial {trial.number}")

            # Clean up memory after successful trial
            self.cleanup_memory()
            return score

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Clean up failed trial directory
            self.cleanup_trial_dir(trial_save_dir)
            # Clean up memory after failed trial
            self.cleanup_memory()
            # Return a very low score for failed trials
            return float("-inf")

    def train_and_evaluate(self, config: Dict[str, Any], save_dir: str, trial) -> float:
        """
        Train the model and return evaluation score
        """
        # Create environment
        if config["env"] == "BipedalWalker-v3":
            env = gym.make(config["env"], hardcore=True)
        else:
            env = gym.make(config["env"])

        # Select trainer based on model
        if config["model"] == "ppo":
            trainer = PPOTrainer(env, config, save_dir=save_dir)
        elif config["model"] == "sac":
            trainer = SACTrainer(env, config, save_dir=save_dir)
        elif config["model"] == "rainbow_dqn":
            trainer = RainbowDQNTrainer(env, config, save_dir=save_dir)
        elif config["model"] == "discrete_sac":
            trainer = DiscreteSACTrainer(env, config, save_dir=save_dir)
        else:
            raise ValueError(f"Unknown model: {config['model']}")

        # Train the model
        trainer.train()

        # Evaluate the trained model
        eval_scores = []
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done and step < config["max_steps"]:
                # Get action from trained agent
                action = trainer.get_action(obs, evaluate=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1

            eval_scores.append(episode_reward)

        # Clean up environment and trainer
        env.close()
        del trainer
        del env

        # Return mean evaluation score
        mean_score = np.mean(eval_scores)
        std_score = np.std(eval_scores)

        print(
            f"Trial {trial.number}: {config['model']} - Score: {mean_score:.4f} ± {std_score:.4f}"
        )

        # Report intermediate values for pruning
        trial.report(mean_score, step=config["episodes"])

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return mean_score

    def get_best_trial_info(self):
        """Get information about the best trial"""
        return self.best_trial_info

    def cleanup_trial_dir(self, save_dir: str):
        """Clean up trial directory to save space"""
        try:
            import shutil

            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        except Exception as e:
            print(f"Failed to cleanup {save_dir}: {e}")

    def cleanup_memory(self):
        """Clean up memory after each trial to prevent OOM errors"""
        try:
            # Clear GPU memory if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(
                    f"Cleared GPU memory. Current GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                )

            # Run garbage collection
            gc.collect()

        except Exception as e:
            print(f"Memory cleanup failed: {e}")


def run_optuna_optimization(
    hpo_config: Dict[str, Any],
    searchable_params: Dict[str, List],
    n_trials: int = 50,
    timeout: int = None,
    study_name: str = "rl_optimization",
    save_dir: str = "results",
):
    """
    Run Optuna hyperparameter optimization
    """
    optimizer = OptunaRLOptimizer(
        hpo_config, searchable_params, save_dir, n_eval_episodes=5
    )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=hpo_config.get("seed", 42), n_startup_trials=10, n_ei_candidates=24
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=10
        ),
    )

    # Optimization callback
    def callback(study, trial):
        if trial.number % 5 == 0:
            print(f"\nTrial {trial.number} completed")
            print(f"Current best score: {study.best_value:.4f}")
            print(f"Best parameters so far:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            print("-" * 50)

    # Run optimization
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    print(f"Environment: {hpo_config['env']}")
    print(f"Base episodes per trial: {hpo_config['episodes']}")
    print("Searchable hyperparameters:")
    for param, values in searchable_params.items():
        print(f"  {param}: {values}")
    print("=" * 50)

    study.optimize(
        optimizer.objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback],
        show_progress_bar=True,
    )

    # Print final results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETED!")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save study results
    results_dir = f"{save_dir}/{study_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    # Save study object
    with open(f"{results_dir}/study.pkl", "wb") as f:
        pickle.dump(study, f)

    # Save best parameters
    with open(f"{results_dir}/best_params.txt", "w") as f:
        f.write(f"Best score: {study.best_value:.4f}\n")
        f.write("Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    # Save trials dataframe
    df = study.trials_dataframe()
    df.to_csv(f"{results_dir}/trials.csv", index=False)

    # Save best trial information in result.json
    best_trial_info = optimizer.get_best_trial_info()
    if best_trial_info:
        import json

        with open(f"{results_dir}/result.json", "w") as f:
            json.dump(best_trial_info, f, indent=2)
        print(f"Best trial info saved to: {results_dir}/result.json")

    print(f"\nResults saved to: {results_dir}")

    return study


def parse_categorical_arg(value_str: str) -> List:
    """Parse categorical argument from comma-separated string"""
    if not value_str:
        return []
    return [
        int(x.strip()) if x.strip().isdigit() else float(x.strip())
        for x in value_str.split(",")
    ]


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

    n_possible_combinations = (
        len(searchable_params["batch_size"])
        * len(searchable_params["gamma"])
        * len(searchable_params["hidden_dim"])
        * len(searchable_params["n_layers"])
        * len(searchable_params["ppo_epochs"])
        * len(searchable_params["clip_eps"])
    )

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
    final_config["episodes"] = (
        hpo_config["episodes"] * 2
    )  # Train longer for final model

    # Create final save directory
    final_save_dir = f"{save_dir}/final_model"

    if final_config["env"] == "BipedalWalker-v3":
        env = gym.make(final_config["env"], hardcore=True)
    else:
        env = gym.make(final_config["env"])

    # Train final model
    if final_config["model"] == "ppo":
        trainer = PPOTrainer(env, final_config, save_dir=final_save_dir)
    elif final_config["model"] == "sac":
        trainer = SACTrainer(env, final_config, save_dir=final_save_dir)
    elif final_config["model"] == "rainbow_dqn":
        trainer = RainbowDQNTrainer(env, final_config, save_dir=final_save_dir)
    elif final_config["model"] == "discrete_sac":
        trainer = DiscreteSACTrainer(env, final_config, save_dir=final_save_dir)
    else:
        raise ValueError(f"Unknown model: {final_config['model']}")

    trainer.train()
    print(f"Final model saved to: {final_save_dir}")


if __name__ == "__main__":
    main()
