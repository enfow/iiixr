import gc
import json
import os
import pickle
import time
from typing import Any, Dict, List

import gymnasium as gym
import optuna
import torch

from trainer.trainer_factory import TrainerFactory


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

        # Initialize trial log
        self.trial_log_path = f"{save_dir}/result.json"
        self.initialize_trial_log()

    def initialize_trial_log(self):
        """Initialize the trial log file"""
        os.makedirs(os.path.dirname(self.trial_log_path), exist_ok=True)

        initial_log = {
            "hpo_config": self.hpo_config,
            "searchable_params": self.searchable_params,
            "trials": [],
            "best_trial": None,
            "summary": {
                "total_trials": 0,
                "completed_trials": 0,
                "failed_trials": 0,
                "best_score": float("-inf"),
                "start_time": time.strftime("%Y%m%d_%H%M%S"),
            },
        }

        with open(self.trial_log_path, "w") as f:
            json.dump(initial_log, f, indent=2)

    def update_trial_log(self, trial_info: Dict[str, Any]):
        """Update the trial log with new trial information"""
        try:
            # Read existing log
            if os.path.exists(self.trial_log_path):
                with open(self.trial_log_path, "r") as f:
                    log_data = json.load(f)
            else:
                self.initialize_trial_log()
                with open(self.trial_log_path, "r") as f:
                    log_data = json.load(f)

            # Add trial to trials list
            log_data["trials"].append(trial_info)

            # Update summary
            log_data["summary"]["total_trials"] += 1
            if trial_info["status"] == "completed":
                log_data["summary"]["completed_trials"] += 1
                # Update best trial if this one is better
                if trial_info["score"] > log_data["summary"]["best_score"]:
                    log_data["summary"]["best_score"] = trial_info["score"]
                    log_data["best_trial"] = trial_info
            elif trial_info["status"] == "failed":
                log_data["summary"]["failed_trials"] += 1

            # Update timestamp
            log_data["summary"]["last_update"] = time.strftime("%Y%m%d_%H%M%S")

            # Write updated log
            with open(self.trial_log_path, "w") as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            print(f"Failed to update trial log: {e}")

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

        # Initialize trial info
        trial_info = {
            "trial_number": trial.number,
            "model": model,
            "parameters": suggested_params,
            "save_dir": trial_save_dir,
            "start_time": time.strftime("%Y%m%d_%H%M%S"),
            "start_timestamp": time.time(),
            "status": "running",
        }

        try:
            # Train the model
            score = self.train_and_evaluate(trial_config, trial_save_dir, trial)

            # Update trial info with results
            trial_info.update(
                {
                    "status": "completed",
                    "score": score,
                    "end_time": time.strftime("%Y%m%d_%H%M%S"),
                    "end_timestamp": time.time(),
                    "duration_seconds": time.time() - trial_info["start_timestamp"],
                }
            )

            # Update best score if this trial is better
            if score > self.best_score:
                self.best_score = score
                self.best_trial_info = trial_info.copy()
                print(f"New best score: {score:.4f} with trial {trial.number}")

            # Update trial log
            self.update_trial_log(trial_info)

            # Clean up memory after successful trial
            self.cleanup_memory()
            return score

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")

            # Update trial info with failure
            trial_info.update(
                {
                    "status": "failed",
                    "error": str(e),
                    "end_time": time.strftime("%Y%m%d_%H%M%S"),
                    "end_timestamp": time.time(),
                    "duration_seconds": time.time() - trial_info["start_timestamp"],
                }
            )

            # Update trial log
            self.update_trial_log(trial_info)

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
        trainer = TrainerFactory(env, config, save_dir=save_dir)

        # Train the model
        trainer.train()

        # Evaluate the trained model
        eval_result = trainer.evaluate(self.n_eval_episodes)

        # Clean up environment and trainer
        env.close()
        del trainer
        del env
        print("Environment and trainer cleaned up.")

        # Return mean evaluation score
        avg_score = eval_result.avg_score
        std_score = eval_result.std_score

        print(
            f"Trial {trial.number}: {config['model']} - Score: {avg_score:.4f} ± {std_score:.4f}"
        )

        # Report intermediate values for pruning
        trial.report(avg_score, step=config["episodes"])

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return avg_score

    def get_best_trial_info(self):
        """Get information about the best trial"""
        return self.best_trial_info

    def get_trial_log_summary(self):
        """Get summary of all trials from the log"""
        try:
            if os.path.exists(self.trial_log_path):
                with open(self.trial_log_path, "r") as f:
                    log_data = json.load(f)
                return log_data
            else:
                return None
        except Exception as e:
            print(f"Failed to read trial log: {e}")
            return None

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
        gc_after_trial=True,
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

    # Get trial log summary
    trial_log = optimizer.get_trial_log_summary()
    if trial_log:
        summary = trial_log["summary"]
        print(f"\nTrial Log Summary:")
        print(f"  Total trials: {summary['total_trials']}")
        print(f"  Completed trials: {summary['completed_trials']}")
        print(f"  Failed trials: {summary['failed_trials']}")
        print(f"  Best score: {summary['best_score']:.4f}")
        print(f"  Start time: {summary['start_time']}")
        print(f"  Last update: {summary.get('last_update', 'N/A')}")

    # Trial log is already being maintained by the optimizer
    print(f"Trial log saved to: {results_dir}/result.json")

    print(f"\nResults saved to: {results_dir}")

    return study
