import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import schema classes
from schema.result import (C51UpdateLoss, DDQNUpdateLoss,
                           DiscreteSACUpdateLoss, EvalResult, PPOUpdateLoss,
                           RainbowDQNUpdateLoss, SACUpdateLoss,
                           SingleEpisodeResult, TD3FORKUpdateLoss,
                           TD3UpdateLoss, TotalTrainResult)


class ResultParser:
    """Parser for metrics.jsonl files containing training and evaluation results"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.metrics_file_path = self.results_dir / "metrics.jsonl"
        self.config_file_path = self.results_dir / "config.json"

        # Load configuration
        self.config = self._load_config()
        self.model_config = self.config.get("model", {})

        # Parse results
        self.train_results: List[SingleEpisodeResult] = []
        self.eval_results: List[EvalResult] = []
        self.total_train_result: Optional[TotalTrainResult] = None

        self.model_name = self.model_config.get("model", "unknown")
        self.env_name = self.config.get("env", "unknown")

        self.training_plot_path = (
            self.results_dir / f"{self.model_name}_{self.env_name}_training.png"
        )
        self.eval_plot_path = (
            self.results_dir / f"{self.model_name}_{self.env_name}_evaluation.png"
        )
        self.combined_plot_path = (
            self.results_dir / f"{self.model_name}_{self.env_name}_combined.png"
        )
        self.loss_plot_path = (
            self.results_dir / f"{self.model_name}_{self.env_name}_loss.png"
        )

        self.parse()
        self.print_summary()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        if not self.config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file_path}")

        with open(self.config_file_path, "r") as f:
            config = json.load(f)

        return config

    def _create_loss_object(self, loss_data: Dict[str, Any], model_name: str):
        """Create appropriate loss object based on model type"""
        if model_name == "ppo":
            return PPOUpdateLoss(**loss_data)
        elif model_name == "discrete_ppo":
            return PPOUpdateLoss(**loss_data)  # Uses same loss as PPO
        elif model_name == "ppo_seq":
            return PPOUpdateLoss(**loss_data)
        elif model_name == "sac":
            return SACUpdateLoss(**loss_data)
        elif model_name == "sac_v2":
            return SACUpdateLoss(**loss_data)  # Uses same loss as SAC
        elif model_name == "discrete_sac":
            return DiscreteSACUpdateLoss(**loss_data)
        elif model_name == "rainbow_dqn":
            return RainbowDQNUpdateLoss(**loss_data)
        elif model_name == "td3":
            return TD3UpdateLoss(**loss_data)
        elif model_name == "td3_seq":
            return TD3UpdateLoss(**loss_data)  # Uses same loss as TD3
        elif model_name == "c51":
            return C51UpdateLoss(**loss_data)
        elif model_name == "ddqn":
            return DDQNUpdateLoss(**loss_data)
        elif model_name == "td3_fork":
            return TD3FORKUpdateLoss(**loss_data)
        elif model_name == "sac_seq":
            return SACUpdateLoss(**loss_data)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def parse(self) -> Tuple[List[SingleEpisodeResult], List[EvalResult]]:
        """Parse the metrics.jsonl file and extract training and evaluation results"""

        if not self.metrics_file_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file_path}")

        model_name = self.model_name
        total_train_result = TotalTrainResult.initialize()

        with open(self.metrics_file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    self._parse_line(data, line_num, model_name, total_train_result)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue

        self.total_train_result = total_train_result
        print("train_results:", len(self.train_results))
        print("eval_results:", len(self.eval_results))
        return self.train_results, self.eval_results

    def _parse_line(
        self,
        data: Dict[str, Any],
        line_num: int,
        model_name: str,
        total_train_result: TotalTrainResult,
    ):
        """Parse a single line of JSON data"""

        # Check if this is a training result (has 'ep' key)
        if "ep" in data:
            # Create loss objects from loss_details
            loss_details = []
            for loss_data in data.get("loss_details", []):
                loss_obj = self._create_loss_object(loss_data, model_name)
                loss_details.append(loss_obj)

            # Create SingleEpisodeResult
            episode_result = SingleEpisodeResult(
                episode_number=data["ep"],
                episode_total_reward=data["total_rewards"],  # Convert to list format
                episode_steps=data["episode_steps"],
                episode_losses=loss_details,
                episode_elapsed_time=data["episode_elapsed_time"],
            )

            self.train_results.append(episode_result)
            total_train_result.update(episode_result)

        # Check if this is an evaluation result (has 'train_episode_number' key)
        elif "train_episode_number" in data:
            eval_result = EvalResult.from_dict(data)
            self.eval_results.append(eval_result)

    def plot_training_results(
        self, save_path: Optional[str] = None, show_plot: bool = False
    ):
        """Plot training results: episode vs total_reward"""

        if save_path is None:
            save_path = self.training_plot_path

        if not self.train_results:
            return

        episodes = [result.episode_number for result in self.train_results]
        total_rewards = [result.episode_total_reward for result in self.train_results]

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, total_rewards, "b-", linewidth=2, label="Total Rewards")
        plt.scatter(episodes, total_rewards, color="blue", alpha=0.6, s=30)

        # Add moving average
        if len(total_rewards) > 10:
            window_size = min(10, len(total_rewards) // 4)
            moving_avg = (
                pd.Series(total_rewards).rolling(window=window_size, center=True).mean()
            )
            plt.plot(
                episodes,
                moving_avg,
                "r--",
                linewidth=2,
                label=f"Moving Average (window={window_size})",
            )

        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Rewards", fontsize=12)

        # Add model and environment info to title
        model_name = self.model_name
        env_name = self.config.get("env", "Unknown")
        plt.title(
            f"Training Progress: {model_name.upper()} on {env_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add some statistics
        avg_reward = np.mean(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)

        stats_text = (
            f"Avg: {avg_reward:.2f}\nMax: {max_reward:.2f}\nMin: {min_reward:.2f}"
        )
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path is not None:
            print(f"Saving training plot to {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print("Done")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_evaluation_results(
        self, save_path: Optional[str] = None, show_plot: bool = False
    ):
        """Plot evaluation results: train episode vs scores (max, min, avg)"""

        if save_path is None:
            save_path = self.eval_plot_path

        if not self.eval_results:
            return

        train_episodes = [result.train_episode_number for result in self.eval_results]
        avg_scores = [result.avg_score for result in self.eval_results]
        min_scores = [result.min_score for result in self.eval_results]
        max_scores = [result.max_score for result in self.eval_results]

        plt.figure(figsize=(12, 8))

        # Fill area between max and min with light blue
        plt.fill_between(
            train_episodes,
            min_scores,
            max_scores,
            alpha=0.2,
            color="lightblue",
            label="Score Range",
        )

        # Plot three separate lines: max, min, and avg (all in blue)
        plt.plot(
            train_episodes,
            max_scores,
            linewidth=2,
            label="Max Score",
            alpha=0.3,
            color="blue",
        )

        plt.plot(
            train_episodes,
            min_scores,
            linewidth=2,
            label="Min Score",
            alpha=0.3,
            color="blue",
        )

        plt.plot(
            train_episodes,
            avg_scores,
            linewidth=4,
            label="Average Score",
            alpha=0.7,
            color="blue",
        )

        plt.xlabel("Training Episode", fontsize=12)
        plt.ylabel("Score", fontsize=12)

        # Add model and environment info to title
        model_name = self.model_name
        env_name = self.config.get("env", "Unknown")
        plt.title(
            f"Evaluation Results: {model_name.upper()} on {env_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add statistics
        best_avg_score = np.max(avg_scores)
        worst_avg_score = np.min(avg_scores)
        final_avg_score = avg_scores[-1] if avg_scores else 0
        best_max_score = np.max(max_scores)
        worst_min_score = np.min(min_scores)

        stats_text = (
            f"Best Avg: {best_avg_score:.2f}\n"
            f"Worst Avg: {worst_avg_score:.2f}\n"
            f"Final Avg: {final_avg_score:.2f}\n"
            f"Best Max: {best_max_score:.2f}\n"
            f"Worst Min: {worst_min_score:.2f}"
        )
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_loss_results(
        self, save_path: Optional[str] = None, show_plot: bool = False
    ):
        """Plot training loss results: episode vs total_loss with stacked loss components"""

        if save_path is None:
            save_path = self.loss_plot_path

        if not self.train_results:
            return

        episodes = [result.episode_number for result in self.train_results]
        total_losses = [result.episode_total_loss for result in self.train_results]

        # Extract individual loss components
        loss_components = self._extract_loss_components()

        if not loss_components:
            # Fallback to simple total loss plot if no components found
            self._plot_simple_loss(episodes, total_losses, save_path, show_plot)
            return

        # Create stacked area plot
        plt.figure(figsize=(14, 10))

        # Create stacked area chart
        component_names = list(loss_components.keys())
        component_data = list(loss_components.values())

        # Plot stacked area
        plt.stackplot(episodes, component_data, labels=component_names, alpha=0.8)

        # Add total loss line on top
        plt.plot(
            episodes, total_losses, "k-", linewidth=2, label="Total Loss", alpha=0.9
        )
        plt.scatter(episodes, total_losses, color="black", alpha=0.7, s=20)

        # Add moving average for total loss
        if len(total_losses) > 10:
            window_size = min(10, len(total_losses) // 4)
            moving_avg = (
                pd.Series(total_losses).rolling(window=window_size, center=True).mean()
            )
            plt.plot(
                episodes,
                moving_avg,
                "k--",
                linewidth=2,
                label=f"Total Loss Moving Avg (window={window_size})",
                alpha=0.8,
            )

        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Loss", fontsize=12)

        # Add model and environment info to title
        model_name = self.model_name
        env_name = self.config.get("env", "Unknown")
        plt.title(
            f"Training Loss Components: {model_name.upper()} on {env_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add statistics
        avg_loss = np.mean(total_losses)
        max_loss = np.max(total_losses)
        min_loss = np.min(total_losses)

        stats_text = f"Total Loss:\nAvg: {avg_loss:.2f}\nMax: {max_loss:.2f}\nMin: {min_loss:.2f}"
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _extract_loss_components(self) -> Dict[str, List[float]]:
        """Extract individual loss components from training results"""
        if not self.train_results:
            return {}

        # Initialize component lists
        loss_per_episode = {}

        zero_episode_indices = []
        for result in self.train_results:
            if not result.episode_losses:
                zero_episode_indices.append(result.episode_number)
                continue

            episode_losses = {}

            for loss_obj in result.episode_losses:
                # Get all attributes of the loss object
                loss_dict = loss_obj.to_dict()

                # Add each component
                for loss_name, loss_value in loss_dict.items():
                    if loss_name not in episode_losses:
                        episode_losses[loss_name] = []
                    episode_losses[loss_name].append(loss_value)

            for loss_name in episode_losses.keys():
                if loss_name not in loss_per_episode:
                    loss_per_episode[loss_name] = []
                loss_per_episode[loss_name].append(np.mean(episode_losses[loss_name]))

        for loss_name in loss_per_episode.keys():
            # insert 0.0 for zero_episode_indices
            for zero_episode_index in zero_episode_indices:
                loss_per_episode[loss_name].insert(zero_episode_index, 0.0)

        # Ensure all component lists have the same length
        max_length = (
            max(len(comp_list) for comp_list in loss_per_episode.values())
            if loss_per_episode
            else 0
        )

        for loss_name in loss_per_episode:
            # Pad shorter lists with zeros
            while len(loss_per_episode[loss_name]) < max_length:
                loss_per_episode[loss_name].append(0.0)

        return loss_per_episode

    def _plot_simple_loss(
        self,
        episodes: List[int],
        total_losses: List[float],
        save_path: Optional[str],
        show_plot: bool,
    ):
        """Fallback method for simple total loss plot"""
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, total_losses, "r-", linewidth=2, label="Total Loss")
        plt.scatter(episodes, total_losses, color="red", alpha=0.6, s=30)

        # Add moving average
        if len(total_losses) > 10:
            window_size = min(10, len(total_losses) // 4)
            moving_avg = (
                pd.Series(total_losses).rolling(window=window_size, center=True).mean()
            )
            plt.plot(
                episodes,
                moving_avg,
                "g--",
                linewidth=2,
                label=f"Moving Average (window={window_size})",
            )

        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Loss", fontsize=12)

        # Add model and environment info to title
        model_name = self.model_name
        env_name = self.config.get("env", "Unknown")
        plt.title(
            f"Training Loss: {model_name.upper()} on {env_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add some statistics
        avg_loss = np.mean(total_losses)
        max_loss = np.max(total_losses)
        min_loss = np.min(total_losses)

        stats_text = f"Avg: {avg_loss:.2f}\nMax: {max_loss:.2f}\nMin: {min_loss:.2f}"
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_combined_results(
        self, save_path: Optional[str] = None, show_plot: bool = False
    ):
        """Plot both training and evaluation results together"""

        if save_path is None:
            save_path = self.combined_plot_path

        if not self.train_results and not self.eval_results:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Training results
        if self.train_results:
            episodes = [result.episode_number for result in self.train_results]
            total_rewards = [
                result.episode_total_reward for result in self.train_results
            ]

            ax1.plot(episodes, total_rewards, "b-", linewidth=2, label="Total Rewards")
            ax1.scatter(episodes, total_rewards, color="blue", alpha=0.6, s=30)

            if len(total_rewards) > 10:
                window_size = min(10, len(total_rewards) // 4)
                moving_avg = (
                    pd.Series(total_rewards)
                    .rolling(window=window_size, center=True)
                    .mean()
                )
                ax1.plot(
                    episodes,
                    moving_avg,
                    "r--",
                    linewidth=2,
                    label=f"Moving Average (window={window_size})",
                )

        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel("Total Rewards", fontsize=12)
        ax1.set_title("Training Progress", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Evaluation results
        if self.eval_results:
            train_episodes = [
                result.train_episode_number for result in self.eval_results
            ]
            avg_scores = [result.avg_score for result in self.eval_results]
            std_scores = [result.std_score for result in self.eval_results]

            ax2.errorbar(
                train_episodes,
                avg_scores,
                yerr=std_scores,
                fmt="o-",
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8,
                label="Average Score ± Std",
            )

        ax2.set_xlabel("Training Episode", fontsize=12)
        ax2.set_ylabel("Average Score", fontsize=12)
        ax2.set_title("Evaluation Results", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add overall title with model and environment info
        model_name = self.model_name
        env_name = self.config.get("env", "Unknown")
        fig.suptitle(
            f"{model_name.upper()} Training Results on {env_name}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the parsed results"""

        summary = {
            "model": self.model_name,
            "environment": self.config.get("env", "Unknown"),
            "total_training_episodes": len(self.train_results),
            "total_evaluation_runs": len(self.eval_results),
            "config": self.config,
        }

        if self.train_results:
            total_rewards = [
                result.episode_total_reward for result in self.train_results
            ]
            summary.update(
                {
                    "training_avg_reward": np.mean(total_rewards),
                    "training_max_reward": np.max(total_rewards),
                    "training_min_reward": np.min(total_rewards),
                    "training_std_reward": np.std(total_rewards),
                    "final_training_reward": total_rewards[-1] if total_rewards else 0,
                }
            )

        if self.eval_results:
            avg_scores = [result.avg_score for result in self.eval_results]
            summary.update(
                {
                    "eval_avg_score": np.mean(avg_scores),
                    "eval_max_score": np.max(avg_scores),
                    "eval_min_score": np.min(avg_scores),
                    "eval_std_score": np.std(avg_scores),
                    "final_eval_score": avg_scores[-1] if avg_scores else 0,
                }
            )

        return summary

    def print_summary(self):
        """Print a summary of the parsed results"""

        summary = self.get_summary_stats()

        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Model: {summary['model'].upper()}")
        print(f"Environment: {summary['environment']}")
        print(f"Training Episodes: {summary['total_training_episodes']}")
        print(f"Evaluation Runs: {summary['total_evaluation_runs']}")

        # Print key config parameters
        config = summary["config"]
        print(f"\nCONFIGURATION:")
        print(f"  Learning Rate: {config.get('lr', 'N/A')}")
        print(f"  Gamma: {config.get('gamma', 'N/A')}")
        print(f"  Hidden Dim: {config.get('hidden_dim', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Device: {config.get('device', 'N/A')}")

        if self.train_results:
            print(f"\nTRAINING RESULTS:")
            print(f"  Average Reward: {summary['training_avg_reward']:.2f}")
            print(f"  Max Reward: {summary['training_max_reward']:.2f}")
            print(f"  Min Reward: {summary['training_min_reward']:.2f}")
            print(f"  Final Reward: {summary['final_training_reward']:.2f}")

        if self.eval_results:
            print(f"\nEVALUATION RESULTS:")
            print(f"  Average Score: {summary['eval_avg_score']:.2f}")
            print(f"  Max Score: {summary['eval_max_score']:.2f}")
            print(f"  Min Score: {summary['eval_min_score']:.2f}")
            print(f"  Final Score: {summary['final_eval_score']:.2f}")

        print("=" * 60)

    def plot_results(self):
        print("Plotting results...")
        self.plot_training_results()
        self.plot_evaluation_results()
        self.plot_loss_results()
        self.plot_combined_results()


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(description="Parse and visualize training results")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing config.json and metrics.jsonl files",
    )

    args = parser.parse_args()

    print("Plotting results...")
    try:
        result_parser = ResultParser(args.results_dir)

        # Print summary
        result_parser.plot_results()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
