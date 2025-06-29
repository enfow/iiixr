"""
GIF Generator for Reinforcement Learning Models

This module provides a simple interface to generate GIFs from trained models
by leveraging the trainer's built-in evaluation and GIF generation methods.
This ensures perfect consistency with training results.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from trainer.trainer_factory import TrainerFactory


class GIFGenerator:
    """Simple GIF generator that uses trainer's built-in methods"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)

        # Load configuration
        self.config = self._load_config()

        # Handle config compatibility
        self._fix_config_compatibility()

        # Initialize trainer
        self.trainer = self._init_trainer()
        self.trainer.load_model()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        config_file_path = self.results_dir / "config.json"
        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r") as f:
            config = json.load(f)

        return config

    def _fix_config_compatibility(self):
        """Fix config compatibility issues"""
        config_model_field = self.config.get("model")
        if isinstance(config_model_field, str):
            self.config["model"] = {
                "model": config_model_field,
                "hidden_dim": self.config.get("hidden_dim"),
                "n_layers": self.config.get("n_layers"),
                "embedding_type": self.config.get("embedding_type", "fc"),
            }

        # Handle buffer config compatibility
        if "buffer" not in self.config:
            self.config["buffer"] = {
                "buffer_size": self.config.get("buffer_size", 1000000),
                "buffer_type": self.config.get("buffer_type", "default"),
                "alpha": self.config.get("alpha", 0.6),
                "beta_start": self.config.get("beta_start", 0.4),
                "beta_frames": self.config.get("beta_frames", 100000),
                "seq_len": self.config.get("seq_len", 1),
            }
        elif isinstance(self.config.get("buffer"), str):
            buffer_type = self.config["buffer"]
            self.config["buffer"] = {
                "buffer_type": buffer_type,
                "buffer_size": self.config.get("buffer_size", 1000000),
                "alpha": self.config.get("alpha", 0.6),
                "beta_start": self.config.get("beta_start", 0.4),
                "beta_frames": self.config.get("beta_frames", 100000),
                "seq_len": self.config.get("seq_len", 1),
            }

    def _init_trainer(self):
        """Initialize the appropriate trainer based on the model type"""
        env_name = self.config.get("env", "unknown")
        return TrainerFactory(env_name, self.config, str(self.results_dir))

    def run_evaluation(self, episodes: int = 10) -> Dict[str, Any]:
        """Run evaluation using the trainer's evaluation code"""
        print(f"Running evaluation with {episodes} episodes...")

        try:
            eval_result = self.trainer.evaluate(episodes=episodes)

            print(f"Evaluation Results:")
            print(f"  Average Score: {eval_result.avg_score:.2f}")
            print(f"  Std Score: {eval_result.std_score:.2f}")
            print(f"  Min Score: {eval_result.min_score:.2f}")
            print(f"  Max Score: {eval_result.max_score:.2f}")
            print(f"  All Scores: {eval_result.all_scores}")

            return {
                "avg_score": eval_result.avg_score,
                "std_score": eval_result.std_score,
                "min_score": eval_result.min_score,
                "max_score": eval_result.max_score,
                "all_scores": eval_result.all_scores,
                "steps": eval_result.steps,
            }
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "avg_score": 0.0,
                "std_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "all_scores": [],
                "steps": [],
                "error": str(e),
            }

    def generate_gif(
        self,
        max_steps: int = 1000,
        fps: int = 30,
        episodes: int = 1,
        render_mode: str = "rgb_array",
        gif_name: str = None,
        gif_dir: str = None,
    ) -> str:
        """Generate a GIF using the trainer's built-in method"""
        return self.trainer.generate_gif(
            max_steps=max_steps,
            fps=fps,
            episodes=episodes,
            render_mode=render_mode,
            gif_name=gif_name,
            gif_dir=gif_dir,
        )


def run_gif_generator(
    results_dir: str,
    max_steps: int = 1000,
    fps: int = 30,
    episodes: int = 1,
    render_mode: str = "rgb_array",
    run_eval: bool = False,
    eval_episodes: int = 10,
):
    """Main function to run GIF generation"""
    generator = GIFGenerator(results_dir)

    # Run evaluation first if requested
    if run_eval:
        eval_results = generator.run_evaluation(episodes=eval_episodes)
        print(f"Evaluation completed. Average score: {eval_results['avg_score']:.2f}")
        print("-" * 50)

    gif_path = generator.generate_gif(
        max_steps=max_steps, fps=fps, episodes=episodes, render_mode=render_mode
    )
    print(f"Generated GIF: {gif_path}")
    return gif_path


def main():
    parser = argparse.ArgumentParser(description="Generate GIFs from trained models")
    parser.add_argument(
        "results_dir", help="Path to results directory containing trained model"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS for the GIF")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to record"
    )
    parser.add_argument(
        "--render_mode", default="rgb_array", help="Gymnasium render mode"
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        default=False,
        help="Run evaluation before generating GIFs",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )

    args = parser.parse_args()

    try:
        print(f"Generating GIF for {args.results_dir}")
        run_gif_generator(
            args.results_dir,
            args.max_steps,
            args.fps,
            args.episodes,
            args.render_mode,
            args.run_eval,
            args.eval_episodes,
        )
    except Exception as e:
        print(f"Error generating GIF: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
