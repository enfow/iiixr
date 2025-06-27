import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import imageio
import numpy as np
import torch

from env.gym import GymEnvFactory
from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer
from trainer.td3_trainer import TD3Trainer
from util.gym_env import is_discrete_action_space
from util.settings import set_seed


class GIFGenerator:
    """Generate GIFs from trained models in results directories"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)

        # Load configuration
        self.config = self._load_config()
        self.model_config = self.config.get("model", {})

        # Set up environment and model
        self.env_name = self.config.get("env", "unknown")
        self.model_name = self.model_config.get("model", "unknown")
        self.device = self.config.get("device", "cpu")

        # Compute environment dimensions before initializing trainer
        self._compute_env_dimensions()

        # Initialize trainer
        self.trainer = self._init_trainer()
        self.trainer.load_model()

        # Set seed for reproducibility
        set_seed(self.config.get("seed", 42))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        config_file_path = self.results_dir / "config.json"
        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r") as f:
            config = json.load(f)

        return config

    def _compute_env_dimensions(self):
        """Compute state_dim, action_dim, and is_discrete from environment"""
        # Create environment to get dimensions
        env = GymEnvFactory(self.env_name)
        self.is_discrete = is_discrete_action_space(env)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = (
            env.action_space.n if self.is_discrete else env.action_space.shape[0]
        )

        # Update config with computed values
        self.config["state_dim"] = self.state_dim
        self.config["action_dim"] = self.action_dim
        self.config["is_discrete"] = self.is_discrete

        env.close()

    def _init_trainer(self):
        """Initialize the appropriate trainer based on the model type"""
        model_name = self.model_name.lower()

        if model_name == "ppo":
            return PPOTrainer(self.env_name, self.config, str(self.results_dir))
        elif model_name == "sac":
            return SACTrainer(self.env_name, self.config, str(self.results_dir))
        elif model_name == "td3":
            return TD3Trainer(self.env_name, self.config, str(self.results_dir))
        elif model_name == "discrete_sac":
            return DiscreteSACTrainer(self.env_name, self.config, str(self.results_dir))
        elif model_name == "rainbow_dqn":
            return RainbowDQNTrainer(self.env_name, self.config, str(self.results_dir))
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using the trained model"""
        action_info = self.trainer.select_action(state)
        return action_info["action"]

    def generate_gif(
        self,
        max_steps: int = 1000,
        fps: int = 30,
        episodes: int = 1,
        render_mode: str = "rgb_array",
    ) -> str:
        """Generate a GIF by running the model in the environment"""

        # Set output path to the results directory
        output_path = self.results_dir / f"{self.model_name}_{self.env_name}_demo.gif"

        # Create environment with rendering
        env = gym.make(self.env_name, render_mode=render_mode)

        all_frames = []
        total_reward = 0

        for episode in range(episodes):
            state, _ = env.reset()
            episode_frames = []
            episode_reward = 0

            for step in range(max_steps):
                # Render current frame
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)

                # Select action
                action = self.select_action(state)

                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state

                if done:
                    # Render final frame
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                    break

            all_frames.extend(episode_frames)
            total_reward += episode_reward
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

        env.close()

        # Save as GIF
        if all_frames:
            imageio.mimsave(output_path, all_frames, fps=fps)
            print(f"GIF saved to: {output_path}")
            print(f"Total frames: {len(all_frames)}")
            print(f"Average reward per episode: {total_reward / episodes:.2f}")
        else:
            print("No frames captured. Check if the environment supports rendering.")

        return str(output_path)

    def generate_multiple_gifs(
        self,
        max_steps: int = 1000,
        fps: int = 30,
        episodes_per_gif: int = 1,
        num_gifs: int = 3,
        render_mode: str = "rgb_array",
    ) -> List[str]:
        """Generate multiple GIFs with different random seeds"""

        # Create gifs subdirectory in results directory
        output_dir = self.results_dir / "gifs"
        output_dir.mkdir(exist_ok=True)

        gif_paths = []

        for i in range(num_gifs):
            # Set different seed for each GIF
            set_seed(self.config.get("seed", 42) + i)

            output_path = (
                output_dir / f"{self.model_name}_{self.env_name}_demo_{i + 1}.gif"
            )

            # Create environment with rendering
            env = gym.make(self.env_name, render_mode=render_mode)

            all_frames = []
            total_reward = 0

            for episode in range(episodes_per_gif):
                state, _ = env.reset()
                episode_frames = []
                episode_reward = 0

                for step in range(max_steps):
                    # Render current frame
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)

                    # Select action
                    action = self.select_action(state)

                    # Take step in environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    state = next_state

                    if done:
                        # Render final frame
                        frame = env.render()
                        if frame is not None:
                            episode_frames.append(frame)
                        break

                all_frames.extend(episode_frames)
                total_reward += episode_reward
                print(
                    f"GIF {i + 1}, Episode {episode + 1}: Reward = {episode_reward:.2f}"
                )

            env.close()

            # Save as GIF
            if all_frames:
                imageio.mimsave(output_path, all_frames, fps=fps)
                print(f"GIF {i + 1} saved to: {output_path}")
                print(f"Total frames: {len(all_frames)}")
                print(
                    f"Average reward per episode: {total_reward / episodes_per_gif:.2f}"
                )
                gif_paths.append(str(output_path))
            else:
                print(f"No frames captured for GIF {i + 1}.")

        return gif_paths


def run_gif_generator(
    results_dir: str,
    max_steps: int = 1000,
    fps: int = 30,
    episodes: int = 1,
    multiple: bool = False,
    num_gifs: int = 3,
    render_mode: str = "rgb_array",
):
    generator = GIFGenerator(results_dir)
    if multiple:
        gif_paths = generator.generate_multiple_gifs(
            max_steps=max_steps,
            fps=fps,
            episodes_per_gif=episodes,
            num_gifs=num_gifs,
            render_mode=render_mode,
        )
    else:
        gif_path = generator.generate_gif(
            max_steps=max_steps, fps=fps, episodes=episodes, render_mode=render_mode
        )
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
        "--multiple", action="store_true", help="Generate multiple GIFs"
    )
    parser.add_argument(
        "--num_gifs", type=int, default=3, help="Number of GIFs to generate"
    )
    parser.add_argument(
        "--render_mode", default="rgb_array", help="Gymnasium render mode"
    )

    args = parser.parse_args()

    try:
        run_gif_generator(
            args.results_dir,
            args.max_steps,
            args.fps,
            args.episodes,
            args.multiple,
            args.num_gifs,
            args.render_mode,
        )
    except Exception as e:
        print(f"Error generating GIF: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
