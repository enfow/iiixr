import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import gymnasium as gym
import imageio
import numpy as np

from env.gym import GymEnvFactory
from model.buffer import ReployBufferFactory
from schema.result import EvalResult, TotalTrainResult
from util.file import log_result, save_json
from util.gym_env import is_discrete_action_space
from util.settings import set_seed


class BaseTrainer:
    config_class = None

    def __init__(self, env_name: str, config_dict: dict, save_dir: str):
        config = self.config_class.from_dict(config_dict)
        print(f"config: {config}")
        set_seed(config.seed)

        self.env_name = env_name
        self.env = GymEnvFactory(env_name, n_envs=config.n_envs)
        print(f"training env_name: {env_name}")
        print(f"evaluation env_name: {config.eval_env}")
        self.eval_env = (
            GymEnvFactory(config.eval_env)
            if config.eval_env is not None
            else GymEnvFactory(env_name)
        )
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.is_discrete = is_discrete_action_space(self.env)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n
            if self.is_discrete
            else self.env.action_space.shape[0]
        )
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")

        self.memory = ReployBufferFactory(self.config)

        self.state_history = deque(maxlen=self.config.buffer.seq_len)
        self.total_train_result = TotalTrainResult.initialize()
        self.best_score = -np.inf
        self.best_results: EvalResult = None
        self.train_episode_number = 0
        self.step_count = 0
        self.log_file = os.path.join(self.save_dir, "metrics.jsonl")
        self.model_file = os.path.join(self.save_dir, "best_model.pth")
        self.config_file = os.path.join(self.save_dir, "config.json")
        self._log_config()
        self._init_models()

    def _init_models(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train_episode(self):
        raise NotImplementedError("Subclasses must implement this method")

    def collect_initial_data(self, start_steps):
        state, _ = self.env.reset()
        for _ in range(start_steps):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, _ = self.env.reset()
                break
        print(f"Collected {start_steps} initial data")

    def _print_trainer_summary(self):
        # print env and model info
        print(f"Env: {self.env_name} Model: {self.config.model.model}")

    def train(self):
        if hasattr(self.config, "start_steps") and self.config.start_steps > 0:
            self.collect_initial_data(self.config.start_steps)

        for ep in range(self.config.episodes):
            self.train_episode_number = ep + 1
            start_time = time.time()
            single_episode_result = self.train_episode()
            elapsed_time = time.time() - start_time

            # log train result
            single_episode_result.episode_elapsed_time = elapsed_time
            single_episode_result.episode_number = self.train_episode_number
            print(single_episode_result)
            self.total_train_result.update(single_episode_result)
            log_result(single_episode_result, self.log_file)

            # evaluate
            if (
                self.config.eval
                and self.train_episode_number % self.config.eval_period == 0
            ):
                eval_result = self.evaluate(self.config.eval_episodes)
                if self.best_results is None or eval_result > self.best_results:
                    self.best_results = eval_result
                    print(
                        f"New best results(params saved for ep={self.train_episode_number})"
                    )
                    self.save_model()
                self._print_trainer_summary()
                print(eval_result)
                log_result(eval_result, self.log_file)

    def evaluate(self, episodes=10):
        self.eval_mode_on()
        scores = []
        steps = []
        for _ in range(episodes):
            state, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                action = self.select_action(state, eval_mode=True)["action"]
                next_state, reward, terminated, truncated, _ = self.eval_env.step(
                    action
                )
                done = terminated or truncated
                state = next_state
                total_reward += reward

                step += 1
                if step >= self.config.max_steps:
                    break

            scores.append(total_reward)
            steps.append(step)

        self.eval_mode_off()

        return EvalResult.from_eval_results(scores, steps, self.train_episode_number)

    def generate_gif(
        self,
        max_steps: int = 1000,
        fps: int = 30,
        episodes: int = 1,
        render_mode: str = "rgb_array",
        gif_name: Optional[str] = None,
        gif_dir: Optional[str] = None,
    ) -> str:
        if gif_dir is None:
            gif_dir = self.save_dir
        else:
            gif_dir = Path(gif_dir)
            gif_dir.mkdir(exist_ok=True)

        if gif_name is None:
            gif_name = f"{self.config.model.model}_{self.env_name}_demo.gif"

        output_path = Path(gif_dir) / gif_name

        # Create environment with rendering
        env = gym.make(self.env_name, render_mode=render_mode)

        all_frames = []
        total_reward = 0

        # Use trainer's evaluation mode
        self.eval_mode_on()

        for episode in range(episodes):
            state, _ = env.reset()

            # Reset episode for sequential trainers
            if hasattr(self, "reset_episode"):
                self.reset_episode()

            episode_frames = []
            episode_reward = 0
            step = 0

            while step < max_steps:
                # Render current frame
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)

                # Use trainer's select_action method (same as evaluation)
                action_info = self.select_action(state, eval_mode=True)
                action = action_info["action"]

                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state
                step += 1

                if done:
                    # Render final frame
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                    break

            all_frames.extend(episode_frames)
            total_reward += episode_reward
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

        # Restore training mode
        self.eval_mode_off()
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

    def select_action(self, state, eval_mode: bool = False) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    def update(self):
        raise NotImplementedError("Subclasses must implement this method")

    def save_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def eval_mode_on(self):
        raise NotImplementedError("Subclasses must implement this method")

    def eval_mode_off(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _log_config(self):
        save_json(self.config.to_dict(), self.config_file)
