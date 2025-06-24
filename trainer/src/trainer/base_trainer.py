import os
import time

import gymnasium as gym
import numpy as np

from schema.config import BaseConfig
from schema.result import EvalResult, TotalTrainResult
from util.file import log_result, save_json
from util.settings import set_seed


class BaseTrainer:
    def __init__(self, env: gym.Env, config: BaseConfig, save_dir: str):
        set_seed(config.seed)

        self.env = env
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = int(
            env.action_space.n
            if isinstance(env.action_space, gym.spaces.Discrete)
            else env.action_space.shape[0]
        )
        self.memory = None
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.total_train_result = TotalTrainResult.initialize()
        self.best_score = -np.inf
        self.best_results: EvalResult = None
        self.train_episode_number = 0
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
            if self.config.eval and ep % self.config.eval_period == 0:
                eval_result = self.evaluate(self.config.eval_episodes)
                if self.best_results is None or eval_result > self.best_results:
                    self.best_results = eval_result
                    print("New best results:")
                print(eval_result)
                log_result(eval_result, self.log_file)
                self.save_model()

    def evaluate(self, episodes=10):
        self.eval_mode_on()

        scores = []
        steps = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                action = self.select_action(state)["action"]
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

                step += 1
                if step >= self.config.max_steps:
                    break

            scores.append(total_reward)
            steps.append(step)

        self.eval_mode_off()

        return EvalResult.from_eval_results(scores, steps)

    def select_action(self, state) -> dict:
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
