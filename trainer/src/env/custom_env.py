from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

# introduce config
DEFAULT_CONFIG = {
    "progress_weight": 10.0,
    "stability_weight": 5.0,
    "velocity_weight": 2.0,
    "contact_weight": 1.0,
    "energy_penalty": 0.1,
    "lidar_weight": 0.5,
    "joint_penalty": 0.05,
    "survival_bonus": 0.1,
}


class CustomBipedalWalkerWrapper(gym.Wrapper):
    # env = CustomBipedalWalkerWrapper(base_env, shaping_config)

    def __init__(self, env, config: Dict = None):
        super().__init__(env)

        self.config = config or DEFAULT_CONFIG

        self.prev_hull_pos = 0.0
        self.prev_hull_vel = 0.0
        self.prev_hull_angle = 0.0
        self.prev_actions = np.zeros(4)
        self.step_count = 0
        self.total_distance = 0.0

        self.episode_rewards = []
        self.shaped_rewards_history = []

        print("CustomBipedalWalkerWrapper initialized with config:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    def reset(self, **kwargs):
        # Gymnasium returns (obs, info) tuple
        result = self.env.reset(**kwargs)

        # check if return value is a tuple
        if isinstance(result, tuple):
            obs, info = result
        else:
            # backward compatibility for previous gym versions
            obs = result
            info = {}

        # initialize state variables
        self.prev_hull_pos = obs[2]
        self.prev_hull_vel = obs[3]
        self.prev_hull_angle = obs[1]
        self.prev_actions = np.zeros(4)
        self.step_count = 0
        self.total_distance = 0.0

        # save first observation
        self._last_obs = obs.copy()

        return obs, info

    def potential_function(self, obs: np.ndarray) -> float:
        hull_pos_x = obs[2]
        hull_vel_x = obs[3]
        hull_angle = obs[1]
        leg_contacts = obs[18:22]
        lidar = obs[8:18]

        # 1. position potential
        position_potential = hull_pos_x * 10.0

        # 2. speed potential
        optimal_speed = 2.0
        speed_potential = -abs(hull_vel_x - optimal_speed) * 5.0

        # 3. stability potential
        stability_potential = -abs(hull_angle) * 10.0

        # 4. contact potential
        contact_potential = np.sum(leg_contacts) * 2.0

        # 5. safety potential
        min_lidar = np.min(lidar)
        safety_potential = min_lidar * 3.0

        total_potential = (
            position_potential
            + speed_potential
            + stability_potential
            + contact_potential
            + safety_potential
        )

        return total_potential

    def compute_shaped_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        original_reward: float,
    ) -> Tuple[float, Dict]:
        # Potential-based shaping
        current_potential = self.potential_function(obs)
        next_potential = self.potential_function(next_obs)
        potential_shaping = 0.99 * next_potential - current_potential  # γ=0.99

        # use only potential-based shaping
        shaped_components = {}
        total_shaped_reward = potential_shaping + original_reward

        # debugging info
        shaped_components["potential"] = potential_shaping
        shaped_components["original"] = original_reward
        shaped_components["total"] = total_shaped_reward

        return total_shaped_reward, shaped_components

    def step(self, action):
        # get current state
        current_obs = getattr(self, "_last_obs", np.zeros(self.observation_space.shape))

        # step environment
        next_obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Gymnasium returns terminated and truncated separately
        # for backward compatibility, we combine them into done
        done = terminated or truncated

        # save next observation
        self._last_obs = next_obs.copy()

        # compute shaped reward
        shaped_reward, reward_components = self.compute_shaped_reward(
            current_obs, action, next_obs, original_reward
        )

        # update state
        self.prev_actions = action.copy()
        self.step_count += 1
        self.total_distance = next_obs[2]

        # update info
        info["reward_components"] = reward_components
        info["original_reward"] = original_reward
        info["shaped_reward"] = shaped_reward
        info["total_distance"] = self.total_distance

        # track performance
        self.shaped_rewards_history.append(reward_components)

        return next_obs, shaped_reward, terminated, truncated, info


if __name__ == "__main__":
    env = CustomBipedalWalkerWrapper(gym.make("BipedalWalkerHardcore-v3"))

    print("Testing environment...")

    # 환경 테스트
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    total_reward = 0
    original_total_reward = 0

    for step in range(100):  # 100 스텝만 테스트
        action = env.action_space.sample()  # 랜덤 액션
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        original_total_reward += info["original_reward"]

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"Total shaped reward: {total_reward:.2f}")
    print(f"Total original reward: {original_total_reward:.2f}")
    print(f"Potential shaping contribution: {total_reward - original_total_reward:.2f}")

    env.close()
