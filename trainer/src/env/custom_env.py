from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

# Enhanced config with feature flags and modular options
DEFAULT_CONFIG = {
    # === FEATURE FLAGS ===
    # Reward shaping options
    "use_potential_shaping": True,
    "use_progress_reward": True,
    "use_stability_reward": True,
    "use_velocity_reward": True,
    "use_contact_reward": True,
    "use_energy_penalty": True,
    "use_lidar_reward": True,
    "use_joint_penalty": True,
    "use_survival_bonus": True,
    # Early termination options
    "enable_early_termination": True,
    "enable_no_progress_termination": True,
    "enable_backward_termination": True,
    "enable_stuck_termination": True,
    "enable_stability_termination": False,  # New: terminate on excessive tilting
    "enable_speed_termination": False,  # New: terminate on excessive speed
    # === REWARD WEIGHTS ===
    "progress_weight": 100.0,  # Forward movement
    "stability_weight": 10.0,  # Keep upright
    "velocity_weight": 5.0,  # Maintain good speed
    "contact_weight": 2.0,  # Foot contact with ground
    "energy_penalty": 0.05,  # Penalize excessive actions
    "lidar_weight": 1.0,  # Obstacle avoidance
    "joint_penalty": 0.02,  # Smooth joint movements
    "survival_bonus": 0.5,  # Stay alive bonus
    # === EARLY TERMINATION PARAMETERS ===
    "no_progress_steps": 200,  # Steps without progress before termination
    "min_progress_threshold": 0.01,  # Minimum distance per step to count as progress
    "backward_threshold": -0.5,  # How far backward before termination
    "stuck_distance_threshold": 0.1,  # Max distance range to be considered stuck
    "max_tilt_angle": 0.8,  # Maximum hull angle before termination (radians)
    "max_speed": 10.0,  # Maximum speed before termination
    # === GENERAL PARAMETERS ===
    "gamma": 0.99,  # Discount factor for potential shaping
    "target_speed": 2.5,  # Target speed for optimal performance
    "reward_mode": "shaped",  # "original", "shaped", or "combined"
    "verbose": False,  # Print detailed reward information
}


class CustomBipedalWalkerWrapper(gym.Wrapper):
    """
    Highly configurable BipedalWalker wrapper with:
    1. Modular potential-based reward shaping
    2. Configurable early termination conditions
    3. Multiple reward calculation modes
    4. Comprehensive tracking and monitoring
    """

    def __init__(self, env, config: Dict = None):
        super().__init__(env)

        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Validate configuration
        self._validate_config()

        # State tracking
        self.prev_hull_pos = 0.0
        self.prev_hull_vel = 0.0
        self.prev_hull_angle = 0.0
        self.prev_actions = np.zeros(4)
        self.step_count = 0
        self.total_distance = 0.0

        # Progress tracking for early termination
        self.last_progress_step = 0
        self.best_distance = 0.0
        self.no_progress_counter = 0
        self.position_history = []

        # Performance tracking
        self.episode_rewards = []
        self.shaped_rewards_history = []

        if self.config["verbose"]:
            print("Enhanced CustomBipedalWalkerWrapper initialized with config:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")

    def _validate_config(self):
        """Validate configuration settings"""
        valid_reward_modes = ["original", "shaped", "combined"]
        if self.config["reward_mode"] not in valid_reward_modes:
            raise ValueError(f"reward_mode must be one of {valid_reward_modes}")

        # Warn about conflicting settings
        if (
            not self.config["use_potential_shaping"]
            and self.config["reward_mode"] == "shaped"
        ):
            print("Warning: reward_mode is 'shaped' but use_potential_shaping is False")

    def reset(self, **kwargs):
        """Reset environment and initialize tracking variables"""
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Reset state variables
        self.prev_hull_pos = obs[2] if len(obs) > 2 else 0.0
        self.prev_hull_vel = obs[3] if len(obs) > 3 else 0.0
        self.prev_hull_angle = obs[0] if len(obs) > 0 else 0.0
        self.prev_actions = np.zeros(4)
        self.step_count = 0
        self.total_distance = 0.0

        # Reset progress tracking
        self.last_progress_step = 0
        self.best_distance = 0.0
        self.no_progress_counter = 0
        self.position_history = [obs[2] if len(obs) > 2 else 0.0]

        # Save observation
        self._last_obs = obs.copy()

        return obs, info

    def potential_function(self, obs: np.ndarray) -> float:
        """
        Compute potential function for reward shaping with configurable components
        """
        if not self.config["use_potential_shaping"]:
            return 0.0

        # Extract state components
        hull_angle = obs[0]
        hull_angular_vel = obs[1]
        hull_vel_x = obs[2]
        hull_vel_y = obs[3]

        # Joint angles and velocities
        joint_angles = obs[4::2][:4]  # Every other element starting from 4
        joint_vels = obs[5::2][:4]  # Every other element starting from 5

        # Leg contacts (legs[1] and legs[3] ground contact)
        leg_contacts = [obs[8], obs[13]] if len(obs) > 13 else [0, 0]

        # Lidar readings (last 10 elements)
        lidar = obs[-10:] if len(obs) >= 10 else np.zeros(10)

        # Get current position
        hull_pos_x = self.total_distance

        total_potential = 0.0

        # 1. Progress potential - reward forward movement
        if self.config["use_progress_reward"]:
            progress_potential = hull_pos_x * self.config["progress_weight"]
            total_potential += progress_potential

        # 2. Velocity potential - encourage optimal speed
        if self.config["use_velocity_reward"]:
            target_speed = self.config["target_speed"]
            speed_diff = abs(hull_vel_x - target_speed)
            velocity_potential = -speed_diff * self.config["velocity_weight"]
            total_potential += velocity_potential

        # 3. Stability potential - keep upright and stable
        if self.config["use_stability_reward"]:
            angle_penalty = abs(hull_angle) * self.config["stability_weight"]
            angular_vel_penalty = (
                abs(hull_angular_vel) * self.config["stability_weight"] * 0.1
            )
            vertical_vel_penalty = (
                abs(hull_vel_y) * self.config["stability_weight"] * 0.1
            )
            stability_potential = -(
                angle_penalty + angular_vel_penalty + vertical_vel_penalty
            )
            total_potential += stability_potential

        # 4. Contact potential - encourage proper foot contact
        if self.config["use_contact_reward"]:
            contact_bonus = sum(leg_contacts) * self.config["contact_weight"]
            total_potential += contact_bonus

        # 5. Safety potential - avoid obstacles using lidar
        if self.config["use_lidar_reward"]:
            min_lidar = np.min(lidar)
            safety_potential = min_lidar * self.config["lidar_weight"]
            total_potential += safety_potential

        # 6. Joint smoothness - penalize erratic movements
        if self.config["use_joint_penalty"]:
            joint_penalty = -np.sum(np.abs(joint_vels)) * self.config["joint_penalty"]
            total_potential += joint_penalty

        # 7. Survival bonus - small bonus for staying alive
        if self.config["use_survival_bonus"]:
            survival_potential = self.config["survival_bonus"]
            total_potential += survival_potential

        return total_potential

    def check_early_termination(self, obs: np.ndarray) -> Tuple[bool, str]:
        """
        Check if episode should terminate early based on configured conditions
        """
        if not self.config["enable_early_termination"]:
            return False, ""

        current_pos = obs[2] if len(obs) > 2 else 0.0
        hull_angle = obs[0] if len(obs) > 0 else 0.0
        hull_vel_x = obs[2] if len(obs) > 2 else 0.0

        self.position_history.append(current_pos)

        # Keep only recent history
        if len(self.position_history) > self.config["no_progress_steps"]:
            self.position_history.pop(0)

        # Update best distance
        if current_pos > self.best_distance:
            self.best_distance = current_pos
            self.last_progress_step = self.step_count
            self.no_progress_counter = 0
        else:
            self.no_progress_counter += 1

        # Check stability termination
        if self.config["enable_stability_termination"]:
            if abs(hull_angle) > self.config["max_tilt_angle"]:
                return True, "excessive_tilt"

        # Check speed termination
        if self.config["enable_speed_termination"]:
            if abs(hull_vel_x) > self.config["max_speed"]:
                return True, "excessive_speed"

        # Check for backward movement termination
        if self.config["enable_backward_termination"]:
            if current_pos < self.config["backward_threshold"]:
                return True, "moved_too_far_backward"

        # Check for no progress termination
        if self.config["enable_no_progress_termination"]:
            steps_without_progress = self.step_count - self.last_progress_step
            if steps_without_progress >= self.config["no_progress_steps"]:
                return True, "no_forward_progress"

        # Check if stuck (very small movements over time)
        if self.config["enable_stuck_termination"]:
            if len(self.position_history) >= self.config["no_progress_steps"]:
                position_range = max(self.position_history) - min(self.position_history)
                if position_range < self.config["stuck_distance_threshold"]:
                    return True, "stuck_in_place"

        return False, ""

    def compute_shaped_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        original_reward: float,
    ) -> Tuple[float, Dict]:
        """
        Compute reward based on configured mode
        """
        reward_components = {
            "original": original_reward,
            "potential_shaping": 0.0,
            "action_penalty": 0.0,
            "total": original_reward,
            "current_potential": 0.0,
            "next_potential": 0.0,
        }

        # Potential-based shaping: γ * Φ(s') - Φ(s)
        if self.config["use_potential_shaping"]:
            current_potential = self.potential_function(obs)
            next_potential = self.potential_function(next_obs)
            potential_shaping = (
                self.config["gamma"] * next_potential - current_potential
            )
            reward_components["potential_shaping"] = potential_shaping
            reward_components["current_potential"] = current_potential
            reward_components["next_potential"] = next_potential

        # Energy penalty for actions
        if self.config["use_energy_penalty"]:
            action_penalty = -np.sum(np.abs(action)) * self.config["energy_penalty"]
            reward_components["action_penalty"] = action_penalty

        # Calculate final reward based on mode
        if self.config["reward_mode"] == "original":
            final_reward = original_reward
        elif self.config["reward_mode"] == "shaped":
            final_reward = (
                reward_components["potential_shaping"]
                + reward_components["action_penalty"]
            )
        elif self.config["reward_mode"] == "combined":
            final_reward = (
                original_reward
                + reward_components["potential_shaping"]
                + reward_components["action_penalty"]
            )
        else:
            final_reward = original_reward

        reward_components["total"] = final_reward

        return final_reward, reward_components

    def step(self, action):
        """Step environment with configurable reward shaping and early termination"""
        # Get current state
        current_obs = getattr(self, "_last_obs", np.zeros(self.observation_space.shape))

        # Step environment
        next_obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Update state tracking
        self.step_count += 1
        self.total_distance = next_obs[2] if len(next_obs) > 2 else 0.0

        # Check for early termination
        early_term, term_reason = self.check_early_termination(next_obs)
        if early_term and not (terminated or truncated):
            terminated = True
            info["early_termination_reason"] = term_reason
            # Configurable penalty for early termination
            if self.config["reward_mode"] != "original":
                original_reward -= 50

        # Compute shaped reward
        shaped_reward, reward_components = self.compute_shaped_reward(
            current_obs, action, next_obs, original_reward
        )

        # Update tracking variables
        self.prev_actions = action.copy()
        self._last_obs = next_obs.copy()

        # Enhanced info
        info.update(
            {
                "reward_components": reward_components,
                "original_reward": original_reward,
                "shaped_reward": shaped_reward,
                "total_distance": self.total_distance,
                "best_distance": self.best_distance,
                "steps_without_progress": self.step_count - self.last_progress_step,
                "no_progress_counter": self.no_progress_counter,
                "reward_mode": self.config["reward_mode"],
            }
        )

        # Track performance
        self.shaped_rewards_history.append(reward_components)

        # Verbose logging
        if self.config["verbose"] and self.step_count % 100 == 0:
            print(
                f"Step {self.step_count}: Reward={shaped_reward:.2f}, "
                f"Distance={self.total_distance:.2f}, "
                f"Progress={self.step_count - self.last_progress_step}"
            )

        return next_obs, shaped_reward, terminated, truncated, info

    def get_episode_stats(self) -> Dict:
        """Get statistics for the completed episode"""
        if not self.shaped_rewards_history:
            return {}

        total_original = sum(r["original"] for r in self.shaped_rewards_history)
        total_shaped = sum(r["total"] for r in self.shaped_rewards_history)
        total_potential = sum(
            r["potential_shaping"] for r in self.shaped_rewards_history
        )

        return {
            "episode_length": self.step_count,
            "total_distance": self.total_distance,
            "best_distance": self.best_distance,
            "total_original_reward": total_original,
            "total_shaped_reward": total_shaped,
            "potential_contribution": total_potential,
            "average_reward_per_step": total_shaped / max(1, self.step_count),
            "reward_mode": self.config["reward_mode"],
        }

    def get_active_features(self) -> Dict:
        """Get a summary of which features are currently active"""
        return {
            "reward_features": {
                "potential_shaping": self.config["use_potential_shaping"],
                "progress_reward": self.config["use_progress_reward"],
                "stability_reward": self.config["use_stability_reward"],
                "velocity_reward": self.config["use_velocity_reward"],
                "contact_reward": self.config["use_contact_reward"],
                "energy_penalty": self.config["use_energy_penalty"],
                "lidar_reward": self.config["use_lidar_reward"],
                "joint_penalty": self.config["use_joint_penalty"],
                "survival_bonus": self.config["use_survival_bonus"],
            },
            "termination_features": {
                "early_termination": self.config["enable_early_termination"],
                "no_progress_termination": self.config[
                    "enable_no_progress_termination"
                ],
                "backward_termination": self.config["enable_backward_termination"],
                "stuck_termination": self.config["enable_stuck_termination"],
                "stability_termination": self.config["enable_stability_termination"],
                "speed_termination": self.config["enable_speed_termination"],
            },
            "reward_mode": self.config["reward_mode"],
        }


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    "minimal": {
        "use_potential_shaping": False,
        "enable_early_termination": False,
        "reward_mode": "original",
        "verbose": False,
    },
    "progress_only": {
        "use_potential_shaping": True,
        "use_progress_reward": True,
        "use_stability_reward": False,
        "use_velocity_reward": False,
        "use_contact_reward": False,
        "use_energy_penalty": False,
        "use_lidar_reward": False,
        "use_joint_penalty": False,
        "use_survival_bonus": False,
        "enable_early_termination": True,
        "enable_no_progress_termination": True,
        "enable_backward_termination": False,
        "enable_stuck_termination": False,
        "reward_mode": "combined",
    },
    "stability_focused": {
        "use_potential_shaping": True,
        "use_progress_reward": True,
        "use_stability_reward": True,
        "use_velocity_reward": False,
        "use_contact_reward": True,
        "stability_weight": 20.0,
        "enable_stability_termination": True,
        "max_tilt_angle": 0.5,
        "reward_mode": "combined",
    },
    "speed_optimized": {
        "use_potential_shaping": True,
        "use_progress_reward": True,
        "use_stability_reward": True,
        "use_velocity_reward": True,
        "target_speed": 3.0,
        "velocity_weight": 10.0,
        "enable_speed_termination": True,
        "max_speed": 8.0,
        "reward_mode": "combined",
    },
    "comprehensive": {
        # All features enabled with balanced weights
        "reward_mode": "combined",
        "verbose": True,
    },
}


if __name__ == "__main__":
    # Test different configurations
    print("Testing configurable BipedalWalker wrapper...")

    # Test 1: Minimal configuration (original environment)
    print("\n=== Test 1: Minimal Configuration ===")
    env1 = CustomBipedalWalkerWrapper(
        gym.make("BipedalWalker-v3"), config=PRESET_CONFIGS["minimal"]
    )
    print("Active features:", env1.get_active_features())

    # Test 2: Progress-only configuration
    print("\n=== Test 2: Progress-Only Configuration ===")
    env2 = CustomBipedalWalkerWrapper(
        gym.make("BipedalWalker-v3"), config=PRESET_CONFIGS["progress_only"]
    )
    print("Active features:", env2.get_active_features())

    # Test 3: Custom configuration
    print("\n=== Test 3: Custom Configuration ===")
    custom_config = {
        "use_potential_shaping": True,
        "use_progress_reward": True,
        "use_stability_reward": True,
        "enable_early_termination": True,
        "enable_no_progress_termination": True,
        "no_progress_steps": 150,
        "reward_mode": "shaped",  # Only shaped rewards, no original
        "verbose": True,
    }

    env3 = CustomBipedalWalkerWrapper(
        gym.make("BipedalWalker-v3"), config=custom_config
    )

    print("Testing custom configuration...")
    obs, info = env3.reset()

    total_reward = 0
    for step in range(200):
        action = env3.action_space.sample()
        obs, reward, terminated, truncated, info = env3.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            if "early_termination_reason" in info:
                print(f"Early termination: {info['early_termination_reason']}")
            break

    # Print final statistics
    stats = env3.get_episode_stats()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(
            f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    env1.close()
    env2.close()
    env3.close()
