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
        """환경 리셋 및 상태 초기화"""
        # Gymnasium은 (obs, info) 튜플을 반환
        result = self.env.reset(**kwargs)

        # 반환값이 튜플인지 확인
        if isinstance(result, tuple):
            obs, info = result
        else:
            # 이전 gym 버전 호환성
            obs = result
            info = {}

        # 상태 변수들 초기화
        self.prev_hull_pos = obs[2]
        self.prev_hull_vel = obs[3]
        self.prev_hull_angle = obs[1]
        self.prev_actions = np.zeros(4)
        self.step_count = 0
        self.total_distance = 0.0

        # 첫 번째 관찰 저장
        self._last_obs = obs.copy()

        return obs, info

    def potential_function(self, obs: np.ndarray) -> float:
        """
        Potential function Φ(s) 정의
        이 함수가 reward shaping의 핵심!
        """
        hull_pos_x = obs[2]
        hull_vel_x = obs[3]
        hull_angle = obs[1]
        leg_contacts = obs[18:22]
        lidar = obs[8:18]

        # 1. 위치 기반 potential (전진할수록 높아짐)
        position_potential = hull_pos_x * 10.0

        # 2. 속도 기반 potential (적절한 속도 유지)
        optimal_speed = 2.0
        speed_potential = -abs(hull_vel_x - optimal_speed) * 5.0

        # 3. 안정성 potential (직립 상태 선호)
        stability_potential = -abs(hull_angle) * 10.0

        # 4. 접촉 potential (양발 접촉 선호)
        contact_potential = np.sum(leg_contacts) * 2.0

        # 5. 안전 potential (장애물로부터 거리)
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
        """
        Shaped reward 계산 (potential-based shaping)
        """
        # Potential-based shaping (이론적으로 안전)
        current_potential = self.potential_function(obs)
        next_potential = self.potential_function(next_obs)
        potential_shaping = 0.99 * next_potential - current_potential  # γ=0.99

        # 기본적으로는 potential-based shaping만 사용 (안전함)
        shaped_components = {}
        total_shaped_reward = potential_shaping + original_reward

        # 디버깅 정보
        shaped_components["potential"] = potential_shaping
        shaped_components["original"] = original_reward
        shaped_components["total"] = total_shaped_reward

        return total_shaped_reward, shaped_components

    def step(self, action):
        """환경 스텝 실행 및 reward shaping 적용"""
        # 현재 상태 가져오기
        current_obs = getattr(self, "_last_obs", np.zeros(self.observation_space.shape))

        # 환경 스텝 실행
        next_obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Gymnasium은 terminated와 truncated를 분리해서 반환
        # 이전 gym 호환성을 위해 done으로 통합
        done = terminated or truncated

        # 다음 관찰을 저장
        self._last_obs = next_obs.copy()

        # Shaped reward 계산
        shaped_reward, reward_components = self.compute_shaped_reward(
            current_obs, action, next_obs, original_reward
        )

        # 상태 업데이트
        self.prev_actions = action.copy()
        self.step_count += 1
        self.total_distance = next_obs[2]

        # 정보 업데이트
        info["reward_components"] = reward_components
        info["original_reward"] = original_reward
        info["shaped_reward"] = shaped_reward
        info["total_distance"] = self.total_distance

        # 성능 추적
        self.shaped_rewards_history.append(reward_components)

        return next_obs, shaped_reward, terminated, truncated, info


# 더 발전된 버전 (추가적인 reward shaping 포함)
class AdvancedCustomBipedalWalkerWrapper(CustomBipedalWalkerWrapper):
    """추가적인 reward shaping을 포함한 고급 버전"""

    def compute_shaped_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        original_reward: float,
    ) -> Tuple[float, Dict]:
        """
        Shaped reward 계산 (potential-based + additional shaping)
        """
        # Potential-based shaping (이론적으로 안전)
        current_potential = self.potential_function(obs)
        next_potential = self.potential_function(next_obs)
        potential_shaping = 0.99 * next_potential - current_potential  # γ=0.99

        # 추가적인 non-potential shaping (조심스럽게 사용)
        shaped_components = {}

        # 1. 전진 진척 보상
        hull_pos_progress = next_obs[2] - obs[2]
        progress_reward = hull_pos_progress * self.config["progress_weight"]
        shaped_components["progress"] = progress_reward

        # 2. 안정성 보상 (각도 변화 최소화)
        angle_stability = -abs(next_obs[1] - obs[1]) * self.config["stability_weight"]
        shaped_components["stability"] = angle_stability

        # 3. 속도 일관성 보상
        target_velocity = 2.0
        current_vel = next_obs[3]
        velocity_reward = (
            -(abs(current_vel - target_velocity)) * self.config["velocity_weight"]
        )
        shaped_components["velocity"] = velocity_reward

        # 4. 발 접촉 보상
        leg_contacts = next_obs[18:22]
        contact_reward = np.sum(leg_contacts > 0.1) * self.config["contact_weight"]
        shaped_components["contact"] = contact_reward

        # 5. 에너지 효율성 (action smoothness)
        action_diff = np.sum(np.square(action - self.prev_actions))
        energy_penalty = -action_diff * self.config["energy_penalty"]
        shaped_components["energy"] = energy_penalty

        # 6. 장애물 회피 보상
        lidar_readings = next_obs[8:18]
        min_distance = np.min(lidar_readings)
        if min_distance < 0.3:  # 위험 구역
            obstacle_penalty = -(0.3 - min_distance) * self.config["lidar_weight"] * 10
        else:
            obstacle_penalty = 0
        shaped_components["obstacle"] = obstacle_penalty

        # 7. 관절 각도 제한 보상
        joint_angles = next_obs[4:8]
        joint_penalty = (
            -np.sum(np.square(np.clip(np.abs(joint_angles) - 1.0, 0, None)))
            * self.config["joint_penalty"]
        )
        shaped_components["joint"] = joint_penalty

        # 8. 생존 보너스
        survival_bonus = self.config["survival_bonus"]
        shaped_components["survival"] = survival_bonus

        # 총 shaped reward
        additional_shaping = sum(shaped_components.values())
        total_shaped_reward = original_reward + potential_shaping + additional_shaping

        # 디버깅 정보
        shaped_components["potential"] = potential_shaping
        shaped_components["original"] = original_reward
        shaped_components["total"] = total_shaped_reward

        return total_shaped_reward, shaped_components


# 사용 예제
def create_bipedal_walker_env(use_advanced=False, shaping_config=None):
    """BipedalWalker 환경 생성 헬퍼 함수"""

    # 기본 환경 생성
    base_env = gym.make("BipedalWalkerHardcore-v3")

    # Wrapper 선택
    if use_advanced:
        env = AdvancedCustomBipedalWalkerWrapper(base_env, shaping_config)
    else:
        env = CustomBipedalWalkerWrapper(base_env, shaping_config)

    return env


# 테스트 코드
if __name__ == "__main__":
    # 기본 설정으로 환경 생성
    env = create_bipedal_walker_env(use_advanced=False)

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
