import gymnasium as gym


def is_discrete_action_space(env: gym.Env) -> bool:
    return isinstance(env.action_space, gym.spaces.Discrete)
