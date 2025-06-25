import argparse
import random
import time

from env.gym import GymEnvFactory
from trainer.trainer_factory import TrainerFactory


def main():
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model."
    )
    parser.add_argument(
        "--env", type=str, default="LunarLander-v3", help="Gymnasium environment name"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=TrainerFactory.selectable_models,
        default="ppo",
        help="Model to train",
    )
    parser.add_argument(
        "--seed", type=int, default=random.randint(0, 1000000), help="Random seed"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension for networks"
    )
    parser.add_argument(
        "--n_layers", type=int, default=3, help="Number of layers for networks"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1000000, help="Replay buffer size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--eval", type=bool, default=False, help="Whether to evaluate the model"
    )
    parser.add_argument("--eval_period", type=int, default=10, help="Evaluation period")
    parser.add_argument(
        "--eval_episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6, help="Alpha parameter for Rainbow DQN"
    )
    parser.add_argument(
        "--beta_start", type=float, default=0.4, help="Beta start for Rainbow DQN"
    )
    parser.add_argument(
        "--beta_frames", type=int, default=100000, help="Beta frames for Rainbow DQN"
    )
    parser.add_argument(
        "--target_update", type=int, default=10, help="Target network update frequency"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="Tau parameter for SAC/TD3"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=1.0, help="Entropy coefficient for SAC"
    )
    parser.add_argument(
        "--start_steps", type=int, default=1000, help="Start steps for SAC/TD3"
    )
    parser.add_argument("--ppo_epochs", type=int, default=10, help="PPO epochs")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument(
        "--n_transactions",
        type=int,
        default=1000,
        help="Number of transactions for PPO",
    )
    parser.add_argument(
        "--normalize_advantages",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to normalize advantages in PPO",
    )
    # TD3 specific parameters
    parser.add_argument(
        "--policy_delay", type=int, default=2, help="Policy delay for TD3"
    )
    parser.add_argument(
        "--policy_noise", type=float, default=0.2, help="Policy noise for TD3"
    )
    parser.add_argument(
        "--noise_clip", type=float, default=0.5, help="Noise clip for TD3"
    )
    parser.add_argument(
        "--exploration_noise", type=float, default=0.1, help="Exploration noise for TD3"
    )
    args = parser.parse_args()

    config = vars(args)

    save_dir = (
        f"{args.save_dir}/{args.env}/{args.model}/{time.strftime('%Y%m%d_%H%M%S')}"
    )

    trainer = TrainerFactory(args.env, config, save_dir=save_dir)

    trainer.train()


if __name__ == "__main__":
    main()
