import argparse
import random
import time

import gymnasium as gym

from trainer.discrete_sac_trainer import DiscreteSACTrainer
from trainer.ppo_trainer import PPOTrainer
from trainer.rainbow_dqn_trainer import RainbowDQNTrainer
from trainer.sac_trainer import SACTrainer


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
        choices=["ppo", "sac", "rainbow_dqn", "discrete_sac"],
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
        "--buffer_size", type=int, default=1000000, help="Replay buffer size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--target_update", type=int, default=10, help="Target network update frequency"
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
    args = parser.parse_args()

    if args.env == "BipedalWalker-v3":
        env = gym.make(args.env, hardcore=True)
    else:
        env = gym.make(args.env)

    config = {
        "model": args.model,
        "env_name": args.env,
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        # reinforcement learning hyperparameters
        "gamma": args.gamma,
        # SAC specific hyperparameters
        "tau": 0.005,
        "start_steps": 1000,
        "entropy_coef": 1.0,
        # PPO specific hyperparameters
        "ppo_epochs": 4,
        "clip_eps": 0.2,
        # Rainbow DQN hyperparameters
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 100000,
        "target_update": args.target_update,
        "eval": args.eval,
        "eval_period": args.eval_period,
        "eval_episodes": args.eval_episodes,
    }

    save_dir = (
        f"{args.save_dir}/{args.env}/{args.model}/{time.strftime('%Y%m%d_%H%M%S')}"
    )

    if args.model == "ppo":
        trainer = PPOTrainer(env, config, save_dir=save_dir)
    elif args.model == "sac":
        trainer = SACTrainer(env, config, save_dir=save_dir)
    elif args.model == "rainbow_dqn":
        trainer = RainbowDQNTrainer(env, config, save_dir=save_dir)
    elif args.model == "discrete_sac":
        trainer = DiscreteSACTrainer(env, config, save_dir=save_dir)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    trainer.train()


if __name__ == "__main__":
    main()
