import argparse

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
    args = parser.parse_args()

    env = gym.make(args.env)
    config = {
        "max_steps": args.max_steps,
        "lr": args.lr,
        "gamma": args.gamma,
        "hidden_dim": args.hidden_dim,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "target_update": args.target_update,
        # SAC specific hyperparameters
        "tau": 0.005,
        "tune_alpha": True,
        "start_steps": 1000,  # Reduced for simpler discrete envs
        "update_every": 1,
        "max_grad_norm": 1.0,
    }

    if args.model == "ppo":
        trainer = PPOTrainer(env, config, save_dir=f"{args.save_dir}/ppo")
    elif args.model == "sac":
        trainer = SACTrainer(env, config, save_dir=f"{args.save_dir}/sac")
    elif args.model == "rainbow_dqn":
        trainer = RainbowDQNTrainer(
            env, config, save_dir=f"{args.save_dir}/rainbow_dqn"
        )
    elif args.model == "discrete_sac":
        trainer = DiscreteSACTrainer(
            env, config, save_dir=f"{args.save_dir}/discrete_sac"
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    trainer.train(episodes=args.episodes, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
