"""
TD3
https://arxiv.org/pdf/1802.09477
"""

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import ReplayBuffer
from model.td3 import TD3Actor, TD3Critic
from schema.config import TD3Config
from schema.result import SingleEpisodeResult, TD3UpdateLoss
from trainer.base_trainer import BaseTrainer


class TD3Trainer(BaseTrainer):
    name = "td3"
    config_class = TD3Config

    def __init__(self, env_name: str, config: TD3Config, save_dir: str = "results/td3"):
        super().__init__(env_name, config, save_dir)
        self.total_it = 0

    def _init_models(self):
        self.max_action = float(self.env.action_space.high[0])
        self.actor = TD3Actor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            n_layers=self.config.n_layers,
            max_action=self.max_action,
        ).to(self.config.device)

        self.critic = TD3Critic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Replay buffer
        self.memory = ReplayBuffer(self.config.buffer_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr)

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.config.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            action = action + np.random.normal(
                0, self.max_action * self.config.exploration_noise, size=self.action_dim
            )

        action = np.clip(action, -self.max_action, self.max_action)
        return {"action": action}

    def update(self) -> TD3UpdateLoss:
        if len(self.memory) < self.config.batch_size:
            return None

        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = self.memory.sample(
            self.config.batch_size
        )

        state = torch.FloatTensor(state).to(self.config.device)
        action = torch.FloatTensor(action).to(self.config.device)
        reward = (
            torch.FloatTensor(reward).unsqueeze(1).to(self.config.device)
        )  # Add dimension
        next_state = torch.FloatTensor(next_state).to(self.config.device)
        done = (
            torch.BoolTensor(done).unsqueeze(1).to(self.config.device)
        )  # Add dimension

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.config.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.config.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data
                    + (1 - self.config.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data
                    + (1 - self.config.tau) * target_param.data
                )

        return TD3UpdateLoss(
            actor_loss=actor_loss.item() if actor_loss is not None else 0.0,
            critic_loss=critic_loss.item(),
        )

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards, episode_losses, episode_steps = [], [], []
        step = 0

        while not done and step < self.config.max_steps:
            action_info = self.select_action(state)
            action = action_info["action"]

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            self.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_rewards.append(reward)
            step += 1

            # Update networks
            if len(self.memory) >= self.config.batch_size:
                update_result = self.update()
                if update_result is not None:
                    episode_losses.append(update_result)

        return SingleEpisodeResult(
            episode_total_reward=np.sum(episode_rewards),
            episode_steps=step,
            episode_losses=episode_losses,
        )

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            self.model_file,
        )

    def load_model(self):
        checkpoint = torch.load(self.model_file)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

    def eval_mode_on(self):
        self.actor.eval()
        self.critic.eval()

    def eval_mode_off(self):
        self.actor.train()
        self.critic.train()
