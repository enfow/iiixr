import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import PPOMemory
from model.ppo import ContinuousActor, ContinuousCritic
from schema.config import PPOConfig
from schema.result import PPOUpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    def __init__(self, env_name: str, config: PPOConfig, save_dir: str = "results/ppo"):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        self.actor = ContinuousActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.critic = ContinuousCritic(
            self.state_dim,
            self.config.hidden_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.memory = PPOMemory()
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=self.config.lr,
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()),
            lr=self.config.lr,
        )

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)

        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions

        return {
            "action": action.cpu().numpy(),
            "logprob": logprob.item(),
        }

    def compute_returns(self, rewards, dones, gamma=0.99):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def _get_current_logprobs(self, states, actions):
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        logprobs = dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions
        return logprobs

    def _sample_transactions(self):
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.config.device)
        actions = torch.FloatTensor(np.array(self.memory.actions)).to(
            self.config.device
        )
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.config.device)
        returns = torch.FloatTensor(
            self.compute_returns(
                self.memory.rewards, self.memory.dones, self.config.gamma
            )
        ).to(self.config.device)
        advantages = returns - self.critic(states).squeeze().detach()
        return states, actions, old_logprobs, returns, advantages

    def _get_entropy(self, states):
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)  # Sum over action dimensions
        return entropy

    def update(self) -> PPOUpdateLoss:
        total_loss = 0
        states, actions, old_logprobs, returns, advantages = self._sample_transactions()

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.ppo_epochs):
            logprobs = self._get_current_logprobs(states, actions)

            # probability ratio: prob of action under new policy / prob of action under old policy
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            # clip the ratio to be between 1-epsilon and 1+epsilon
            surr2 = (
                torch.clamp(
                    ratio,
                    1 - self.config.clip_eps,
                    1 + self.config.clip_eps,
                )
                * advantages
            )
            # min(r_t(theta)A_t, clip(r_t(theta)A_t, 1-epsilon, 1+epsilon))
            actor_loss = -torch.min(surr1, surr2).mean()
            # a squared-error loss (V_theta(st) − V_target(st))2
            critic_loss = torch.nn.functional.mse_loss(
                self.critic(states).squeeze(), returns
            )

            # Entropy bonus
            entropy = self._get_entropy(states)
            entropy_loss = -self.config.entropy_coef * entropy.mean()
            total_actor_loss = actor_loss + entropy_loss

            # Update actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            total_loss += actor_loss.item() + critic_loss.item() + entropy_loss.item()
        # on-policy: clear memory
        self.memory.clear()

        return PPOUpdateLoss(
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            entropy_loss=entropy_loss.item(),
        )

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards, episode_losses, episode_steps = [], [], [0]

        while sum(episode_steps) < self.config.n_transactions:
            action_info = self.select_action(state)
            action, logprob = action_info["action"], action_info["logprob"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.store(state, action, logprob, reward, done)
            state = next_state
            episode_rewards.append(reward)
            episode_steps[-1] += 1

            if done:
                state, _ = self.env.reset()
                episode_steps.append(0)

        episode_steps = round(np.mean(episode_steps))

        update_result = self.update()

        if update_result is not None:
            episode_losses.append(update_result)

        return SingleEpisodeResult(
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            episode_losses=episode_losses,
        )

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            self.model_file,
        )

    def load_model(self):
        self.actor.load_state_dict(torch.load(self.model_file)["actor"])
        self.critic.load_state_dict(torch.load(self.model_file)["critic"])

    def eval_mode_on(self):
        self.actor.eval()
        self.critic.eval()

    def eval_mode_off(self):
        self.actor.train()
        self.critic.train()
