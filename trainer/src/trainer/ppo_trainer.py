"""
PPO Trainer

Reference
---------
- [Proximal Policy Optimization Algorithms](<https://arxiv.org/pdf/1707.06347>)
"""

import numpy as np
import torch
import torch.optim as optim

from model.ppo import ContinuousActor, ContinuousCritic
from schema.config import PPOConfig
from schema.result import PPOUpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    name = "ppo"
    config_class = PPOConfig

    def __init__(self, env_name: str, config: PPOConfig, save_dir: str = "results/ppo"):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        self.actor = ContinuousActor(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
            use_layernorm=self.config.model.use_layernorm,
        ).to(self.config.device)
        self.critic = ContinuousCritic(
            self.state_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
            use_layernorm=self.config.model.use_layernorm,
        ).to(self.config.device)
        self.memory = None
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=self.config.lr,
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()),
            lr=self.config.lr,
        )

    def select_action(self, state, eval_mode: bool = False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.config.device)
            mean, log_std = self.actor(state)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)

            if eval_mode:
                action = mean
                logprob = dist.log_prob(action).sum(dim=-1)
            else:
                action = dist.sample()
                logprob = dist.log_prob(action).sum(dim=-1)

            return {
                "action": action.cpu().numpy(),
                "logprob": logprob.item(),
            }

    def _get_current_logprobs(self, states, actions):
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        logprobs = dist.log_prob(actions).sum(dim=-1)
        return logprobs

    def _get_entropy(self, states):
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)
        return entropy

    def collect_episode_data(self):
        """Collect data for one complete episode"""
        state, _ = self.env.reset()
        episode_states, episode_actions, episode_logprobs = [], [], []
        episode_rewards, episode_dones = [], []

        while True:
            action_info = self.select_action(state)
            action, logprob = action_info["action"], action_info["logprob"]

            episode_states.append(state)
            episode_actions.append(action)
            episode_logprobs.append(logprob)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            episode_rewards.append(reward)
            episode_dones.append(done)

            state = next_state

            if done:
                break

        return {
            "states": episode_states,
            "actions": episode_actions,
            "logprobs": episode_logprobs,
            "rewards": episode_rewards,
            "dones": episode_dones,
            "episode_length": len(episode_states),
        }

    def compute_monte_carlo_returns(self, rewards, dones):
        """Compute Monte Carlo returns for a single episode"""
        returns = []
        _return = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                _return = 0
            _return = r + self.config.gamma * _return
            returns.insert(0, _return)
        return torch.FloatTensor(returns).to(self.config.device)

    def compute_episode_gae(self, episode_data):
        """Compute GAE for a single episode"""
        states = torch.FloatTensor(np.array(episode_data["states"])).to(
            self.config.device
        )
        rewards = torch.tensor(episode_data["rewards"], dtype=torch.float32).to(
            self.config.device
        )
        dones = torch.tensor(episode_data["dones"], dtype=torch.float32).to(
            self.config.device
        )

        with torch.no_grad():
            values = self.critic(states).squeeze()
            if values.dim() == 0:  # Handle single-step episodes
                values = values.unsqueeze(0)

            # For episode-based collection, compute next values properly
            next_values = torch.zeros_like(values)
            if len(values) > 1:
                next_values[:-1] = values[1:]  # V(s_{t+1}) for t < T-1
            next_values[-1] = 0.0  # V(s_T) = 0 for terminal state

        advantages = torch.zeros_like(rewards)
        gae_advantage = 0

        for t in reversed(range(len(rewards))):
            td_error = (
                rewards[t]
                + self.config.gamma * next_values[t] * (1 - dones[t])
                - values[t]
            )

            gae_advantage = (
                td_error
                + self.config.gamma
                * self.config.gae_lambda
                * (1 - dones[t])
                * gae_advantage
            )
            advantages[t] = gae_advantage

        returns = advantages + values
        return {
            "states": states,
            "actions": episode_data["actions"],
            "logprobs": episode_data["logprobs"],
            "returns": returns,
            "advantages": advantages,
            "rewards": rewards,
        }

    def compute_episode_mc_returns(self, episode_data):
        """Compute Monte Carlo returns for a single episode"""
        states = torch.FloatTensor(np.array(episode_data["states"])).to(
            self.config.device
        )
        returns = self.compute_monte_carlo_returns(
            episode_data["rewards"], episode_data["dones"]
        )

        with torch.no_grad():
            values = self.critic(states).squeeze()
            if values.dim() == 0:  # Handle single-step episodes
                values = values.unsqueeze(0)

        advantages = returns - values

        return {
            "states": states,
            "actions": episode_data["actions"],
            "logprobs": episode_data["logprobs"],
            "returns": returns,
            "advantages": advantages,
            "rewards": torch.tensor(episode_data["rewards"], dtype=torch.float32).to(
                self.config.device
            ),
        }

    def get_all_episode_data(self, all_episode_data):
        all_states = torch.cat([ep["states"] for ep in all_episode_data], dim=0)
        all_actions = torch.FloatTensor(
            np.concatenate([ep["actions"] for ep in all_episode_data])
        ).to(self.config.device)
        all_logprobs = torch.FloatTensor(
            np.concatenate([ep["logprobs"] for ep in all_episode_data])
        ).to(self.config.device)
        all_returns = torch.cat([ep["returns"] for ep in all_episode_data], dim=0)
        all_advantages = torch.cat([ep["advantages"] for ep in all_episode_data], dim=0)
        return all_states, all_actions, all_logprobs, all_returns, all_advantages

    def update(self, all_episode_data) -> PPOUpdateLoss:
        """Update policy using collected episode data"""
        all_states, all_actions, all_logprobs, all_returns, all_advantages = (
            self.get_all_episode_data(all_episode_data)
        )

        # Normalize advantages
        if self.config.normalize_advantages:
            all_advantages = (all_advantages - all_advantages.mean()) / (
                all_advantages.std() + 1e-8
            )

        # Perform PPO updates
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0

        for _ in range(self.config.ppo_epochs):
            logprobs = self._get_current_logprobs(all_states, all_actions)

            # importance sampling ratio
            ratio = torch.exp(logprobs - all_logprobs)
            surr1 = ratio * all_advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
                * all_advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = torch.nn.functional.mse_loss(
                self.critic(all_states).squeeze(), all_returns
            )

            # Entropy bonus
            entropy = self._get_entropy(all_states)
            entropy_loss = -self.config.entropy_coef * entropy.mean()
            total_actor_loss_step = actor_loss + entropy_loss

            # Update actor
            self.actor_optimizer.zero_grad()
            total_actor_loss_step.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()

        return PPOUpdateLoss(
            actor_loss=total_actor_loss / self.config.ppo_epochs,
            critic_loss=total_critic_loss / self.config.ppo_epochs,
            entropy_loss=total_entropy_loss / self.config.ppo_epochs,
        )

    def train_episode(self) -> SingleEpisodeResult:
        """Train using episode-by-episode collection"""
        all_episode_data = []
        total_rewards = []
        episode_lengths = []
        total_steps = 0
        episode_count = 0

        # Collect multiple episodes until we have enough transitions
        while total_steps < self.config.n_transactions:
            episode_data = self.collect_episode_data()
            episode_count += 1
            episode_length = episode_data["episode_length"]
            total_steps += episode_length
            episode_lengths.append(episode_length)
            total_rewards.extend(episode_data["rewards"])

            # Compute returns/advantages for this episode
            if self.config.gae:
                processed_episode = self.compute_episode_gae(episode_data)
            else:
                processed_episode = self.compute_episode_mc_returns(episode_data)

            all_episode_data.append(processed_episode)

        # Update policy using all collected episodes
        update_result = self.update(all_episode_data)

        return SingleEpisodeResult(
            episode_total_reward=round(np.sum(total_rewards) / episode_count, 2),
            episode_steps=round(np.mean(episode_lengths), 2),
            episode_losses=[update_result],
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
