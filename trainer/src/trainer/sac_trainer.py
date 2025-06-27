"""
SAC
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor(Haarnoja, et al. 2018)
https://arxiv.org/pdf/1812.05905

Note
- introduces a temperature parameter alpha(SACv2) to control policy entropy, unlike the original paper
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import ReplayBuffer
from model.sac import SACPolicy, SACQNetwork, SACValueNetwork
from schema.config import SACConfig
from schema.result import SACUpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer


class SACTrainer(BaseTrainer):
    name = "sac"
    config_class = SACConfig

    def __init__(self, env_name: str, config: SACConfig, save_dir: str = "results/sac"):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        # Policy network: outputs mean and log_std for Gaussian policy
        self.actor = SACPolicy(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)

        # Q-networks: Q(s,a) -> R (continuous actions only in original SAC)
        self.critic1 = SACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.critic2 = SACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)

        # Value network: V(s) -> R
        self.value_net = SACValueNetwork(
            self.state_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.target_value_net = SACValueNetwork(
            self.state_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)

        # Initialize target value network
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        # Temperature parameter: alpha (learnable)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.config.device)
        self.alpha = self.log_alpha.exp()

        # Target entropy: -dim(action_space) for continuous actions
        self.target_entropy = -self.action_dim

        # Replay buffer
        # self.memory = ReplayBuffer(self.config.buffer.buffer_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.config.lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.config.lr
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.config.lr
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)

    def select_action(self, state, evaluation=False):
        """Select action using the policy network with reparameterization trick"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            if evaluation:
                # During evaluation, use mean action
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                # During training, sample from the policy
                action, _ = self.actor.sample(state)

            return {
                "action": action.squeeze(0).cpu().numpy(),
            }

    def _sample_transactions(self):
        """Sample batch from replay buffer"""
        state, action, reward, next_state, done = self.memory.sample(
            self.config.batch_size
        )

        state = torch.FloatTensor(state).to(self.config.device)
        next_state = torch.FloatTensor(next_state).to(self.config.device)
        action = torch.FloatTensor(action).to(self.config.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.config.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.config.device)

        return state, action, reward, next_state, done

    def _get_actor_action_and_log_prob(self, state):
        """Sample action from policy and compute log probability using reparameterization trick"""
        return self.actor.sample(state)

    def update(self) -> SACUpdateLoss:
        """Update all networks following the original SAC algorithm"""
        if len(self.memory) < self.config.batch_size:
            return

        state, action, reward, next_state, done = self._sample_transactions()

        # 1. Update Value Network (Equation 6 on SAC Paper)
        with torch.no_grad():
            # Sample action from current policy
            next_action, next_log_prob = self._get_actor_action_and_log_prob(state)

            # Compute Q-values for sampled actions
            q1_pi = self.critic1(state, next_action)
            q2_pi = self.critic2(state, next_action)
            min_q_pi = torch.min(q1_pi, q2_pi)

            # Target for value function: Q(s,a) - α*log π(a|s)
            value_target = min_q_pi - self.alpha * next_log_prob

        current_v = self.value_net(state)
        value_loss = F.mse_loss(current_v, value_target)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 2. Update Q-networks (Equation 9 on SAC Paper)
        with torch.no_grad():
            # Target for Q-function: r + γ*V(s')
            next_v = self.target_value_net(next_state)
            q_target = reward + (1 - done) * self.config.gamma * next_v

        # Current Q-values
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Q-function losses
        critic1_loss = F.mse_loss(current_q1, q_target)
        critic2_loss = F.mse_loss(current_q2, q_target)

        # Update Q-networks
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 3. Update Policy Network (Equation 13 on SAC Paper)
        # Sample new actions for policy update
        pi_action, log_prob = self._get_actor_action_and_log_prob(state)

        # Compute Q-values for policy actions
        q1_pi = self.critic1(state, pi_action)
        q2_pi = self.critic2(state, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Policy loss: maximize Q(s,a) - α*log π(a|s)
        # Equivalent to minimizing α*log π(a|s) - Q(s,a)
        # alpha: entropy regularization term
        # next_log_prob: log probability of the next action
        # negative next_log_prob is representative of the entropy
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Update temperature parameter α
        # Sample actions again to avoid computation graph issues
        with torch.no_grad():
            _, log_prob_detached = self._get_actor_action_and_log_prob(state)

        # Alpha loss: α * (log π(a|s) + target_entropy)
        alpha_loss = -(
            self.log_alpha * (log_prob_detached + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update alpha value
        self.alpha = self.log_alpha.exp()

        # 5. Soft update target networks
        self._soft_update_target()

        return SACUpdateLoss(
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            value_loss=value_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
        )

    def _soft_update_target(self):
        """Soft update target networks"""
        tau = self.config.tau
        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards, episode_losses, episode_steps = [], [], []

        while not done and len(episode_steps) < self.config.max_steps:
            action_info = self.select_action(state)
            action = action_info["action"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            self.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_rewards.append(reward)
            episode_steps.append(1)

            # Update networks if enough samples
            if len(self.memory) > self.config.batch_size:
                update_result = self.update()
                if update_result is not None:
                    episode_losses.append(update_result)

        return SingleEpisodeResult(
            episode_total_reward=round(np.sum(episode_rewards), 2),
            episode_steps=round(np.sum(episode_steps), 2),
            episode_losses=episode_losses,
        )

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "value_net": self.value_net.state_dict(),
                "target_value_net": self.target_value_net.state_dict(),
                "log_alpha": self.log_alpha,
            },
            self.model_file,
        )

    def load_model(self):
        checkpoint = torch.load(self.model_file)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.target_value_net.load_state_dict(checkpoint["target_value_net"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha = self.log_alpha.exp()

    def eval_mode_on(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.value_net.eval()
        self.target_value_net.eval()

    def eval_mode_off(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.value_net.train()
        self.target_value_net.train()
