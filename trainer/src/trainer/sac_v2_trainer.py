"""
SACv2
Soft Actor-Critic Algorithms and Applications(Haarnoja, et al. 2018)
https://arxiv.org/pdf/1812.05905

Differences from SAC:
- No explicit V-function
- introduce double target networks for Q-function
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import ReplayBuffer
from model.sac import SACPolicy, SACQNetwork
from schema.config import SACConfig
from schema.result import SACUpdateLoss
from trainer.sac_trainer import SACTrainer


class SACV2Trainer(SACTrainer):
    name = "sac_v2"
    config_class = SACConfig

    def __init__(
        self, env_name: str, config: SACConfig, save_dir: str = "results/sac_v2"
    ):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        # Policy network
        self.actor = SACPolicy(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)

        # Q-networks
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

        self.target_critic1 = SACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.target_critic2 = SACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)

        # Double Target Q-networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Temperature parameter: alpha (learnable)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.config.device)
        self.alpha = self.log_alpha.exp()

        # Target entropy
        self.target_entropy = -self.action_dim

        # Replay buffer
        self.memory = ReplayBuffer(self.config.buffer_size)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.config.lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.config.lr
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)

    def update(self) -> SACUpdateLoss:
        """Update all networks following the modern SAC algorithm (no explicit V-function)"""
        if len(self.memory) < self.config.batch_size:
            return

        state, action, reward, next_state, done = self._sample_transactions()

        # 1. Update Q-networks (Critic)
        with torch.no_grad():
            next_action, next_log_prob = self._get_actor_action_and_log_prob(next_state)

            target_q1_next = self.target_critic1(next_state, next_action)
            target_q2_next = self.target_critic2(next_state, next_action)
            min_target_q_next = torch.min(target_q1_next, target_q2_next)

            # V(s') = E[Q_target(s',a') - α*logπ(a'|s')]
            next_state_value = min_target_q_next - self.alpha * next_log_prob

            # Q-target: y = r + γ * V(s')
            q_target = reward + (1 - done) * self.config.gamma * next_state_value

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        critic1_loss = F.mse_loss(current_q1, q_target)
        critic2_loss = F.mse_loss(current_q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 2. Update Policy Network (Actor)
        pi_action, log_prob = self._get_actor_action_and_log_prob(state)

        q1_pi = self.critic1(state, pi_action)
        q2_pi = self.critic2(state, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Update temperature parameter α
        with torch.no_grad():
            _, log_prob_detached = self._get_actor_action_and_log_prob(state)

        alpha_loss = (
            self.log_alpha * (-log_prob_detached - self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 4. Soft update target networks
        for param, target_param in zip(
            self.critic1.parameters(), self.target_critic1.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic2.parameters(), self.target_critic2.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

        return SACUpdateLoss(
            actor_loss=actor_loss.item(),
            value_loss=0,
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha_loss=alpha_loss.item(),
        )

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "log_alpha": self.log_alpha,
            },
            self.model_file,
        )

    def load_model(self):
        checkpoint = torch.load(self.model_file)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.log_alpha.data = checkpoint["log_alpha"]
        self.alpha = self.log_alpha.exp()
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def eval_mode_on(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()

    def eval_mode_off(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.target_critic1.train()
        self.target_critic2.train()
