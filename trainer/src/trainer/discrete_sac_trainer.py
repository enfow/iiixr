"""
Discrete SAC implementation
- Ref: https://arxiv.org/pdf/1910.07207

Deferences with SAC:
- Q function moves from (s, a) -> R  =>  (s) -> R^A
- Directly output action distribution instead of mean and cov
  - Policy: phi: S -> R^2|A|  =>  phi: S -> [0, 1]^|A|
- Do not need to use  monte carlo estimate for V(s)
- Do not need to use monte carlo estimate for Temperature parameter Alpha
- Do not need to use reparametrization trick for policy update
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import ReplayBuffer
from model.discrete_sac import DiscreteSACPolicy, DiscreteSACQNetwork
from model.sac import SACPolicy
from schema.config import SACConfig
from schema.result import DiscreteSACUpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer


class DiscreteSACTrainer(BaseTrainer):
    name = "discrete_sac"
    config_class = SACConfig

    def __init__(
        self, env_name: str, config: SACConfig, save_dir: str = "results/discrete_sac"
    ):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        self.actor = DiscreteSACPolicy(
            self.state_dim,
            self.action_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.critic1 = DiscreteSACQNetwork(
            self.state_dim,
            self.action_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.critic2 = DiscreteSACQNetwork(
            self.state_dim,
            self.action_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.target_critic1 = DiscreteSACQNetwork(
            self.state_dim,
            self.action_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.target_critic2 = DiscreteSACQNetwork(
            self.state_dim,
            self.action_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.memory = ReplayBuffer(self.config.buffer_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.config.lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.config.lr
        )
        self.target_entropy = -np.log(1.0 / self.action_dim) * self.config.entropy_coef
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.config.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(probs).sample()
            return {
                "action": action.item(),
            }

    def _sample_transactions(self):
        state, action, reward, next_state, done = self.memory.sample(
            self.config.batch_size
        )
        state = torch.FloatTensor(state).to(self.config.device)
        next_state = torch.FloatTensor(next_state).to(self.config.device)
        action = torch.LongTensor(action).to(self.config.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.config.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.config.device)
        return state, action, reward, next_state, done

    def update(self) -> DiscreteSACUpdateLoss:
        if len(self.memory) < self.config.batch_size:
            return

        state, action, reward, next_state, done = self._sample_transactions()

        # Target Q
        with torch.no_grad():
            next_logits = self.actor(next_state)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            target_q1 = self.target_critic1(next_state)
            target_q2 = self.target_critic2(next_state)
            target_q = torch.min(target_q1, target_q2)
            next_v = (next_probs * (target_q - self.alpha * next_log_probs)).sum(
                dim=1, keepdim=True
            )
            target = reward + (1 - done) * self.config.gamma * next_v

        # Current Q
        q1 = self.critic1(state).gather(1, action.unsqueeze(1))
        q2 = self.critic2(state).gather(1, action.unsqueeze(1))

        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        q1_all = self.critic1(state)
        q2_all = self.critic2(state)
        min_q = torch.min(q1_all, q2_all)

        actor_loss = (
            (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update temperature parameter
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Soft updates
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

        return DiscreteSACUpdateLoss(
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha_loss=alpha_loss.item(),
        )

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards, episode_losses, episode_steps = [], [], 0

        for step in range(self.config.max_steps):
            action = self.select_action(state)["action"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_rewards.append(reward)

            loss = self.update()
            if loss is not None:
                episode_losses.append(loss)

            if done:
                episode_steps = step
                break

        return SingleEpisodeResult(
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            episode_losses=episode_losses,
        )

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic1.state_dict(),
            },
            self.model_file,
        )

    def load_model(self):
        self.actor.load_state_dict(torch.load(self.model_file)["actor"])
        self.critic1.load_state_dict(torch.load(self.model_file)["critic"])

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
