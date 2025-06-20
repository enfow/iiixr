from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import ReplayBuffer
from model.sac import SACPolicy, SACQNetwork
from trainer.base_trainer import BaseConfig, BaseTrainer


@dataclass
class SACConfig(BaseConfig):
    tau: float = 0.005
    entropy_coef: float = 1.0
    start_steps: int = 1000


class SACTrainer(BaseTrainer):
    def __init__(self, env, config, save_dir="results/sac"):
        config = SACConfig.from_dict(config)
        super().__init__(env, config, save_dir)

    def _init_models(self):
        self.actor = SACPolicy(self.state_dim, self.action_dim).to(self.config.device)
        self.critic1 = SACQNetwork(self.state_dim, self.action_dim).to(
            self.config.device
        )
        self.critic2 = SACQNetwork(self.state_dim, self.action_dim).to(
            self.config.device
        )
        self.target_critic1 = SACQNetwork(self.state_dim, self.action_dim).to(
            self.config.device
        )
        self.target_critic2 = SACQNetwork(self.state_dim, self.action_dim).to(
            self.config.device
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.buffer = ReplayBuffer(self.config.buffer_size)

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
            if self.is_discrete:
                logits = self.actor(state)
                probs = F.softmax(logits, dim=-1)
                action = torch.distributions.Categorical(probs).sample()
                return {
                    "action": action.item(),
                }
            else:
                mean = self.actor(state)
                action = torch.tanh(mean)
                return {
                    "action": action.squeeze(0).cpu().numpy(),
                }

    def update(self):
        if len(self.buffer) < self.config.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(
            self.config.batch_size
        )
        state = torch.FloatTensor(state).to(self.config.device)
        next_state = torch.FloatTensor(next_state).to(self.config.device)
        action = (
            torch.FloatTensor(action).to(self.config.device)
            if not self.is_discrete
            else torch.LongTensor(action).to(self.config.device)
        )
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.config.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.config.device)

        if self.is_discrete:
            action = F.one_hot(action, self.action_dim).float().to(self.config.device)
        else:
            action = torch.FloatTensor(action).to(self.config.device)

        with torch.no_grad():
            next_logits = self.actor(next_state)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            all_actions = torch.eye(self.action_dim).to(self.config.device)
            next_q1 = torch.stack(
                [
                    self.target_critic1(
                        next_state, all_actions[i].expand(next_state.size(0), -1)
                    )
                    for i in range(self.action_dim)
                ],
                dim=1,
            ).squeeze(-1)
            next_q2 = torch.stack(
                [
                    self.target_critic2(
                        next_state, all_actions[i].expand(next_state.size(0), -1)
                    )
                    for i in range(self.action_dim)
                ],
                dim=1,
            ).squeeze(-1)
            target_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (target_q - self.alpha * next_log_probs)).sum(
                dim=1, keepdim=True
            )
            target = reward + (1 - done) * self.config.gamma * next_v

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        all_actions = torch.eye(self.action_dim).to(self.config.device)
        q1_pi = torch.stack(
            [
                self.critic1(state, all_actions[i].expand(state.size(0), -1))
                for i in range(self.action_dim)
            ],
            dim=1,
        ).squeeze(-1)
        q2_pi = torch.stack(
            [
                self.critic2(state, all_actions[i].expand(state.size(0), -1))
                for i in range(self.action_dim)
            ],
            dim=1,
        ).squeeze(-1)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (
            (probs * (self.alpha.detach() * log_probs - min_q_pi)).sum(dim=1).mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

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
        return (
            actor_loss.item()
            + critic1_loss.item()
            + critic2_loss.item()
            + alpha_loss.item()
        )

    def train_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        losses = []
        for step in range(self.config.max_steps):
            if self.total_steps < self.config.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state)["action"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            self.total_steps += 1

            loss = self.update()
            if loss is not None:
                losses.append(loss)

            if done:
                break

        return {
            "total_reward": total_reward,
            "losses": losses,
        }

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
