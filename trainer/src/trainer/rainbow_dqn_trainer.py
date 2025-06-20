from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import PrioritizedReplayBuffer
from model.rainbow_dqn import DuelingNetwork
from trainer.base_trainer import BaseConfig, BaseTrainer


@dataclass
class RainbowDQNConfig(BaseConfig):
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000
    target_update: int = 10


class RainbowDQNTrainer(BaseTrainer):
    def __init__(self, env, config, save_dir="results/rainbow_dqn"):
        config = RainbowDQNConfig.from_dict(config, env)
        super().__init__(env, config, save_dir)

    def _init_models(self):
        # Networks
        self.policy_net = DuelingNetwork(
            self.state_dim, self.action_dim, self.config.hidden_dim
        ).to(self.config.device)
        self.target_net = DuelingNetwork(
            self.state_dim, self.action_dim, self.config.hidden_dim
        ).to(self.config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            alpha=self.config.alpha,
            beta_start=self.config.beta_start,
            beta_frames=self.config.beta_frames,
        )

        self.total_steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = (
            self.replay_buffer.sample(self.config.batch_size)
        )

        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.config.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

        td_errors = (q_values - target_q_values).detach().cpu().numpy().squeeze()
        loss = (
            weights * F.smooth_l1_loss(q_values, target_q_values, reduction="none")
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode_losses.append(loss.item())

        self.replay_buffer.update_priorities(indices, td_errors)

        if self.total_steps % self.config.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        self.total_steps = 0
        for ep in range(self.config.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            self.episode_losses = []

            for t in range(self.config.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                self.update()

                state = next_state
                total_reward += reward
                self.total_steps += 1

                if done:
                    break

            avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0.0
            self.scores.append(total_reward)
            self.losses.append(avg_loss)

            if total_reward > self.best_score:
                self.best_score = total_reward
                self.save_model()
            self._log_metrics()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_file)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_file))

    def ready_to_evaluate(self):
        self.load_model()
        self.policy_net.eval()
