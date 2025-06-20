import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import PrioritizedReplayBuffer
from model.rainbow_dqn import DuelingNetwork
from trainer.base_trainer import BaseTrainer


class RainbowDQNTrainer(BaseTrainer):
    def __init__(self, env, config, save_dir="results/rainbow_dqn"):
        super().__init__(env, config, save_dir)

        hidden_dim = config.get("hidden_dim", 256)
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update", 10)

        # Networks
        self.policy_net = DuelingNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.device)
        self.target_net = DuelingNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.get("lr", 3e-4)
        )

        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.get("buffer_size", 100000),
            alpha=config.get("alpha", 0.6),
            beta_start=config.get("beta_start", 0.4),
            beta_frames=config.get("beta_frames", 100000),
        )

        self.total_steps = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # TD errors and weighted loss
        td_errors = (q_values - target_q_values).detach().cpu().numpy().squeeze()
        loss = (
            weights * F.smooth_l1_loss(q_values, target_q_values, reduction="none")
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode_losses.append(loss.item())

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Target network update
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, episodes=1000, max_steps=1000):
        self.total_steps = 0
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            self.episode_losses = []

            for t in range(max_steps):
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

            # Save best model
            if total_reward > self.best_score:
                self.best_score = total_reward
                torch.save(self.policy_net.state_dict(), self.model_file)

            self._log_metrics()
