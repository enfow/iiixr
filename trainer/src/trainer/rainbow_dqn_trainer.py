"""
Rainbow DQN
Rainbow: Combining Improvements in Deep Reinforcement Learning(Hessel, et al. 2017)
https://arxiv.org/pdf/1710.02298
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from model.buffer import PrioritizedReplayBuffer
from model.rainbow_dqn import DuelingNetwork
from schema.config import RainbowDQNConfig
from schema.result import RainbowDQNUpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer


class RainbowDQNTrainer(BaseTrainer):
    name = "rainbow_dqn"
    config_class = RainbowDQNConfig

    def __init__(
        self,
        env_name: str,
        config: RainbowDQNConfig,
        save_dir: str = "results/rainbow_dqn",
    ):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        # Networks
        self.policy_net = DuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.target_net = DuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            n_layers=self.config.n_layers,
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
        return {
            "action": q_values.argmax(dim=1).item(),
            "q_values": q_values.detach().cpu().numpy(),
        }

    def _sample_transactions(self):
        states, actions, rewards, next_states, dones, indices, weights = (
            self.replay_buffer.sample(self.config.batch_size)
        )
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.config.device)
        return states, actions, rewards, next_states, dones, indices, weights

    def update(self) -> RainbowDQNUpdateLoss:
        if len(self.replay_buffer) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = (
            self._sample_transactions()
        )

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

        self.replay_buffer.update_priorities(indices, td_errors)

        return RainbowDQNUpdateLoss(
            loss=loss.item(),
        )

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards = []
        episode_losses = []
        episode_steps = 0

        for step in range(self.config.max_steps):
            action = self.select_action(state)["action"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.push(state, action, reward, next_state, done)

            loss = self.update()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_rewards.append(reward)

            if done:
                episode_steps = step
                break

        return SingleEpisodeResult(
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            episode_losses=episode_losses,
        )

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_file)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_file))

    def eval_mode_on(self):
        self.policy_net.eval()

    def eval_mode_off(self):
        self.policy_net.train()
