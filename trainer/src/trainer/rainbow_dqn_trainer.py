"""
Rainbow DQN Trainer

Reference
---------
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](<https://arxiv.org/pdf/1710.02298>)
- [Prioritized Experience Replay](<https://arxiv.org/pdf/1511.05952>)
- [Dueling Network Architectures for Deep Reinforcement Learning](<https://arxiv.org/pdf/1511.06581>)
- [Multi-step Learning](<https://arxiv.org/pdf/1602.04485>)
- [Distributional Reinforcement Learning with Quantile Regression](<https://arxiv.org/pdf/1710.10044>)
- [Noisy Networks for Exploration](<https://arxiv.org/pdf/1706.10295>)
- [Categorical DQN](<https://arxiv.org/pdf/1707.06887>)
"""

import numpy as np
import torch
import torch.optim as optim

from model.rainbow_dqn import CategoricalDuelingNetwork
from schema.config import RainbowDQNConfig
from schema.result import RainbowDQNUpdateLoss, SingleEpisodeResult
from trainer.c51_trainer import C51Trainer


class RainbowDQNTrainer(C51Trainer):
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
        self.policy_net = CategoricalDuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            self.config.n_atoms,
            self.config.v_min,
            self.config.v_max,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.target_net = CategoricalDuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            self.config.n_atoms,
            self.config.v_min,
            self.config.v_max,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> dict:
        """
        Note
        ----
        - NoisyNet Exploration
        """
        if not eval_mode:
            self.policy_net.reset_noise()

        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            dist = self.policy_net(state)
            q_values = (dist * self.policy_net.support).sum(2)
            action = q_values.argmax().item()

        return {
            "action": action,
            "q_values": q_values.cpu().numpy(),
        }

    def _sample_transactions(self):
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.config.batch_size)
        )

        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.config.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update(self) -> RainbowDQNUpdateLoss:
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = (
            self._sample_transactions()
        )

        losses = self._update_distributional_rl(
            states, actions, rewards, next_states, dones
        )

        # td_errors for PER
        td_errors = losses.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        return RainbowDQNUpdateLoss(loss=loss.item())

    def _update_target_net(self):
        """Copies weights from the policy network to the target network."""
        if self.total_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.reset_noise()

    def train_episode(self) -> SingleEpisodeResult:
        """Trains the agent for a single episode."""
        state, _ = self.env.reset()
        done = False
        episode_rewards = []
        episode_losses = []
        episode_steps = 0

        for step in range(self.config.max_steps):
            self.total_steps += 1

            action_info = self.select_action(state)
            action = action_info["action"]

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_rewards.append(reward)
            episode_steps += 1

            update_result = self.update()
            if update_result:
                episode_losses.append(update_result)

            self._update_target_net()

            if done:
                break

        return SingleEpisodeResult(
            episode_total_reward=np.sum(episode_rewards),
            episode_steps=episode_steps,
            episode_losses=episode_losses,
        )

    def save_model(self):
        """Saves the policy network's state dictionary."""
        torch.save(self.policy_net.state_dict(), self.model_file)

    def load_model(self):
        """Loads the policy network's state dictionary."""
        self.policy_net.load_state_dict(torch.load(self.model_file))

    def eval_mode_on(self):
        """Sets the policy network to evaluation mode."""
        self.policy_net.eval()
        self.target_net.eval()

    def eval_mode_off(self):
        """Sets the policy network to training mode."""
        self.policy_net.train()
        self.target_net.train()
