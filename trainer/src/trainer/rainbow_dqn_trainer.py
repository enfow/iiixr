"""
Rainbow DQN
Rainbow: Combining Improvements in Deep Reinforcement Learning(Hessel, et al. 2017)
https://arxiv.org/pdf/1710.02298

Note
- use Prioritized Experience Replay(Schaul, et al. 2015)
- use Dueling Network Architecture(Wang, et al. 2015)
- use Multi-step Learning(Mnih, et al. 2015)
- use Distributional RL(Bellemare, et al. 2017)
- use Noisy Networks(Fortunato, et al. 2017)
- use Categorical DQN(Bellemare, et al. 2017)
"""

import numpy as np
import torch
import torch.optim as optim

from model.buffer import PrioritizedReplayBuffer
from model.rainbow_dqn import CategoricalDuelingNetwork
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
        self.total_steps = 0

    def _init_models(self):
        # Networks
        self.policy_net = CategoricalDuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.config.n_atoms,
            self.config.v_min,
            self.config.v_max,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.target_net = CategoricalDuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.config.n_atoms,
            self.config.v_min,
            self.config.v_max,
            n_layers=self.config.n_layers,
        ).to(self.config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            alpha=self.config.alpha,
            beta_start=self.config.beta_start,
            beta_frames=self.config.beta_frames,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
        )

    def select_action(self, state: np.ndarray) -> dict:
        """Selects an action using the policy network."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.reset_noise()  # Reset noise after action selection
        return {
            "action": q_values.argmax().item(),
            "q_values": q_values.cpu().numpy(),
        }

    def _sample_transactions(self):
        """Samples a batch of transitions from the replay buffer."""
        states, actions, rewards, next_states, dones, indices, weights = (
            self.replay_buffer.sample(self.config.batch_size)
        )

        states = torch.FloatTensor(states).to(self.config.device)
        # Actions is a 1D LongTensor of shape [batch_size]
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.config.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update(self) -> RainbowDQNUpdateLoss:
        """Updates the network weights."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = (
            self._sample_transactions()
        )

        # Calculate the target distribution for the N-step return
        with torch.no_grad():
            # Double DQN: select action with policy_net, evaluate with target_net
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_dist = self.target_net.dist(next_states)
            next_dist = next_dist[range(self.config.batch_size), next_actions]

            # Compute the projection of the target distribution onto the support
            t_z = (
                rewards
                + (1 - dones)
                * (self.config.gamma**self.config.n_steps)
                * self.target_net.support
            )
            t_z = t_z.clamp(min=self.config.v_min, max=self.config.v_max)
            b = (t_z - self.config.v_min) / self.target_net.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Calculate the weights for the lower and upper bins
            # These are the interpolation weights for distributing probability
            lw = u.float() - b
            uw = b - l.float()

            # Handle the case where l = u (i.e., b is an integer)
            # In this case, the probability mass should go entirely to bin l.
            # We set the lower weight to 1 and upper to 0 for these cases.
            eq_mask = l == u
            lw[eq_mask] = 1.0
            uw[eq_mask] = 0.0

            # Create the projected distribution tensor
            proj_dist = torch.zeros(next_dist.size(), device=self.config.device)

            # Create offset to map from 2D (batch, atom) to 1D
            offset = (
                torch.linspace(
                    0,
                    (self.config.batch_size - 1) * self.config.n_atoms,
                    self.config.batch_size,
                )
                .long()
                .unsqueeze(1)
                .expand(self.config.batch_size, self.config.n_atoms)
                .to(self.config.device)
            )

            # Distribute the probability mass to the lower and upper bins
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * lw).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * uw).view(-1)
            )

        # Calculate the cross-entropy loss
        dist = self.policy_net.dist(states)
        log_p = torch.log(dist[range(self.config.batch_size), actions])

        # The loss is the negative dot product of the projected target distribution and the log of the predicted distribution
        loss = -(proj_dist * log_p).sum(1)

        # Update priorities in the replay buffer
        td_errors = loss.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Apply importance-sampling weights to the loss
        loss = (weights * loss).mean()

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset noise in the noisy linear layers for exploration
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        return RainbowDQNUpdateLoss(loss=loss.item())

    def _update_target_net(self):
        """Copies weights from the policy network to the target network."""
        if self.total_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

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

            self.replay_buffer.push(state, action, reward, next_state, done)

            loss_info = self.update()
            if loss_info:
                episode_losses.append(loss_info.loss)

            self._update_target_net()

            state = next_state
            episode_rewards.append(reward)

            if done:
                episode_steps = step + 1
                break

        # Calculate average loss, handling the case of no updates
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        return SingleEpisodeResult(
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            episode_losses=[RainbowDQNUpdateLoss(loss=avg_loss)],
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

    def eval_mode_off(self):
        """Sets the policy network to training mode."""
        self.policy_net.train()
