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

from model.rainbow_dqn import CategoricalDuelingNetwork
from schema.config import RainbowDQNConfig
from schema.result import RainbowDQNUpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer

V_MIN_MAX = {
    "BipedalWalker-v3": (-200.0, 300.0),
    "BipedalWalkerHardcore-v3": (-200.0, 300.0),
    "LunarLander-v3": (-400.0, 400.0),
}


class RainbowDQNTrainer(BaseTrainer):
    name = "rainbow_dqn"
    config_class = RainbowDQNConfig

    def __init__(
        self,
        env_name: str,
        config: RainbowDQNConfig,
        save_dir: str = "results/rainbow_dqn",
    ):
        if env_name in V_MIN_MAX:
            config["v_min"], config["v_max"] = V_MIN_MAX[env_name]
            print(
                f"V_MIN, V_MAX Automatically set: {config['v_min']}, {config['v_max']}"
            )

        super().__init__(env_name, config, save_dir)

        self.total_steps = 0

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

    def select_action(self, state: np.ndarray) -> dict:
        """Selects an action using the policy network."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            dist = self.policy_net(state)
            q_values = (dist * self.policy_net.support).sum(2)
            action = q_values.argmax().item()

        self.policy_net.reset_noise()
        return {
            "action": action,
            "q_values": q_values.cpu().numpy(),
        }

    def _sample_transactions(self):
        """Samples a batch of transitions from the replay buffer."""
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
        """Updates the network weights."""
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = (
            self._sample_transactions()
        )

        with torch.no_grad():
            # policy net output: batch of probability distribution over atoms
            # shape: (batch_size, action_dim, n_atoms)
            # probability of each atom for each action

            # support tensor: shape: (n_atoms,) from v_min to v_max
            # value of each atom

            # probability of each atom * value of each atom => expected value of each atom
            # sum of expected value of each atom => expected value of each action => q_values
            next_q_values_policy = (
                self.policy_net(next_states) * self.policy_net.support
            ).sum(2)
            # select best q value action
            next_actions = next_q_values_policy.argmax(1)

            next_dist_target = self.target_net(next_states)
            next_dist = next_dist_target[range(self.config.batch_size), next_actions]

            # distributional multi-step return

            # rewards: accumulated reward from n-step return (PER already applied gamma^n_steps when it pushed)
            # (1 - dones) * gamma^n_steps * support: if is done, all atoms are 0
            # if not done, all atoms are gamma^n_steps * support
            # t_z is the target distribution
            t_z = (
                rewards  # shape: (batch_size, 1)
                + (1 - dones)  # shape: (batch_size, 1)
                * (self.config.gamma**self.config.n_steps)  # shape: (batch_size, 1)
                * self.target_net.support  # shape: (n_atoms,)
            )  # shape: (batch_size, n_atoms)

            # clamp to valid range
            t_z = t_z.clamp(min=self.config.v_min, max=self.config.v_max)

            # b: the index of the atom that the target distribution is closest to
            # delta_z is the width of each atom
            b = (t_z - self.config.v_min) / self.target_net.delta_z

            # l: the lower bound of the atom that the target distribution is closest to
            # u: the upper bound of the atom that the target distribution is closest to
            l = b.floor().long()
            u = b.ceil().long()

            # clamp to valid range
            l = torch.clamp(l, 0, self.config.n_atoms - 1)
            u = torch.clamp(u, 0, self.config.n_atoms - 1)

            # proj_dist: the projected distribution
            proj_dist = torch.zeros_like(next_dist)

            # offset: the offset of the atom that the target distribution is closest to
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

            # index_add: add the value of the next_dist to the proj_dist at the index of l + offset and u + offset
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        # dist: the predicted distribution
        dist = self.policy_net(states)

        # log_p: log of the predicted distribution
        log_p = torch.log(dist[range(self.config.batch_size), actions])

        # loss: cross-entropy loss
        loss = -(proj_dist * log_p).sum(1)

        # td_errors for PER
        td_errors = loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights * loss).mean()

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

    def train_episode(self) -> SingleEpisodeResult:
        """Trains the agent for a single episode."""
        state, _ = self.env.reset()
        done = False
        episode_rewards = []
        episode_losses = []
        episode_steps = 0

        for step in range(self.config.max_steps):
            self.total_steps += 1

            # Select action
            action_info = self.select_action(state)
            action = action_info["action"]

            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            self.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_rewards.append(reward)
            episode_steps += 1

            # Update networks if enough samples
            if len(self.memory) >= self.config.batch_size:
                update_result = self.update()
                if update_result is not None:
                    episode_losses.append(update_result)

            # Update target network
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
