"""
C51 Trainer

Reference
---------
- [A Distributional Perspective on Reinforcement Learning](<https://arxiv.org/pdf/1707.06887>)
"""

import math

import numpy as np
import torch
import torch.optim as optim

from model.categorial_rl import CategoricalDQNNetwork
from schema.config import C51Config
from schema.result import C51UpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer

V_MIN_MAX = {
    "BipedalWalker-v3": (-200.0, 300.0),
    "BipedalWalkerHardcore-v3": (-200.0, 300.0),
    "LunarLander-v2": (-400.0, 400.0),
}


class C51Trainer(BaseTrainer):
    name = "c51"
    config_class = C51Config

    def __init__(
        self,
        env_name: str,
        config_dict: dict,
        save_dir: str = "results/c51",
    ):
        # automatically set v_min and v_max for known environments
        if env_name in V_MIN_MAX:
            if "v_min" not in config_dict or "v_max" not in config_dict:
                config_dict["v_min"], config_dict["v_max"] = V_MIN_MAX[env_name]
                print(
                    f"V_MIN, V_MAX Automatically set: {config_dict['v_min']}, {config_dict['v_max']}"
                )

        super().__init__(env_name, config_dict, save_dir)
        self.total_steps = 0
        if self.name == "c51":
            self.epsilon = self.config.eps_start

    def _init_models(self):
        self.policy_net = CategoricalDQNNetwork(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            self.config.n_atoms,
            self.config.v_min,
            self.config.v_max,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.target_net = CategoricalDQNNetwork(
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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> dict:
        """
        Note
        ----
        - Epsilon-Greedy Exploration
        """
        # epsilon decay
        self.epsilon = self.config.eps_end + (
            self.config.eps_start - self.config.eps_end
        ) * math.exp(-1.0 * self.total_steps / self.config.eps_decay)

        if not eval_mode and np.random.rand() < self.epsilon:
            # exploration
            action = self.env.action_space.sample()
            return {"action": action, "epsilon": self.epsilon}

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            q_values = self.policy_net.get_q_values(state_tensor)
            action = q_values.argmax().item()

        return {
            "action": action,
            "q_values": q_values.cpu().numpy(),
            "epsilon": self.epsilon,
        }

    def _sample_transactions(self):
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )

        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)

        return states, actions, rewards, next_states, dones

    def _update_distributional_rl(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
        - losses: shape: (batch_size,)
        """
        with torch.no_grad():
            # policy net output: batch of probability distribution over atoms
            # shape: (batch_size, action_dim, n_atoms)
            # probability of each atom for each action

            # support tensor: shape: (n_atoms,) from v_min to v_max
            # value of each atom

            # probability of each atom * value of each atom => expected value of each atom
            # sum of expected value of each atom => expected value of each action => q_values
            next_q_values_policy = self.policy_net.get_q_values(next_states)
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
        losses = -(proj_dist * log_p).sum(1)

        return losses

    def update(self) -> C51UpdateLoss:
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self._sample_transactions()

        losses = self._update_distributional_rl(
            states, actions, rewards, next_states, dones
        )

        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return C51UpdateLoss(loss=loss.item())

    def _update_target_net(self):
        if self.total_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_episode(self) -> SingleEpisodeResult:
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
        torch.save(self.policy_net.state_dict(), self.model_file)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_file))

    def eval_mode_on(self):
        self.policy_net.eval()

    def eval_mode_off(self):
        self.policy_net.train()
