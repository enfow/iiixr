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
        # Automatically set v_min and v_max for known environments
        if env_name in V_MIN_MAX:
            if "v_min" not in config_dict or "v_max" not in config_dict:
                config_dict["v_min"], config_dict["v_max"] = V_MIN_MAX[env_name]
                print(
                    f"V_MIN, V_MAX Automatically set: {config_dict['v_min']}, {config_dict['v_max']}"
                )

        super().__init__(env_name, config_dict, save_dir)
        self.total_steps = 0
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
        # Epsilon decay
        self.epsilon = self.config.eps_end + (
            self.config.eps_start - self.config.eps_end
        ) * math.exp(-1.0 * self.total_steps / self.config.eps_decay)

        if not eval_mode and np.random.rand() < self.epsilon:
            # Exploration
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

    def update(self):
        if len(self.memory) < self.config.batch_size:
            return None

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )

        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)

        with torch.no_grad():
            # Get Q-values for the next state from the target network to select actions
            next_q_values_target = self.target_net.get_q_values(next_states)
            next_actions = next_q_values_target.argmax(1)

            # Get the full distribution for the next state from the target network
            next_dist_target = self.target_net(next_states)
            # Select the distribution for the best actions
            next_dist = next_dist_target[range(self.config.batch_size), next_actions]

            # --- Project the target distribution onto the support ---
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

            # Fix for the case where l and u are equal
            l[(u > 0) * (l == u)] -= 1
            u[(l < self.config.n_atoms - 1) * (l == u)] += 1

            # Distribute probability
            proj_dist = torch.zeros_like(next_dist)
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

            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        # --- Calculate Loss ---
        # Get the distribution for the actions that were actually taken
        dist = self.policy_net(states)
        log_p = torch.log(dist[range(self.config.batch_size), actions])

        # Cross-entropy loss
        loss = -(proj_dist * log_p).sum(1).mean()

        # --- Optimize the model ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

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
                episode_losses.append(C51UpdateLoss(loss=update_result["loss"]))

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
