from collections import deque

import numpy as np
import torch
import torch.optim as optim

from model.ppo import (LSTMContinuousActor, LSTMContinuousCritic,
                       TransformerContinuousActor, TransformerContinuousCritic)
from schema.config import ModelEmbeddingType, PPOConfig
from schema.result import PPOUpdateLoss, SingleEpisodeResult
from trainer.ppo_trainer import PPOTrainer


class PPOSequentialTrainer(PPOTrainer):
    name = "ppo_seq"
    config_class = PPOConfig

    def __init__(
        self, env_name: str, config: PPOConfig, save_dir: str = "results/ppo_seq"
    ):
        super().__init__(env_name, config, save_dir)
        self.seq_len = self.config.seq_len

    def _init_models(self):
        """Initializes models based on the specified embedding_type."""
        device = self.config.device
        model_config = self.config.model

        if model_config.embedding_type == ModelEmbeddingType.LSTM:
            self.actor = LSTMContinuousActor(
                self.state_dim,
                self.action_dim,
                model_config.hidden_dim,
                n_layers=model_config.n_layers,
                use_layernorm=model_config.use_layernorm,
            ).to(device)
            self.critic = LSTMContinuousCritic(
                self.state_dim,
                model_config.hidden_dim,
                n_layers=model_config.n_layers,
                use_layernorm=model_config.use_layernorm,
            ).to(device)
            self.actor_hidden = None
            self.critic_hidden = None
        elif model_config.embedding_type == ModelEmbeddingType.TRANSFORMER:
            self.actor = TransformerContinuousActor(
                self.state_dim,
                self.action_dim,
                model_config.hidden_dim,
                n_layers=model_config.n_layers,
            ).to(device)
            self.critic = TransformerContinuousCritic(
                self.state_dim,
                model_config.hidden_dim,
                n_layers=model_config.n_layers,
            ).to(device)
            self.state_history = deque(maxlen=self.config.seq_len)
        else:
            raise ValueError(f"Unknown embedding type: {model_config.embedding_type}")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr)

    def _reset_hidden_state(self, batch_size=1):
        """Resets LSTM hidden states to zeros."""
        device = self.config.device
        hidden_dim = self.config.model.hidden_dim
        num_layers = self.config.model.n_layers
        self.actor_hidden = (
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
        )
        self.critic_hidden = (
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
            torch.zeros(num_layers, batch_size, hidden_dim).to(device),
        )

    def _get_transformer_sequence(self):
        """Pads the state history to the required sequence length for the Transformer."""
        sequence = list(self.state_history)
        padding = [sequence[0]] * (self.seq_len - len(sequence))
        return np.array(padding + sequence)

    def select_action(self, state, eval_mode: bool = False):
        """Selects an action by dispatching to the appropriate model-specific method."""
        if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
            return self._select_lstm_action(state, eval_mode)
        else:
            return self._select_transformer_action(state, eval_mode)

    def _select_lstm_action(self, state, eval_mode):
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.config.device)
            )
            mean, log_std, next_hidden = self.actor(state_tensor, self.actor_hidden)
            self.actor_hidden = next_hidden
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = mean if eval_mode else dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)
            return {
                "action": action.squeeze().cpu().numpy(),
                "logprob": logprob.squeeze().item(),
            }

    def _select_transformer_action(self, state, eval_mode):
        self.state_history.append(state)
        state_sequence = self._get_transformer_sequence()
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state_sequence).unsqueeze(0).to(self.config.device)
            )
            mean, log_std = self.actor(state_tensor)
            mean, log_std = mean[:, -1, :], log_std[:, -1, :]
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = mean if eval_mode else dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)
            return {"action": action.squeeze().cpu().numpy(), "logprob": logprob.item()}

    def collect_episode_data(self):
        """Collects data for one full episode, resetting state/history at the start."""
        if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
            self._reset_hidden_state()
        else:
            self.state_history.clear()

        state, _ = self.env.reset()
        if self.config.model.embedding_type == ModelEmbeddingType.TRANSFORMER:
            self.state_history.append(state)

        episode_data = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "dones": [],
        }
        while True:
            action_info = self.select_action(state)
            action, logprob = action_info["action"], action_info["logprob"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_data["states"].append(state)
            episode_data["actions"].append(action)
            episode_data["logprobs"].append(logprob)
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)
            state = next_state
            if done:
                break
        episode_data["episode_length"] = len(episode_data["states"])
        return episode_data

    # --- Overridden Methods to Fix Inheritance Issues ---

    def train_episode(self) -> SingleEpisodeResult:
        """Main training loop for an episode. Overrides the base method."""
        all_episode_data = []
        total_rewards = 0
        episode_lengths = []
        total_steps = 0
        episode_count = 0

        while total_steps < self.config.n_transactions:
            episode_data = self.collect_episode_data()
            episode_count += 1
            episode_length = episode_data["episode_length"]
            total_steps += episode_length
            episode_lengths.append(episode_length)
            total_rewards += np.sum(episode_data["rewards"])

            if self.config.gae:
                processed_episode = self.compute_episode_gae(episode_data)
            else:
                processed_episode = self.compute_episode_mc_returns(episode_data)
            all_episode_data.append(processed_episode)

        update_result = self.update(all_episode_data)
        return SingleEpisodeResult(
            episode_total_reward=round(total_rewards / episode_count, 2),
            episode_steps=round(np.mean(episode_lengths), 2),
            episode_losses=[update_result],
        )

    def compute_episode_gae(self, episode_data):
        """Overrides the base method to correctly call sequential critics."""
        states = torch.FloatTensor(np.array(episode_data["states"])).to(
            self.config.device
        )
        rewards = torch.tensor(episode_data["rewards"], dtype=torch.float32).to(
            self.config.device
        )
        dones = torch.tensor(episode_data["dones"], dtype=torch.float32).to(
            self.config.device
        )

        with torch.no_grad():
            states_batch = states.unsqueeze(0)
            if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
                self._reset_hidden_state()
                values, _ = self.critic(states_batch, self.critic_hidden)
            else:
                values = self.critic(states_batch)
            values = values.squeeze(0).squeeze(-1)
            if values.dim() == 0:
                values = values.unsqueeze(0)

            next_values = torch.zeros_like(values)
            if len(values) > 1:
                next_values[:-1] = values[1:]

        advantages = torch.zeros_like(rewards)
        gae_advantage = 0
        for t in reversed(range(len(rewards))):
            td_error = (
                rewards[t]
                + self.config.gamma * next_values[t] * (1 - dones[t])
                - values[t]
            )
            gae_advantage = (
                td_error
                + self.config.gamma
                * self.config.gae_lambda
                * (1 - dones[t])
                * gae_advantage
            )
            advantages[t] = gae_advantage

        returns = advantages + values
        # Add episode_length back for the transformer update logic
        episode_data.update(
            {
                "states": states,
                "returns": returns,
                "advantages": advantages,
                "rewards": rewards,
            }
        )
        return episode_data

    def compute_episode_mc_returns(self, episode_data):
        """Overrides the base method for Monte Carlo returns."""
        states = torch.FloatTensor(np.array(episode_data["states"])).to(
            self.config.device
        )
        returns = self.compute_monte_carlo_returns(
            episode_data["rewards"], episode_data["dones"]
        )
        with torch.no_grad():
            states_batch = states.unsqueeze(0)
            if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
                self._reset_hidden_state()
                values, _ = self.critic(states_batch, self.critic_hidden)
            else:
                values = self.critic(states_batch)
            values = values.squeeze(0).squeeze(-1)
            if values.dim() == 0:
                values = values.unsqueeze(0)
        advantages = returns - values
        episode_data.update(
            {
                "states": states,
                "returns": returns,
                "advantages": advantages,
                "rewards": torch.tensor(
                    episode_data["rewards"], dtype=torch.float32
                ).to(self.config.device),
            }
        )
        return episode_data

    # --- Update Logic ---

    def update(self, all_episode_data) -> PPOUpdateLoss:
        """Dispatches to the correct update method based on model type."""
        if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
            return self._update_lstm(all_episode_data)
        else:
            return self._update_transformer(all_episode_data)

    def _update_lstm(self, all_episode_data):
        """Performs PPO updates for the LSTM model using full episodes as sequences."""
        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        ppo_epochs = self.config.ppo_epochs

        for episode_data in all_episode_data:
            states = episode_data["states"].unsqueeze(0)
            actions = (
                torch.FloatTensor(np.array(episode_data["actions"]))
                .unsqueeze(0)
                .to(self.config.device)
            )
            old_logprobs = torch.FloatTensor(episode_data["logprobs"]).to(
                self.config.device
            )
            returns = episode_data["returns"].to(self.config.device)
            advantages = episode_data["advantages"].to(self.config.device)
            if self.config.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            for _ in range(ppo_epochs):
                self._reset_hidden_state()
                mean, log_std, _ = self.actor(states, self.actor_hidden)
                dist = torch.distributions.Normal(mean, torch.exp(log_std))
                logprobs = dist.log_prob(actions).sum(dim=-1).squeeze(0)
                ratio = torch.exp(logprobs - old_logprobs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                    )
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                entropy_loss = -self.config.entropy_coef * entropy
                total_actor_loss_step = actor_loss + entropy_loss
                self.actor_optimizer.zero_grad()
                total_actor_loss_step.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()

                self._reset_hidden_state()
                values, _ = self.critic(states, self.critic_hidden)
                values = values.squeeze(0).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(values, returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()

        num_updates = ppo_epochs * len(all_episode_data)
        return PPOUpdateLoss(
            actor_loss=total_actor_loss / num_updates,
            critic_loss=total_critic_loss / num_updates,
            entropy_loss=total_entropy_loss / num_updates,
        )

    def _update_transformer(self, all_episode_data):
        """Performs PPO updates for the Transformer by sampling valid windows."""
        all_states, all_actions, all_logprobs, all_returns, all_advantages = (
            self.get_all_episode_data(all_episode_data)
        )
        if self.config.normalize_advantages:
            all_advantages = (all_advantages - all_advantages.mean()) / (
                all_advantages.std() + 1e-8
            )

        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        ppo_epochs = self.config.ppo_epochs
        batch_size = self.config.batch_size

        valid_indices = []
        start = 0
        for episode_data in all_episode_data:
            end = start + episode_data["episode_length"]
            if end - start >= self.seq_len:
                for i in range(start, end - self.seq_len + 1):
                    valid_indices.append(i)
            start = end

        if not valid_indices:
            return PPOUpdateLoss(actor_loss=0, critic_loss=0, entropy_loss=0)
        indices = np.array(valid_indices)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                state_batch = torch.stack(
                    [all_states[j : j + self.seq_len] for j in batch_indices]
                ).to(self.config.device)
                action_batch = torch.stack(
                    [all_actions[j : j + self.seq_len] for j in batch_indices]
                ).to(self.config.device)
                old_logprob_batch = torch.stack(
                    [all_logprobs[j : j + self.seq_len] for j in batch_indices]
                ).to(self.config.device)
                return_batch = torch.stack(
                    [all_returns[j : j + self.seq_len] for j in batch_indices]
                ).to(self.config.device)
                advantage_batch = torch.stack(
                    [all_advantages[j : j + self.seq_len] for j in batch_indices]
                ).to(self.config.device)

                mean, log_std = self.actor(state_batch)
                dist = torch.distributions.Normal(mean, torch.exp(log_std))
                logprobs = dist.log_prob(action_batch).sum(dim=-1)
                ratio = torch.exp(logprobs - old_logprob_batch)
                surr1 = ratio * advantage_batch
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                    )
                    * advantage_batch
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                entropy_loss = -self.config.entropy_coef * entropy
                total_actor_loss_step = actor_loss + entropy_loss
                self.actor_optimizer.zero_grad()
                total_actor_loss_step.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()

                values = self.critic(state_batch).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(values, return_batch)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()

        num_updates = ppo_epochs * (len(indices) // batch_size)
        if num_updates == 0:
            return PPOUpdateLoss(actor_loss=0, critic_loss=0, entropy_loss=0)
        return PPOUpdateLoss(
            actor_loss=total_actor_loss / num_updates,
            critic_loss=total_critic_loss / num_updates,
            entropy_loss=total_entropy_loss / num_updates,
        )
