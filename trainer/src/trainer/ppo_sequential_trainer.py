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
        self._init_models()

    def _init_models(self):
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
                self.state_dim, model_config.hidden_dim, n_layers=model_config.n_layers
            ).to(device)

        if model_config.embedding_type in [
            ModelEmbeddingType.LSTM,
            ModelEmbeddingType.TRANSFORMER,
        ]:
            if self.n_envs > 1:
                self.state_history = [
                    deque(maxlen=self.seq_len) for _ in range(self.n_envs)
                ]
            else:
                self.state_history = deque(maxlen=self.seq_len)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr)

    def _reset_hidden_state(self, batch_size=None, dones=None):
        if self.config.model.embedding_type != ModelEmbeddingType.LSTM:
            return

        if dones is not None:
            done_indices = np.where(dones)[0]
            for i in done_indices:
                self.actor_hidden[0][:, i, :].zero_()
                self.actor_hidden[1][:, i, :].zero_()
                self.critic_hidden[0][:, i, :].zero_()
                self.critic_hidden[1][:, i, :].zero_()
        else:
            b_size = batch_size if batch_size is not None else self.n_envs
            device = self.config.device
            hidden_dim = self.config.model.hidden_dim
            num_layers = self.config.model.n_layers
            self.actor_hidden = (
                torch.zeros(num_layers, b_size, hidden_dim).to(device),
                torch.zeros(num_layers, b_size, hidden_dim).to(device),
            )
            self.critic_hidden = (
                torch.zeros(num_layers, b_size, hidden_dim).to(device),
                torch.zeros(num_layers, b_size, hidden_dim).to(device),
            )

    # def select_action(self, state, eval_mode: bool = False):
    #     with torch.no_grad():
    #         is_parallel = state.ndim > 1
    #         state_tensor = torch.FloatTensor(state).to(self.config.device)

    #         if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
    #             state_tensor = (
    #                 state_tensor.unsqueeze(0) if not is_parallel else state_tensor
    #             )
    #             state_tensor = state_tensor.unsqueeze(1)
    #             mean, log_std, next_hidden = self.actor(state_tensor, self.actor_hidden)
    #             self.actor_hidden = next_hidden
    #             output = mean.squeeze(1), log_std.squeeze(1)
    #         elif self.config.model.embedding_type == ModelEmbeddingType.TRANSFORMER:
    #             # This part is complex and requires careful sequence management
    #             # The logic here is simplified for a single step action selection
    #             if is_parallel:
    #                 for i in range(self.n_envs):
    #                     self.state_history[i].append(state[i])
    #                 sequences = [list(hist) for hist in self.state_history]
    #                 state_batch = torch.FloatTensor(np.array(sequences)).to(
    #                     self.config.device
    #                 )
    #             else:
    #                 self.state_history.append(state)
    #                 sequence = list(self.state_history)
    #                 padding = (
    #                     [sequence[0]] * (self.seq_len - len(sequence))
    #                     if sequence
    #                     else [np.zeros(self.state_dim)] * self.seq_len
    #                 )
    #                 state_batch = torch.FloatTensor(np.array([padding + sequence])).to(
    #                     self.config.device
    #                 )

    #             mean, log_std = self.actor(state_batch)
    #             output = mean[:, -1, :], log_std[:, -1, :]
    #         else:  # FC
    #             mean, log_std = self.actor(state_tensor)
    #             output = mean, log_std

    #         mean, log_std = output
    #         std = torch.exp(log_std)
    #         dist = torch.distributions.Normal(mean, std)
    #         action = mean if eval_mode else dist.sample()
    #         logprob = dist.log_prob(action).sum(dim=-1)

    #         result = {"action": action.cpu().numpy(), "logprob": logprob.cpu().numpy()}
    #         if not is_parallel:
    #             result["action"] = result["action"].squeeze(0)
    #             result["logprob"] = result["logprob"].item()
    #         return result
        
    def select_action(self, state, eval_mode: bool = False):
        with torch.no_grad():
            # Evaluation mode always uses a single environment, so is_parallel is effectively false.
            is_parallel = not eval_mode and state.ndim > 1
            state_tensor = torch.FloatTensor(state).to(self.config.device)
            
            if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
                # Add batch and sequence dimensions
                state_tensor = state_tensor.unsqueeze(0) if not is_parallel else state_tensor
                state_tensor = state_tensor.unsqueeze(1)
                mean, log_std, next_hidden = self.actor(state_tensor, self.actor_hidden)
                self.actor_hidden = next_hidden
                output = mean.squeeze(1), log_std.squeeze(1)

            elif self.config.model.embedding_type == ModelEmbeddingType.TRANSFORMER:
                if eval_mode:
                    self.eval_state_history.append(state)
                    sequence = list(self.eval_state_history)
                    padding = [sequence[0]] * (self.seq_len - len(sequence)) if sequence else [np.zeros(self.state_dim)] * self.seq_len
                    state_batch = torch.FloatTensor(np.array([padding + sequence])).to(self.config.device)
                elif is_parallel:
                    for i in range(self.n_envs): self.state_history[i].append(state[i])
                    sequences = [list(hist) for hist in self.state_history]
                    padded_sequences = []
                    for seq in sequences:
                        if len(seq) < self.seq_len:
                            padding = [seq[0] if seq else np.zeros(self.state_dim)] * (self.seq_len - len(seq))
                            padded_sequences.append(padding + seq)
                        else:
                            padded_sequences.append(seq)
                    state_batch = torch.FloatTensor(np.array(padded_sequences)).to(self.config.device)
                else: # Sequential training
                    self.state_history.append(state)
                    sequence = list(self.state_history)
                    padding = [sequence[0]] * (self.seq_len - len(sequence)) if sequence else [np.zeros(self.state_dim)] * self.seq_len
                    state_batch = torch.FloatTensor(np.array([padding + sequence])).to(self.config.device)
                
                mean, log_std = self.actor(state_batch)
                output = mean[:, -1, :], log_std[:, -1, :]
            else: # FC
                # Assuming a simple FC model for non-recurrent case
                # mean, log_std = self.actor(state_tensor)
                # output = mean, log_std
                raise NotImplementedError("FC model not fully implemented in this example.")

            mean, log_std = output
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = mean if eval_mode else dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)
            
            result = {"action": action.cpu().numpy(), "logprob": logprob.cpu().numpy()}
            if not is_parallel:
                result["action"] = result["action"].squeeze(0)
                result["logprob"] = result["logprob"].item()
            return result


    def collect_episode_data(self):
        # (Sequential Mode)
        state, _ = self.env.reset()
        self._reset_hidden_state(batch_size=1)
        episode_data = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "dones": [],
            "episode_length": 0,
        }

        while True:
            action_info = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(
                action_info["action"]
            )
            done = terminated or truncated
            for k, v in {
                "states": state,
                "actions": action_info["action"],
                "logprobs": action_info["logprob"],
                "rewards": reward,
                "dones": done,
            }.items():
                episode_data[k].append(v)
            state = next_state
            if done:
                break
        episode_data["episode_length"] = len(episode_data["rewards"])
        return episode_data

    def collect_trajectories(self):
        # (Parallel Mode)
        num_steps = self.config.n_transactions // self.n_envs
        data = {
            "states": np.zeros(
                (num_steps, self.n_envs, self.state_dim), dtype=np.float32
            ),
            "actions": np.zeros(
                (num_steps, self.n_envs, self.action_dim), dtype=np.float32
            ),
            "logprobs": np.zeros((num_steps, self.n_envs), dtype=np.float32),
            "rewards": np.zeros((num_steps, self.n_envs), dtype=np.float32),
            "dones": np.zeros((num_steps, self.n_envs), dtype=np.float32),
        }

        state, _ = self.env.reset()
        self._reset_hidden_state(batch_size=self.n_envs)

        for step in range(num_steps):
            action_info = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(
                action_info["action"]
            )
            done = terminated | truncated
            (
                data["states"][step],
                data["actions"][step],
                data["logprobs"][step],
                data["rewards"][step],
                data["dones"][step],
            ) = state, action_info["action"], action_info["logprob"], reward, done
            state = next_state
            self._reset_hidden_state(dones=done)

        data["last_state"] = state
        return data

    def train_episode(self) -> SingleEpisodeResult:
        if self.n_envs > 1:
            trajectories = self.collect_trajectories()
            # This part is complex and requires careful GAE implementation for recurrent models
            # For simplicity, we create a compatible dict and use the sequential GAE logic
            all_episode_data = []
            for i in range(self.n_envs):
                episode_data = {
                    "states": trajectories["states"][:, i, :],
                    "actions": trajectories["actions"][:, i, :],
                    "logprobs": trajectories["logprobs"][:, i],
                    "rewards": trajectories["rewards"][:, i],
                    "dones": trajectories["dones"][:, i],
                    "episode_length": trajectories["states"].shape[0],
                }
                processed_episode = self.compute_episode_gae(episode_data)
                all_episode_data.append(processed_episode)

            update_result = self.update(all_episode_data)
            total_rewards = np.sum(trajectories["rewards"])
            return SingleEpisodeResult(
                episode_total_reward=round(total_rewards / self.n_envs, 2),
                episode_steps=round(trajectories["states"].shape[0], 2),
                episode_losses=[update_result],
            )
        else:
            (
                all_episode_data,
                total_rewards,
                episode_lengths,
                total_steps,
                episode_count,
            ) = [], 0, [], 0, 0
            while total_steps < self.config.n_transactions:
                episode_data = self.collect_episode_data()
                episode_count += 1
                total_steps += episode_data["episode_length"]
                episode_lengths.append(episode_data["episode_length"])
                total_rewards += np.sum(episode_data["rewards"])
                processed_episode = self.compute_episode_gae(episode_data)
                all_episode_data.append(processed_episode)

            update_result = self.update(all_episode_data)
            return SingleEpisodeResult(
                episode_total_reward=round(total_rewards / episode_count, 2),
                episode_steps=round(np.mean(episode_lengths), 2),
                episode_losses=[update_result],
            )

    def compute_episode_gae(self, episode_data):
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
                self._reset_hidden_state(batch_size=1)
                values, _ = self.critic(states_batch, self.critic_hidden)
                values = values.squeeze(0).squeeze(-1)
            else:  # Transformer
                values = self.critic(states_batch).squeeze(0).squeeze(-1)

        advantages = torch.zeros_like(rewards)
        gae_advantage = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            td_error = (
                rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            )
            gae_advantage = (
                td_error
                + self.config.gamma
                * self.config.gae_lambda
                * (1 - dones[t])
                * gae_advantage
            )
            advantages[t] = gae_advantage

        episode_data.update(
            {
                "states": states,
                "returns": advantages + values,
                "advantages": advantages,
                "rewards": rewards,
                "logprobs": episode_data["logprobs"],
                "actions": episode_data["actions"],
            }
        )
        return episode_data

    def update(self, all_episode_data):
        if self.config.model.embedding_type == ModelEmbeddingType.LSTM:
            if self.seq_stride > 1:
                return self._update_lstm_strided(all_episode_data)
            else:
                return self._update_lstm(all_episode_data)
        else:  # Transformer
            return self._update_transformer(all_episode_data)

    def _update_lstm(self, all_episode_data):
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
                self._reset_hidden_state(batch_size=states.shape[0])
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

                self._reset_hidden_state(batch_size=states.shape[0])
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
                for i in range(start, end - self.seq_len + 1, self.seq_stride):
                    valid_indices.append(i)
            start = end

        if not valid_indices:
            return PPOUpdateLoss(0, 0, 0)
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
            return PPOUpdateLoss(0, 0, 0)
        return PPOUpdateLoss(
            actor_loss=total_actor_loss / num_updates,
            critic_loss=total_critic_loss / num_updates,
            entropy_loss=total_entropy_loss / num_updates,
        )

    def _update_lstm_strided(self, all_episode_data):
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
                for i in range(start, end - self.seq_len + 1, self.seq_stride):
                    valid_indices.append(i)
            start = end

        if not valid_indices:
            return PPOUpdateLoss(0, 0, 0)
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

                self._reset_hidden_state(batch_size=state_batch.size(0))
                mean, log_std, _ = self.actor(state_batch, self.actor_hidden)
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

                self._reset_hidden_state(batch_size=state_batch.size(0))
                values, _ = self.critic(state_batch, self.critic_hidden)
                values = values.squeeze(-1)
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
            return PPOUpdateLoss(0, 0, 0)
        return PPOUpdateLoss(
            actor_loss=total_actor_loss / num_updates,
            critic_loss=total_critic_loss / num_updates,
            entropy_loss=total_entropy_loss / num_updates,
        )

    def eval_mode_on(self):
        self.actor.eval()
        self.critic.eval()
        self._reset_hidden_state(batch_size=1)

    def eval_mode_off(self):
        self.actor.train()
        self.critic.train()
        self._reset_hidden_state(batch_size=self.n_envs)