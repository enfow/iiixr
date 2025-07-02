"""
TD3 Sequential Trainer with Transformer Actor

Reference
---------
- [TD3: Twin Delayed DDPG](<https://arxiv.org/pdf/1802.09477>)
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.td3 import LSTMTD3Actor, TD3Critic, TransformerTD3Actor
from schema.config import ModelEmbeddingType, TD3Config
from schema.result import SingleEpisodeResult, TD3UpdateLoss
from trainer.base_trainer import BaseTrainer

N_TRANSFORMER_HEADS = 8


class TD3SequentialTrainer(BaseTrainer):
    name = "td3_seq"
    config_class = TD3Config

    def __init__(
        self,
        env_name: str,
        config: TD3Config,
        save_dir: str = "results/td3_sequential",
    ):
        super().__init__(env_name, config, save_dir)
        self.total_it = 0

    def _init_models(self):
        self.max_action = float(self.env.action_space.high[0])

        if self.config.model.embedding_type == ModelEmbeddingType.TRANSFORMER:
            self.actor = TransformerTD3Actor(
                self.state_dim,
                self.action_dim,
                self.max_action,
                hidden_dim=self.config.model.hidden_dim,
                nhead=N_TRANSFORMER_HEADS,
                n_layers=self.config.model.n_layers,
                use_layernorm=self.config.model.use_layernorm,
            ).to(self.config.device)
        elif self.config.model.embedding_type == ModelEmbeddingType.LSTM:
            self.actor = LSTMTD3Actor(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                hidden_dim=self.config.model.hidden_dim,
                n_layers=self.config.model.n_layers,
                use_layernorm=self.config.model.use_layernorm,
            ).to(self.config.device)
        else:
            raise ValueError(f"Invalid actor type: {self.config.model.actor_type}")

        self.critic = TD3Critic(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
            use_layernorm=self.config.model.use_layernorm,
        ).to(self.config.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr)

    def reset_episode(self):
        """Reset state history at the beginning of each episode"""
        self.state_history.clear()

    def _create_state_sequence(self, state):
        """Create a sequence from current state and history"""
        self.state_history.append(state)

        if len(self.state_history) < self.config.buffer.seq_len:
            padding_needed = self.config.buffer.seq_len - len(self.state_history)
            padded_states = [state] * padding_needed + list(self.state_history)
        else:
            padded_states = list(self.state_history)

        padded_states = np.array(padded_states)
        state_sequence = (
            torch.FloatTensor(padded_states).unsqueeze(0).to(self.config.device)
        )
        return state_sequence

    def select_action(self, state, eval_mode: bool = False):
        with torch.no_grad():
            state_sequence = self._create_state_sequence(state)

            action = self.actor(state_sequence).cpu().data.numpy().flatten()

            # Exploration
            if not eval_mode:
                action = action + np.random.normal(
                    0,
                    self.max_action * self.config.exploration_noise,
                    size=self.action_dim,
                )

            action = np.clip(action, -self.max_action, self.max_action)
            return {"action": action}

    def update(self) -> TD3UpdateLoss:
        if len(self.memory) < self.config.batch_size:
            return None

        self.total_it += 1

        # Sample sequences from replay buffer
        sequences = self.memory.sample(self.config.batch_size)
        if sequences is None:
            return None

        state_seqs, action_seqs, reward_seqs, next_state_seqs, done_seqs = sequences

        state_seqs = torch.FloatTensor(state_seqs).to(
            self.config.device
        )  # (batch, seq_len, state_dim)
        next_state_seqs = torch.FloatTensor(next_state_seqs).to(
            self.config.device
        )  # (batch, seq_len, state_dim)
        action_seqs = torch.FloatTensor(action_seqs).to(
            self.config.device
        )  # (batch, seq_len, action_dim)

        reward = (
            torch.FloatTensor(reward_seqs[:, -1]).unsqueeze(1).to(self.config.device)
        )  # (batch, 1)
        done = (
            torch.BoolTensor(done_seqs[:, -1]).unsqueeze(1).to(self.config.device)
        )  # (batch, 1)

        with torch.no_grad():
            target_action = self.actor_target(next_state_seqs)  # (batch, action_dim)

            noise = (torch.randn_like(target_action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (target_action + noise).clamp(
                -self.max_action, self.max_action
            )

            current_next_state = next_state_seqs[:, -1]  # (batch, state_dim)
            target_q1, target_q2 = self.critic_target(current_next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (~done) * self.config.gamma * target_q

        current_state = state_seqs[:, -1]  # (batch, state_dim) - last timestep
        current_action = action_seqs[:, -1]  # (batch, action_dim) - last timestep
        current_q1, current_q2 = self.critic(current_state, current_action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % self.config.policy_delay == 0:
            actor_action = self.actor(state_seqs)  # (batch, action_dim)
            actor_loss = -self.critic.get_q1_value(current_state, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data
                    + (1 - self.config.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data
                    + (1 - self.config.tau) * target_param.data
                )

        return TD3UpdateLoss(
            actor_loss=actor_loss.item() if actor_loss is not None else 0.0,
            critic_loss=critic_loss.item(),
        )

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        self.reset_episode()

        done = False
        episode_rewards, episode_losses, episode_steps = [], [], []
        step = 0

        while not done and step < self.config.max_steps:
            action_info = self.select_action(state)
            action = action_info["action"]

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.memory.push(state, action, reward, next_state, done)

            if done:
                self.memory.episode_starts.append(len(self.memory) - 1)

            state = next_state
            episode_rewards.append(reward)
            step += 1

            # Update networks
            if len(self.memory) >= self.config.batch_size:
                update_result = self.update()
                if update_result is not None:
                    episode_losses.append(update_result)

        return SingleEpisodeResult(
            episode_total_reward=np.sum(episode_rewards),
            episode_steps=step,
            episode_losses=episode_losses,
        )

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            self.model_file,
        )

    def load_model(self):
        checkpoint = torch.load(self.model_file)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

    def eval_mode_on(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def eval_mode_off(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
