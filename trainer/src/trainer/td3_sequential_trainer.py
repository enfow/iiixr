"""
TD3 Sequential Trainer with Transformer Actor

Reference
---------
- [TD3: Twin Delayed DDPG](<https://arxiv.org/pdf/1802.09477>)
"""

import copy
from collections import deque
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

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
        save_dir: str = "results/td3_seq",
    ):
        super().__init__(env_name, config, save_dir)
        self.total_it = 0
        self._reset_histories()

    def _init_models(self):
        self.max_action = float(self.env.action_space.high.flat[0])

        actor_class = None
        if self.config.model.embedding_type == ModelEmbeddingType.TRANSFORMER:
            actor_class = TransformerTD3Actor
        elif self.config.model.embedding_type == ModelEmbeddingType.LSTM:
            actor_class = LSTMTD3Actor
        else:
            raise ValueError(f"Invalid actor type: {self.config.model.embedding_type}")

        self.actor = actor_class(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            hidden_dim=self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
            use_layernorm=self.config.model.use_layernorm,
            **(
                {"nhead": N_TRANSFORMER_HEADS}
                if actor_class == TransformerTD3Actor
                else {}
            ),
        ).to(self.config.device)
        self.critic = TD3Critic(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
            use_layernorm=self.config.model.use_layernorm,
        ).to(self.config.device)
        self.actor_target, self.critic_target = (
            copy.deepcopy(self.actor),
            copy.deepcopy(self.critic),
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr)

        if isinstance(self.actor, LSTMTD3Actor):
            self.actor.lstm.flatten_parameters()
            self.actor_target.lstm.flatten_parameters()

    def _reset_histories(self):
        self.state_histories = [
            deque(maxlen=self.config.model.seq_len) for _ in range(self.config.n_envs)
        ]

    def _create_state_sequence(
        self, state: np.ndarray, env_idx: int, to_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        history = self.state_histories[env_idx]
        history.append(state)

        if len(history) < self.config.model.seq_len:
            padding = [history[0]] * (self.config.model.seq_len - len(history))
            state_sequence_np = np.array(padding + list(history))
        else:
            state_sequence_np = np.array(history)

        if to_tensor:
            return (
                torch.FloatTensor(state_sequence_np).unsqueeze(0).to(self.config.device)
            )
        return state_sequence_np

    def select_action(
        self, state: np.ndarray, env_idx: int = 0, eval_mode: bool = False
    ) -> Dict[str, Any]:
        with torch.no_grad():
            state_sequence = self._create_state_sequence(state, env_idx, to_tensor=True)
            action = self.actor(state_sequence).cpu().data.numpy().flatten()

        if not eval_mode:
            noise = np.random.normal(
                0, self.max_action * self.exploration_noise, size=self.action_dim
            )
            action = action + noise

        clipped_action = np.clip(action, -self.max_action, self.max_action)
        return {"action": clipped_action}

    def update(self) -> TD3UpdateLoss:
        if len(self.memory) < self.config.batch_size:
            return None
        self.total_it += 1

        if self.memory.is_per:
            (
                state_seqs,
                action_seqs,
                reward_seqs,
                next_state_seqs,
                done_seqs,
                idxs,
                weights,
            ) = self.memory.sample(self.config.batch_size)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.config.device)
        else:
            state_seqs, action_seqs, reward_seqs, next_state_seqs, done_seqs = (
                self.memory.sample(self.config.batch_size)
            )

        if state_seqs is None or len(state_seqs) == 0:
            return None

        state_seqs = torch.FloatTensor(state_seqs).to(self.config.device)
        next_state_seqs = torch.FloatTensor(next_state_seqs).to(self.config.device)
        action_seqs = torch.FloatTensor(action_seqs).to(self.config.device)
        reward = (
            torch.FloatTensor(reward_seqs[:, -1]).unsqueeze(1).to(self.config.device)
        )
        done = torch.BoolTensor(done_seqs[:, -1]).unsqueeze(1).to(self.config.device)

        with torch.no_grad():
            target_action = self.actor_target(next_state_seqs)
            noise = (torch.randn_like(target_action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (target_action + noise).clamp(
                -self.max_action, self.max_action
            )
            current_next_state = next_state_seqs[:, -1]
            target_q1, target_q2 = self.critic_target(current_next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (~done) * self.config.gamma * target_q
        current_state, current_action = state_seqs[:, -1], action_seqs[:, -1]
        current_q1, current_q2 = self.critic(current_state, current_action)

        if self.memory.is_per:
            td_errors = torch.abs(current_q1 - target_q)
            critic_loss = (
                weights
                * (
                    F.mse_loss(current_q1, target_q, reduction="none")
                    + F.mse_loss(current_q2, target_q, reduction="none")
                )
            ).mean()
            self.memory.update_priorities(
                idxs, td_errors.detach().cpu().numpy().flatten()
            )

        else:
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2, target_q
            )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = None
        if self.total_it % self.config.policy_delay == 0:
            actor_action = self.actor(state_seqs)
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
            actor_loss=actor_loss.item() if actor_loss else 0.0,
            critic_loss=critic_loss.item(),
        )

    def collect_initial_data(self, start_steps: int):
        if self.config.n_envs <= 1:
            state, _ = self.env.reset()
            for _ in range(start_steps):
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                if done:
                    state, _ = self.env.reset()
        else:
            states, _ = self.env.reset()
            with tqdm(total=start_steps, desc="Collecting initial data") as pbar:
                while len(self.memory) < start_steps:
                    actions = self.env.action_space.sample()
                    next_states, rewards, terminations, truncations, infos = (
                        self.env.step(actions)
                    )
                    final_observations = infos.get("final_observation")
                    for i in range(self.config.n_envs):
                        if len(self.memory) >= start_steps:
                            break
                        done = terminations[i] or truncations[i]
                        if done and final_observations is not None:
                            real_next_state = final_observations[i]
                        else:
                            real_next_state = next_states[i]
                        self.memory.push(
                            states[i], actions[i], rewards[i], real_next_state, done
                        )
                        pbar.update(1)
                    states = next_states
        print(f"Collected {len(self.memory)} initial data points.")

    def train_episode(self) -> SingleEpisodeResult:
        self.exploration_noise = self._get_decayed_exploration_noise(
            self.train_episode_number
        )
        if self.config.n_envs > 1:
            return self._train_vectorized()
        else:
            return self._train_single_episode()

    def _train_single_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        self.state_histories[0].clear()
        done = False
        episode_rewards, episode_losses, step = [], [], 0
        while not done and step < self.config.max_steps:
            action_info = self.select_action(state, env_idx=0)
            action = action_info["action"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_rewards.append(reward)
            step += 1
            if len(self.memory) >= self.config.batch_size:
                update_result = self.update()
                if update_result is not None:
                    episode_losses.append(update_result)
        return SingleEpisodeResult(
            episode_total_reward=np.sum(episode_rewards),
            episode_steps=step,
            episode_losses=episode_losses,
        )

    def _train_vectorized(self) -> SingleEpisodeResult:
        states, _ = self.env.reset()
        self._reset_histories()
        total_rewards = np.zeros(self.config.n_envs)
        episode_losses = []

        for step in range(self.config.max_steps):
            state_sequences_np = [
                self._create_state_sequence(state, i, to_tensor=False)
                for i, state in enumerate(states)
            ]
            state_sequences_tensor = torch.FloatTensor(np.array(state_sequences_np)).to(
                self.config.device
            )
            with torch.no_grad():
                actions_tensor = self.actor(state_sequences_tensor)
            actions = actions_tensor.cpu().numpy()
            actions += np.random.normal(
                0, self.max_action * self.exploration_noise, size=actions.shape
            )
            actions = np.clip(actions, -self.max_action, self.max_action)

            next_states, rewards, terminations, truncations, infos = self.env.step(
                actions
            )
            total_rewards += rewards
            final_observations = infos.get("final_observation")

            for i in range(self.config.n_envs):
                done = terminations[i] or truncations[i]
                if done:
                    real_next_state = (
                        final_observations[i]
                        if final_observations is not None
                        else next_states[i]
                    )
                    self.state_histories[i].clear()
                else:
                    real_next_state = next_states[i]
                self.memory.push(
                    states[i], actions[i], rewards[i], real_next_state, done
                )
            states = next_states

            if len(self.memory) >= self.config.batch_size:
                update_result = self.update()
                if update_result is not None:
                    episode_losses.append(update_result)

        return SingleEpisodeResult(
            episode_total_reward=np.sum(total_rewards),
            episode_steps=step + 1,
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
        checkpoint = torch.load(self.model_file, map_location=self.config.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

        if isinstance(self.actor, LSTMTD3Actor):
            self.actor.lstm.flatten_parameters()
            self.actor_target.lstm.flatten_parameters()

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
