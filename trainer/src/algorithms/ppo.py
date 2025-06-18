import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()

        # Actor (Policy) Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic (Value) Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class PPOMemory:
    def __init__(self, batch_size: int):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store(
        self,
        state: np.ndarray,
        action: int,
        probs: float,
        val: float,
        reward: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(
        self,
    ) -> Tuple[
        List[np.ndarray], List[int], List[float], List[float], List[float], List[bool]
    ]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def __len__(self):
        return len(self.states)

class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 64,
        n_epochs: int = 10,
        hidden_dim: int = 64,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.memory = PPOMemory(batch_size)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

    def choose_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state = torch.FloatTensor(state)
        action, log_prob, value = self.actor_critic.get_action(state)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        probs: float,
        val: float,
        reward: float,
        done: bool,
    ):
        self.memory.store(state, action, probs, val, reward, done)

    def learn(self):
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - dones_arr[k])
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage)
            values = torch.tensor(values)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                old_probs = torch.tensor(old_prob_arr[batch])
                actions = torch.tensor(action_arr[batch])

                for _ in range(10):  # Multiple epochs of training
                    action_logits, value = self.actor_critic(states)
                    dist = torch.distributions.Categorical(
                        F.softmax(action_logits, dim=-1)
                    )
                    new_probs = dist.log_prob(actions)
                    prob_ratio = torch.exp(new_probs - old_probs)

                    weighted_probs = prob_ratio * advantage[batch]
                    weighted_clipped_probs = (
                        torch.clamp(
                            prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                        )
                        * advantage[batch]
                    )
                    actor_loss = -torch.min(
                        weighted_probs, weighted_clipped_probs
                    ).mean()

                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns - value.squeeze()).pow(2).mean()

                    total_loss = actor_loss + 0.5 * critic_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

        self.memory.clear()
