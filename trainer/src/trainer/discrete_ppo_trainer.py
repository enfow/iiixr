"""
Discrete PPO Trainer
Reference
---------
- [Proximal Policy Optimization Algorithms](<https://arxiv.org/pdf/1707.06347>)
"""

import numpy as np
import torch
import torch.optim as optim

from model.ppo import DiscreteActor, DiscreteCritic
from schema.config import PPOConfig
from trainer.ppo_trainer import PPOTrainer, PPOUpdateLoss


class DiscretePPOTrainer(PPOTrainer):
    name = "discrete_ppo"
    config_class = PPOConfig

    def __init__(self, env_name: str, config: PPOConfig, save_dir: str = "results/ppo"):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        """Override to use discrete actor/critic"""
        self.actor = DiscreteActor(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
            use_layernorm=self.config.model.use_layernorm,
        ).to(self.config.device)
        self.critic = DiscreteCritic(
            self.state_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
            use_layernorm=self.config.model.use_layernorm,
        ).to(self.config.device)
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=self.config.lr,
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()),
            lr=self.config.lr,
        )

    def select_action(self, state, eval_mode: bool = False):
        """Override for discrete action selection"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.config.device)
            probs = self.actor(state)
            dist = torch.distributions.Categorical(probs)

            if eval_mode:
                action = probs.argmax(dim=-1)
                logprob = dist.log_prob(action)
            else:
                action = dist.sample()
                logprob = dist.log_prob(action)

            return {
                "action": action.cpu().numpy(),
                "logprob": logprob.item(),
            }

    def _get_current_logprobs(self, states, actions):
        """Override for discrete action log probabilities"""
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        logprobs = dist.log_prob(actions)
        return logprobs

    def _get_entropy(self, states):
        """Override for discrete action entropy"""
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def get_all_episode_data(self, all_episode_data):
        all_states = torch.cat([ep["states"] for ep in all_episode_data], dim=0)
        all_actions = torch.LongTensor(
            np.concatenate([ep["actions"] for ep in all_episode_data])
        ).to(self.config.device)
        all_logprobs = torch.FloatTensor(
            np.concatenate([ep["logprobs"] for ep in all_episode_data])
        ).to(self.config.device)
        all_returns = torch.cat([ep["returns"] for ep in all_episode_data], dim=0)
        all_advantages = torch.cat([ep["advantages"] for ep in all_episode_data], dim=0)

        return all_states, all_actions, all_logprobs, all_returns, all_advantages
