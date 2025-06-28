import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model.buffer import PPOMemory
from model.ppo import DiscreteActor, DiscreteCritic
from schema.config import PPOConfig
from trainer.ppo_trainer import PPOTrainer


class DiscretePPOTrainer(PPOTrainer):
    name = "discrete_ppo"
    config_class = PPOConfig

    def __init__(self, env_name: str, config: PPOConfig, save_dir: str = "results/ppo"):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        self.actor = DiscreteActor(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.critic = DiscreteCritic(
            self.state_dim,
            self.config.model.hidden_dim,
            n_layers=self.config.model.n_layers,
        ).to(self.config.device)
        self.memory = PPOMemory()
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()),
            lr=self.config.lr,
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()),
            lr=self.config.lr,
        )

    def select_action(self, state, eval_mode: bool = False):
        state = torch.FloatTensor(state).to(self.config.device)

        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)

        if eval_mode:
            # During evaluation, use greedy action (argmax)
            action = probs.argmax(dim=-1)
            logprob = dist.log_prob(action)
        else:
            # During training, sample from the policy
            action = dist.sample()
            logprob = dist.log_prob(action)

        return {
            "action": action.item(),
            "logprob": logprob.item(),
        }

    def _get_current_logprobs(self, states, actions):
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        logprobs = dist.log_prob(actions)
        return logprobs

    def _sample_transactions(self):
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.config.device)
        actions = torch.LongTensor(self.memory.actions).to(self.config.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.config.device)
        returns = torch.FloatTensor(
            self.compute_returns(
                self.memory.rewards, self.memory.dones, self.config.gamma
            )
        ).to(self.config.device)
        advantages = returns - self.critic(states).squeeze().detach()
        return states, actions, old_logprobs, returns, advantages

    def _get_entropy(self, states):
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
