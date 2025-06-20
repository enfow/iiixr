from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

from model.buffer import PPOMemory
from model.ppo import Actor, Critic
from trainer.base_trainer import BaseConfig, BaseTrainer


@dataclass
class PPOConfig(BaseConfig):
    ppo_epochs: int = 4
    clip_eps: float = 0.2


class PPOTrainer(BaseTrainer):
    def __init__(self, env, config, save_dir="results/ppo"):
        config = PPOConfig.from_dict(config, env)
        super().__init__(env, config, save_dir)

    def _init_models(self):
        self.actor = Actor(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.config.hidden_dim,
        ).to(self.config.device)
        self.critic = Critic(
            self.env.observation_space.shape[0], self.config.hidden_dim
        ).to(self.config.device)
        self.memory = PPOMemory()
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.lr,
        )

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item()

    def compute_returns(self, rewards, dones, gamma=0.99):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def update(self):
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.config.device)
        actions = torch.LongTensor(self.memory.actions).to(self.config.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.config.device)
        returns = torch.FloatTensor(
            self.compute_returns(
                self.memory.rewards, self.memory.dones, self.config.gamma
            )
        ).to(self.config.device)
        advantages = returns - self.critic(states).squeeze().detach()
        for _ in range(self.config.ppo_epochs):
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            logprobs = dist.log_prob(actions)
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1 - self.config.clip_eps,
                    1 + self.config.clip_eps,
                )
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.mse_loss(
                self.critic(states).squeeze(), returns
            )
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.episode_losses.append(loss.item())
        self.memory.clear()

    def train(self):
        for ep in range(self.config.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            self.episode_losses = []
            for t in range(self.config.max_steps):
                action, logprob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.memory.store(state, action, logprob, reward, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            self.update()
            avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0.0
            self.scores.append(total_reward)
            self.losses.append(avg_loss)
            if total_reward > self.best_score:
                self.best_score = total_reward
                self.save_model()
            self._log_metrics()

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            self.model_file,
        )

    def load_model(self):
        self.actor.load_state_dict(torch.load(self.model_file)["actor"])
        self.critic.load_state_dict(torch.load(self.model_file)["critic"])

    def ready_to_evaluate(self):
        self.load_model()
        self.actor.eval()
        self.critic.eval()

    def evaluate(self, episodes=10):
        self.ready_to_evaluate()

        scores = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            scores.append(total_reward)

        return scores
