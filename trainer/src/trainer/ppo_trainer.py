import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model.buffer import PPOMemory
from model.ppo import Actor, Critic
from schema.config import PPOConfig
from schema.result import SingleEpisodeResult
from trainer.base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    def __init__(self, env: gym.Env, config: PPOConfig, save_dir: str = "results/ppo"):
        config = PPOConfig.from_dict(config)
        super().__init__(env, config, save_dir)

    def _init_models(self):
        self.actor = Actor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            is_discrete=self.is_discrete,
        ).to(self.config.device)
        self.critic = Critic(self.state_dim, self.config.hidden_dim).to(
            self.config.device
        )
        self.memory = PPOMemory()
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.lr,
        )

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)

        if self.is_discrete:
            # Discrete action space (e.g., LunarLander)
            probs = self.actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            return {
                "action": action.item(),
                "logprob": logprob.item(),
            }
        else:
            # Continuous action space (e.g., BipedalWalker)
            mean, log_std = self.actor(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
            # Clamp actions to valid range (typically [-1, 1] for most continuous envs)
            action = torch.tanh(action)
            return {
                "action": action.detach().cpu().numpy(),
                "logprob": logprob.item(),
            }

    def compute_returns(self, rewards, dones, gamma=0.99):
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def _get_actions_from_memory(self):
        if self.is_discrete:
            actions = torch.LongTensor(self.memory.actions).to(self.config.device)
        else:
            actions = torch.FloatTensor(np.array(self.memory.actions)).to(
                self.config.device
            )
        return actions

    def _get_current_logprobs(self, states, actions):
        if self.is_discrete:
            # Discrete action space
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            logprobs = dist.log_prob(actions)
        else:
            # Continuous action space
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            # Apply tanh to match action selection
            raw_actions = torch.atanh(torch.clamp(actions, -0.999, 0.999))
            logprobs = dist.log_prob(raw_actions).sum(dim=-1)
            # Adjust log probability for tanh transformation
            logprobs -= torch.log(1 - actions.pow(2) + 1e-8).sum(dim=-1)
        return logprobs

    def update(self):
        total_loss = 0
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.config.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.config.device)
        returns = torch.FloatTensor(
            self.compute_returns(
                self.memory.rewards, self.memory.dones, self.config.gamma
            )
        ).to(self.config.device)
        advantages = returns - self.critic(states).squeeze().detach()

        actions = self._get_actions_from_memory()

        for _ in range(self.config.ppo_epochs):
            logprobs = self._get_current_logprobs(states, actions)

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
            total_loss += loss.item()
        self.memory.clear()
        return total_loss

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards, episode_losses, episode_steps = [], [], [0]

        while sum(episode_steps) < self.config.n_transactions:
            action_info = self.select_action(state)
            action = action_info["action"]
            logprob = action_info["logprob"]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.store(state, action, logprob, reward, done)
            state = next_state
            episode_rewards.append(reward)
            episode_steps[-1] += 1

            if done:
                state, _ = self.env.reset()
                episode_steps.append(0)

        episode_steps = round(np.mean(episode_steps))

        loss = self.update()

        if loss is not None:
            episode_losses.append(loss)

        return SingleEpisodeResult(
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            episode_losses=episode_losses,
        )

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

    def eval_mode_on(self):
        self.actor.eval()
        self.critic.eval()

    def eval_mode_off(self):
        self.actor.train()
        self.critic.train()
