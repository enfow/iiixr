import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model.buffer import PPOMemory
from model.ppo import Actor, Critic
from schema.config import PPOConfig
from schema.result import PPOUpdateLoss, SingleEpisodeResult
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
            n_layers=self.config.n_layers,
            is_discrete=self.is_discrete,
        ).to(self.config.device)
        self.critic = Critic(
            self.state_dim,
            self.config.hidden_dim,
            n_layers=self.config.n_layers,
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

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)

        if self.is_discrete:
            probs = self.actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            return {
                "action": action.item(),
                "logprob": logprob.item(),
            }
        else:
            mean, log_std = self.actor(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)
            action = torch.tanh(action)
            return {
                "action": action.cpu().detach().numpy(),
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
            raw_actions = torch.atanh(torch.clamp(actions, -0.999, 0.999))
            logprobs = dist.log_prob(raw_actions).sum(dim=-1)
            logprobs -= torch.log(1 - actions.pow(2) + 1e-8).sum(dim=-1)
        return logprobs

    def _sample_transactions(self):
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.config.device)
        if self.is_discrete:
            actions = torch.LongTensor(self.memory.actions).to(self.config.device)
        else:
            actions = torch.FloatTensor(np.array(self.memory.actions)).to(
                self.config.device
            )
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.config.device)
        returns = torch.FloatTensor(
            self.compute_returns(
                self.memory.rewards, self.memory.dones, self.config.gamma
            )
        ).to(self.config.device)
        advantages = returns - self.critic(states).squeeze().detach()
        return states, actions, old_logprobs, returns, advantages

    def _get_entropy(self, states):
        if self.is_discrete:
            # Discrete action space: use softmax entropy
            logits = self.actor(states)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
        else:
            # Continuous action space: use normal distribution entropy
            mean, log_std = self.actor(states)
            std = log_std.exp()
            # Entropy of normal distribution: 0.5 * log(2πe) + log(std)
            # Use torch.tensor for constants to ensure tensor operations
            pi = torch.tensor(torch.pi, device=states.device, dtype=states.dtype)
            e = torch.tensor(torch.e, device=states.device, dtype=states.dtype)
            entropy = 0.5 * (torch.log(2 * pi * e) + 2 * log_std).sum(dim=-1)
        return entropy

    def update(self) -> PPOUpdateLoss:
        total_loss = 0
        states, actions, old_logprobs, returns, advantages = self._sample_transactions()

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.ppo_epochs):
            logprobs = self._get_current_logprobs(states, actions)

            # probability ratio: prob of action under new policy / prob of action under old policy
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            # clip the ratio to be between 1-epsilon and 1+epsilon
            surr2 = (
                torch.clamp(
                    ratio,
                    1 - self.config.clip_eps,
                    1 + self.config.clip_eps,
                )
                * advantages
            )
            # min(r_t(theta)A_t, clip(r_t(theta)A_t, 1-epsilon, 1+epsilon))
            actor_loss = -torch.min(surr1, surr2).mean()
            # a squared-error loss (V_theta(st) − V_target(st))2
            critic_loss = torch.nn.functional.mse_loss(
                self.critic(states).squeeze(), returns
            )

            # Entropy bonus
            entropy = self._get_entropy(states)
            entropy_loss = -self.config.entropy_coef * entropy.mean()
            total_actor_loss = actor_loss + entropy_loss

            # Update actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            total_loss += actor_loss.item() + critic_loss.item() + entropy_loss.item()
        # on-policy: clear memory
        self.memory.clear()

        return PPOUpdateLoss(
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            entropy_loss=entropy_loss.item(),
        )

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards, episode_losses, episode_steps = [], [], [0]

        while sum(episode_steps) < self.config.n_transactions:
            action_info = self.select_action(state)
            action, logprob = action_info["action"], action_info["logprob"]
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

        update_result = self.update()

        if update_result is not None:
            episode_losses.append(update_result)

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
