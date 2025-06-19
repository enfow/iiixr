# Discrete SAC implementation for LunarLander-v3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random


def to_one_hot(action, action_dim):
    return F.one_hot(action, num_classes=action_dim).float()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class DiscreteQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # [batch_size, action_dim]


class DiscreteSAC:
    def __init__(self, env, config):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor = DiscretePolicy(self.state_dim, self.action_dim).to(self.device)
        self.critic1 = DiscreteQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = DiscreteQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic1 = DiscreteQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic2 = DiscreteQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.buffer = ReplayBuffer(config.get("buffer_size", 100_000))

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get("lr", 3e-4))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.get("lr", 3e-4))
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.get("lr", 3e-4))

        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.target_entropy = -np.log(1.0 / self.action_dim) * config.get("entropy_coef", 1.0)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.get("lr", 3e-4))
        self.alpha = self.log_alpha.exp()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(probs).sample()
            return action.item()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Target Q
        with torch.no_grad():
            next_logits = self.actor(next_state)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            target_q1 = self.target_critic1(next_state)
            target_q2 = self.target_critic2(next_state)
            target_q = torch.min(target_q1, target_q2)
            next_v = (next_probs * (target_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target = reward + (1 - done) * self.gamma * next_v

        # Current Q
        q1 = self.critic1(state).gather(1, action.unsqueeze(1))
        q2 = self.critic2(state).gather(1, action.unsqueeze(1))

        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        q1_all = self.critic1(state)
        q2_all = self.critic2(state)
        min_q = torch.min(q1_all, q2_all)

        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss
        entropy = - (probs * log_probs).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Soft updates
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, episodes, max_steps, batch_size, start_steps=1000):
        total_steps = 0
        for ep in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                if total_steps < start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                total_steps += 1

                self.update(batch_size)

                if done:
                    break

            print(f"Episode {ep + 1}, Reward: {episode_reward:.2f}, Alpha: {self.alpha.item():.4f}")


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 100000,
        "entropy_coef": 0.98,
    }
    sac = DiscreteSAC(env, config)
    sac.train(episodes=500, max_steps=1000, batch_size=256)
