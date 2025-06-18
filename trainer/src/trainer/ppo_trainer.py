import os
import torch
import torch.optim as optim
import numpy as np
from model.ppo import Actor, Critic, PPOMemory

class PPOTrainer:
    def __init__(self, env, config, save_dir="results/ppo"):
        self.env = env
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(env.observation_space.shape[0], env.action_space.n, config.get("hidden_dim", 64)).to(self.device)
        self.critic = Critic(env.observation_space.shape[0], config.get("hidden_dim", 64)).to(self.device)
        self.memory = PPOMemory()
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.get("lr", 3e-4))
        self.best_score = -np.inf
        self.scores = []
        self.losses = []
        self.episode_returns = []
        self.episode_losses = []
        self.log_file = os.path.join(self.save_dir, "metrics.jsonl")
        self.model_file = os.path.join(self.save_dir, "best_model.pth")
        self.config_file = os.path.join(self.save_dir, "config.json")
        with open(self.config_file, "w") as f:
            import json
            json.dump(config, f, indent=2)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
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
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(self.device)
        returns = torch.FloatTensor(self.compute_returns(self.memory.rewards, self.memory.dones, self.config.get("gamma", 0.99))).to(self.device)
        advantages = returns - self.critic(states).squeeze().detach()
        for _ in range(self.config.get("ppo_epochs", 4)):
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            logprobs = dist.log_prob(actions)
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.get("clip_eps", 0.2), 1 + self.config.get("clip_eps", 0.2)) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.mse_loss(self.critic(states).squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.episode_losses.append(loss.item())
        self.memory.clear()

    def train(self, episodes=1000, max_steps=1000):
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            self.episode_losses = []
            for t in range(max_steps):
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
            self.episode_returns.append(total_reward)
            self.scores.append(total_reward)
            self.losses.append(avg_loss)
            # Save best model
            if total_reward > self.best_score:
                self.best_score = total_reward
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()
                }, self.model_file)
            # Log metrics
            with open(self.log_file, "a") as f:
                import json
                f.write(json.dumps({
                    "episode": ep+1,
                    "return": float(total_reward),
                    "loss": float(avg_loss)
                }) + "\n")
            print(f"Episode {ep+1}: Return={total_reward:.2f}, Loss={avg_loss:.4f}") 