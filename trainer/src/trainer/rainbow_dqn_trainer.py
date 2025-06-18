import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model.rainbow_dqn import DuelingNetwork, ReplayBuffer

class RainbowDQNTrainer:
    def __init__(self, env, config, save_dir="results/rainbow_dqn"):
        self.env = env
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingNetwork(env.observation_space.shape[0], env.action_space.n, config.get("hidden_dim", 256)).to(self.device)
        self.target_net = DuelingNetwork(env.observation_space.shape[0], env.action_space.n, config.get("hidden_dim", 256)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.get("lr", 3e-4))
        self.replay_buffer = ReplayBuffer(config.get("buffer_size", 1000000))
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
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.policy_net(next_state).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            expected_q_values = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.config.get("gamma", 0.99) * next_q_values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.episode_losses.append(loss.item())
        if self.config.get("target_update", 10) % self.config.get("target_update", 10) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, episodes=1000, max_steps=1000):
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            self.episode_losses = []
            for t in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.update()
                if done:
                    break
            avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0.0
            self.episode_returns.append(total_reward)
            self.scores.append(total_reward)
            self.losses.append(avg_loss)
            # Save best model
            if total_reward > self.best_score:
                self.best_score = total_reward
                torch.save(self.policy_net.state_dict(), self.model_file)
            # Log metrics
            with open(self.log_file, "a") as f:
                import json
                f.write(json.dumps({
                    "episode": ep+1,
                    "return": float(total_reward),
                    "loss": float(avg_loss)
                }) + "\n")
            print(f"Episode {ep+1}: Return={total_reward:.2f}, Loss={avg_loss:.4f}") 