import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.sac import GaussianPolicy, QNetwork, ReplayBuffer


class SACTrainer:
    def __init__(self, env, config, save_dir="results/sac"):
        self.env = env
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if action space is discrete
        self.is_discrete = hasattr(env.action_space, "n")
        action_dim = (
            env.action_space.n if self.is_discrete else env.action_space.shape[0]
        )

        self.actor = GaussianPolicy(
            env.observation_space.shape[0],
            action_dim,
            config.get("hidden_dim", 256),
            is_discrete=self.is_discrete,
        ).to(self.device)

        self.critic1 = QNetwork(
            env.observation_space.shape[0], action_dim, config.get("hidden_dim", 256)
        ).to(self.device)
        self.critic2 = QNetwork(
            env.observation_space.shape[0], action_dim, config.get("hidden_dim", 256)
        ).to(self.device)
        self.target_critic1 = QNetwork(
            env.observation_space.shape[0], action_dim, config.get("hidden_dim", 256)
        ).to(self.device)
        self.target_critic2 = QNetwork(
            env.observation_space.shape[0], action_dim, config.get("hidden_dim", 256)
        ).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.get("lr", 3e-4)
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=config.get("lr", 3e-4)
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=config.get("lr", 3e-4)
        )

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
        action, log_prob, _ = self.actor.sample(state)
        if self.is_discrete:
            return action.item()
        return action.detach().cpu().numpy()

    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = (
            torch.LongTensor(action).to(self.device)
            if self.is_discrete
            else torch.FloatTensor(action).to(self.device)
        )
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            if self.is_discrete:
                next_action_one_hot = F.one_hot(
                    next_action, num_classes=self.actor.action_dim
                ).float()
                target_q1 = self.target_critic1(next_state, next_action_one_hot)
                target_q2 = self.target_critic2(next_state, next_action_one_hot)
            else:
                target_q1 = self.target_critic1(next_state, next_action)
                target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.config.get(
                "gamma", 0.99
            ) * (target_q - self.config.get("alpha", 0.2) * next_log_prob.unsqueeze(1))

        if self.is_discrete:
            action_one_hot = F.one_hot(
                action, num_classes=self.actor.action_dim
            ).float()
            current_q1 = self.critic1(state, action_one_hot)
            current_q2 = self.critic2(state, action_one_hot)
        else:
            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_action, log_prob, _ = self.actor.sample(state)
        if self.is_discrete:
            new_action_one_hot = F.one_hot(
                new_action, num_classes=self.actor.action_dim
            ).float()
            q1 = self.critic1(state, new_action_one_hot)
            q2 = self.critic2(state, new_action_one_hot)
        else:
            q1 = self.critic1(state, new_action)
            q2 = self.critic2(state, new_action)

        q = torch.min(q1, q2)
        actor_loss = (self.config.get("alpha", 0.2) * log_prob.unsqueeze(1) - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(
            self.critic1.parameters(), self.target_critic1.parameters()
        ):
            target_param.data.copy_(
                self.config.get("tau", 0.005) * param.data
                + (1 - self.config.get("tau", 0.005)) * target_param.data
            )
        for param, target_param in zip(
            self.critic2.parameters(), self.target_critic2.parameters()
        ):
            target_param.data.copy_(
                self.config.get("tau", 0.005) * param.data
                + (1 - self.config.get("tau", 0.005)) * target_param.data
            )

        self.episode_losses.append(
            actor_loss.item() + critic1_loss.item() + critic2_loss.item()
        )

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
                torch.save(
                    {
                        "actor": self.actor.state_dict(),
                        "critic1": self.critic1.state_dict(),
                        "critic2": self.critic2.state_dict(),
                    },
                    self.model_file,
                )
            # Log metrics
            with open(self.log_file, "a") as f:
                import json

                f.write(
                    json.dumps(
                        {
                            "episode": ep + 1,
                            "return": float(total_reward),
                            "loss": float(avg_loss),
                        }
                    )
                    + "\n"
                )
            print(f"Episode {ep + 1}: Return={total_reward:.2f}, Loss={avg_loss:.4f}")
