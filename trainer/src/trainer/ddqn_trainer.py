import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.rainbow_dqn import DQNNetwork
from schema.config import DoubleDQNConfig
from schema.result import DDQNUpdateLoss, SingleEpisodeResult
from trainer.base_trainer import BaseTrainer


class DDQNTrainer(BaseTrainer):
    name = "ddqn"
    config_class = DoubleDQNConfig

    def __init__(
        self,
        env_name: str,
        config_dict: dict,
        save_dir: str = "results/ddqn",
    ):
        super().__init__(env_name, config_dict, save_dir)
        self.total_steps = 0
        self.epsilon = self.config.eps_start

    def _init_models(self):
        self.policy_net = DQNNetwork(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            self.config.model.n_layers,
        ).to(self.config.device)
        self.target_net = DQNNetwork(
            self.state_dim,
            self.action_dim,
            self.config.model.hidden_dim,
            self.config.model.n_layers,
        ).to(self.config.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> dict:
        """Selects an action using epsilon-greedy policy."""
        # Epsilon decay
        self.epsilon = self.config.eps_end + (
            self.config.eps_start - self.config.eps_end
        ) * math.exp(-1.0 * self.total_steps / self.config.eps_decay)

        if not eval_mode and np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
            return {"action": action, "epsilon": self.epsilon}

        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
            action = q_values.argmax().item()

        return {
            "action": action,
            "q_values": q_values.cpu().numpy(),
            "epsilon": self.epsilon,
        }

    def update(self) -> dict:
        """Updates the network weights using a batch from the replay buffer."""
        if len(self.memory) < self.config.batch_size:
            return None

        # use PER
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.config.batch_size)
        )

        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.config.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.config.device)

        # get Q-values for the actions that were actually taken
        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            # select best actions for next states using the policy network
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)

            # evaluate the Q-values of these actions using the target network (Double DQN)
            next_q_values_target = self.target_net(next_states).gather(1, next_actions)

            # compute the target Q-value
            target_q_values = (
                rewards + (1 - dones) * self.config.gamma * next_q_values_target
            )

        # calculate loss (using Smooth L1 loss, which is less sensitive to outliers)
        elementwise_loss = F.smooth_l1_loss(
            current_q_values, target_q_values, reduction="none"
        )

        # update priorities in the PER buffer
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Apply importance sampling weights
        loss = (weights * elementwise_loss).mean()

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def _update_target_net(self):
        if self.total_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_episode(self) -> SingleEpisodeResult:
        state, _ = self.env.reset()
        done = False
        episode_rewards = []
        episode_losses = []
        episode_steps = 0

        for step in range(self.config.max_steps):
            self.total_steps += 1

            action_info = self.select_action(state)
            action = action_info["action"]

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_rewards.append(reward)
            episode_steps += 1

            update_result = self.update()
            if update_result:
                episode_losses.append(DDQNUpdateLoss(loss=update_result["loss"]))

            self._update_target_net()

            if done:
                break

        return SingleEpisodeResult(
            episode_total_reward=np.sum(episode_rewards),
            episode_steps=episode_steps,
            episode_losses=episode_losses,
        )

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_file)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_file))

    def eval_mode_on(self):
        self.policy_net.eval()

    def eval_mode_off(self):
        self.policy_net.train()
