import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.sac import LSTMSACPolicy, LSTMSACQNetwork
from schema.config import SACConfig
from schema.result import SACUpdateLoss
from trainer.sac_v2_trainer import SACV2Trainer


class SACSequentialTrainer(SACV2Trainer):
    name = "sac_seq"
    config_class = SACConfig

    def __init__(
        self, env_name: str, config: SACConfig, save_dir: str = "results/sac_seq"
    ):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        device = self.config.device
        model_config = self.config.model

        self.actor = LSTMSACPolicy(
            self.state_dim,
            self.action_dim,
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
        ).to(device)

        self.critic1 = LSTMSACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
        ).to(device)
        self.critic2 = LSTMSACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
        ).to(device)

        self.target_critic1 = LSTMSACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
        ).to(device)
        self.target_critic2 = LSTMSACQNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
        ).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -self.action_dim

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.config.lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.config.lr
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> dict:
        history = self.eval_state_history if eval_mode else self.state_history
        history.append(state)

        sequence = list(history)
        if len(sequence) < self.config.model.seq_len:
            padding = [
                np.zeros_like(state)
                for _ in range(self.config.model.seq_len - len(sequence))
            ]
            sequence = padding + sequence

        state_seq_np = np.array(sequence)[np.newaxis, ...]
        state_seq_tensor = torch.FloatTensor(state_seq_np).to(self.config.device)

        with torch.no_grad():
            action, _, _ = self.actor.sample(state_seq_tensor)

        return {"action": action.cpu().numpy().flatten()}

    def on_episode_end(self):
        self.state_history.clear()
        self.eval_state_history.clear()

    def update(self) -> SACUpdateLoss:
        """Update all networks using a batch of sequences from the buffer."""
        if len(self.memory) < self.config.batch_size:
            return SACUpdateLoss.empty()

        states_seq, actions_seq, rewards_seq, next_states_seq, dones_seq = (
            self.memory.sample(self.config.batch_size)
        )

        states_tensor = torch.FloatTensor(states_seq).to(self.config.device)
        actions_tensor = torch.FloatTensor(actions_seq).to(self.config.device)
        rewards_tensor = torch.FloatTensor(rewards_seq).to(self.config.device)
        next_states_tensor = torch.FloatTensor(next_states_seq).to(self.config.device)
        dones_tensor = torch.FloatTensor(dones_seq).to(self.config.device)

        last_actions = actions_tensor[:, -1, :]

        last_rewards = rewards_tensor[:, -1].unsqueeze(1)
        last_dones = dones_tensor[:, -1].unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states_tensor)

            target_q1 = self.target_critic1(next_states_tensor, next_actions)
            target_q2 = self.target_critic2(next_states_tensor, next_actions)
            min_target_q = torch.min(target_q1, target_q2)

            next_state_values = min_target_q - self.alpha * next_log_probs
            q_target = (
                last_rewards
                + (1.0 - last_dones) * self.config.gamma * next_state_values
            )

        current_q1 = self.critic1(states_tensor, last_actions)
        current_q2 = self.critic2(states_tensor, last_actions)

        critic1_loss = F.mse_loss(current_q1, q_target)
        critic2_loss = F.mse_loss(current_q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)

        if self.step_count % self.config.policy_update_interval == 0:
            pi_actions, log_probs, _ = self.actor.sample(states_tensor)

            q1_pi = self.critic1(states_tensor, pi_actions)
            q2_pi = self.critic2(states_tensor, pi_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (self.alpha * log_probs - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = (
                self.log_alpha * (-log_probs.detach() - self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        self._soft_update_target_networks()

        return SACUpdateLoss(
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha_loss=alpha_loss.item(),
            value_loss=0,
        )

    def _soft_update_target_networks(self):
        for param, target_param in zip(
            self.critic1.parameters(), self.target_critic1.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic2.parameters(), self.target_critic2.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
