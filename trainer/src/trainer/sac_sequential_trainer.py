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
        self.actor_hidden = None
        self.eval_actor_hidden = None

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
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.config.device)
            )

            if eval_mode:
                action, _, self.eval_actor_hidden = self.actor.sample(
                    state_tensor, self.eval_actor_hidden
                )
            else:
                action, _, self.actor_hidden = self.actor.sample(
                    state_tensor, self.actor_hidden
                )

        return {"action": action.cpu().numpy().flatten()}

    def on_episode_end(self):
        print("on_episode_end")
        self.state_history.clear()
        self.eval_state_history.clear()
        self.actor_hidden = None
        self.eval_actor_hidden = None

    def update(self) -> SACUpdateLoss:
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

        with torch.no_grad():
            next_actions_seq, next_log_probs_seq, _ = self.actor.evaluate(
                next_states_tensor
            )

            target_q1_seq = self.target_critic1(next_states_tensor, next_actions_seq)
            target_q2_seq = self.target_critic2(next_states_tensor, next_actions_seq)
            min_target_q_seq = torch.min(target_q1_seq, target_q2_seq)

            next_state_values_seq = min_target_q_seq - self.alpha * next_log_probs_seq

            last_rewards = rewards_tensor[:, -1].unsqueeze(1)
            last_dones = dones_tensor[:, -1].unsqueeze(1)

            last_next_state_value = next_state_values_seq[:, -1, :]

            q_target = (
                last_rewards
                + (1.0 - last_dones) * self.config.gamma * last_next_state_value
            )

        current_q1_seq = self.critic1(states_tensor, actions_tensor)
        current_q2_seq = self.critic2(states_tensor, actions_tensor)

        current_q1 = current_q1_seq[:, -1, :]
        current_q2 = current_q2_seq[:, -1, :]

        critic1_loss = F.mse_loss(current_q1, q_target)
        critic2_loss = F.mse_loss(current_q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.config.device)
        alpha_loss = torch.tensor(0.0, device=self.config.device)

        if self.step_count % self.config.policy_update_interval == 0:
            pi_actions_seq, log_probs_seq, _ = self.actor.evaluate(states_tensor)

            q1_pi_seq = self.critic1(states_tensor, pi_actions_seq)
            q2_pi_seq = self.critic2(states_tensor, pi_actions_seq)
            min_q_pi_seq = torch.min(q1_pi_seq, q2_pi_seq)

            actor_loss = (self.alpha.detach() * log_probs_seq - min_q_pi_seq).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = (
                self.log_alpha * (-log_probs_seq.detach() - self.target_entropy)
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
