"""
FORK: A FORWARD-LOOKING ACTOR FOR MODEL-FREE REINFORCEMENT LEARNING
https://arxiv.org/pdf/2010.01652
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from model.fork import RewardNetwork, SystemNetwork
from schema.config import TD3ForkConfig
from schema.result import TD3FORKUpdateLoss
from trainer.td3_trainer import TD3Trainer


class TD3FORKTrainer(TD3Trainer):
    name = "td3_fork"
    config_class = TD3ForkConfig

    def __init__(
        self, env_name: str, config: TD3ForkConfig, save_dir: str = "results/td3_fork"
    ):
        super().__init__(env_name, config, save_dir)

    def _init_models(self):
        super()._init_models()

        self.system_net = SystemNetwork(
            self.state_dim,
            self.action_dim,
            self.config.fork_hidden_dim,
            n_layers=self.config.fork_n_layers,
        ).to(self.config.device)

        self.reward_net = RewardNetwork(
            self.state_dim,
            self.action_dim,
            self.config.fork_hidden_dim,
            n_layers=self.config.fork_n_layers,
        ).to(self.config.device)

        self.system_optimizer = optim.Adam(
            self.system_net.parameters(), lr=self.config.lr
        )
        self.reward_optimizer = optim.Adam(
            self.reward_net.parameters(), lr=self.config.lr
        )

    def update(self) -> TD3FORKUpdateLoss:
        if len(self.memory) < self.config.batch_size:
            return None

        self.total_it += 1
        state, action, reward, next_state, done = self.memory.sample(
            self.config.batch_size
        )

        state = torch.FloatTensor(state).to(self.config.device)
        action = torch.FloatTensor(action).to(self.config.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.config.device)
        next_state = torch.FloatTensor(next_state).to(self.config.device)
        done = torch.BoolTensor(done).unsqueeze(1).to(self.config.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (~done) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # FORK update
        predicted_next_state = self.system_net(state, action)
        system_loss = F.mse_loss(predicted_next_state, next_state)

        self.system_optimizer.zero_grad()
        system_loss.backward()
        self.system_optimizer.step()

        # Reward Network Update
        predicted_reward = self.reward_net(state, action)
        reward_loss = F.mse_loss(predicted_reward, reward)

        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        actor_loss = None
        if self.total_it % self.config.policy_delay == 0:
            current_action = self.actor(state)
            standard_actor_loss = -self.critic.get_q1_value(
                state, current_action
            ).mean()

            imagined_next_state = self.system_net(state, current_action)
            forward_looking_loss = -self.critic.get_q1_value(
                imagined_next_state, self.actor(imagined_next_state)
            ).mean()

            actor_loss = (
                1 - self.config.fork_alpha
            ) * standard_actor_loss + self.config.fork_alpha * forward_looking_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data
                    + (1 - self.config.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data
                    + (1 - self.config.tau) * target_param.data
                )

        return TD3FORKUpdateLoss(
            actor_loss=actor_loss.item() if actor_loss is not None else 0.0,
            critic_loss=critic_loss.item(),
            system_loss=system_loss.item(),
            reward_loss=reward_loss.item(),
        )

    def save_model(self):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "system_net": self.system_net.state_dict(),
                "reward_net": self.reward_net.state_dict(),
            },
            self.model_file,
        )

    def load_model(self):
        checkpoint = torch.load(self.model_file)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.system_net.load_state_dict(checkpoint["system_net"])
        self.reward_net.load_state_dict(checkpoint["reward_net"])

    def eval_mode_on(self):
        super().eval_mode_on()
        self.system_net.eval()
        self.reward_net.eval()

    def eval_mode_off(self):
        super().eval_mode_off()
        self.system_net.train()
        self.reward_net.train()
