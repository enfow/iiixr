"""
PPO Models

Reference
---------
- [Proximal Policy Optimization Algorithms](<https://arxiv.org/pdf/1707.06347>)
"""

import torch
import torch.nn as nn


class DiscreteActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.shared_layers(state)
        return torch.softmax(self.output(x), dim=-1)


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.fc_out(x)


class ContinuousActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared_layers = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.shared_layers(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class ContinuousCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.fc_out(x)
