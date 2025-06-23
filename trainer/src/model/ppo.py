import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        is_discrete: bool = True,
    ):
        super().__init__()
        self.is_discrete = is_discrete
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared_layers = nn.Sequential(*layers)
        if is_discrete:
            self.output = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.shared_layers(state)
        if self.is_discrete:
            return torch.softmax(self.output(x), dim=-1)
        else:
            mean = self.mean(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, -20, 2)  # Prevent extreme values
            return mean, log_std


class Critic(nn.Module):
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
