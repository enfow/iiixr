import torch
import torch.nn as nn


class TD3Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        max_action: float = 1.0,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.shared_layers(state)
        action = torch.tanh(self.output(x))
        return action * self.max_action


class TD3Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        # Q1 network
        q1_layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
        self.q1_layers = nn.Sequential(*q1_layers)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 network
        q2_layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
        self.q2_layers = nn.Sequential(*q2_layers)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = self.q1_layers(sa)
        q1 = self.q1_out(q1)

        q2 = self.q2_layers(sa)
        q2 = self.q2_out(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_layers(sa)
        return self.q1_out(q1)
