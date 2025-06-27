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


class TransformerTD3Actor(nn.Module):
    def __init__(
        self, state_dim, action_dim, max_action, d_model=256, nhead=8, num_layers=6
    ):
        super().__init__()
        self.max_action = max_action
        self.embedding = nn.Linear(state_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, state_sequence):
        # state_sequence: (batch, seq_len, state_dim)
        embedded = self.embedding(state_sequence).transpose(
            0, 1
        )  # (seq_len, batch, d_model)
        transformed = self.transformer(embedded)
        action = torch.tanh(self.action_head(transformed[-1]))  # [-1, 1]
        return action * self.max_action  # Scale to [-max_action, max_action]c


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
        # self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 network
        q2_layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
        self.q2_layers = nn.Sequential(*q2_layers)
        # self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = self.q1_layers(sa)
        # q1 = self.q1_out(q1)

        q2 = self.q2_layers(sa)
        # q2 = self.q2_out(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_layers(sa)
        # return self.q1_out(q1)
        return q1
