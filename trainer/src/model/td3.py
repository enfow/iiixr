"""
TD3 Actor

Reference
---------
- [TD3: Twin Delayed DDPG](<https://arxiv.org/pdf/1802.09477>)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TD3Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        max_action: float = 1.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.max_action = max_action

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        self.shared_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.shared_layers(state)
        action = torch.tanh(self.output(x))
        return action * self.max_action


class TransformerTD3Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=256,
        nhead=8,
        n_layers=6,
        use_layernorm=False,
    ):
        super().__init__()
        self.max_action = max_action
        self.embedding = nn.Linear(state_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True,
        )

        norm_layer = nn.LayerNorm(hidden_dim) if use_layernorm else None

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=norm_layer,
            enable_nested_tensor=True,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence):
        embedded = self.embedding(state_sequence)
        transformed = self.transformer(embedded)
        action = torch.tanh(self.action_head(transformed[:, -1]))
        return action * self.max_action


class LSTMTD3Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        max_action: float = 1.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.max_action = max_action
        self.embedding = nn.Linear(state_dim, hidden_dim)
        self.embedding_norm = (
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.output_norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.out_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence: torch.Tensor) -> torch.Tensor:
        embedded = F.relu(self.embedding_norm(self.embedding(state_sequence)))
        lstm_output, _ = self.lstm(embedded)
        last_output = self.output_norm(lstm_output[:, -1, :])
        action = torch.tanh(self.out_layer(last_output))
        return action * self.max_action


class TD3Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()

        # Q1 network
        q1_layers = []
        q1_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        if use_layernorm:
            q1_layers.append(nn.LayerNorm(hidden_dim))
        q1_layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                q1_layers.append(nn.LayerNorm(hidden_dim))
            q1_layers.append(nn.ReLU())
        self.q1_layers = nn.Sequential(*q1_layers)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 network
        q2_layers = []
        q2_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        if use_layernorm:
            q2_layers.append(nn.LayerNorm(hidden_dim))
        q2_layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                q2_layers.append(nn.LayerNorm(hidden_dim))
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

    def get_q1_value(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_layers(sa)
        return self.q1_out(q1)
