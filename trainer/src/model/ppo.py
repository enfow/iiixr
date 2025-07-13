"""
PPO Models

Reference
---------
- [Proximal Policy Optimization Algorithms](<https://arxiv.org/pdf/1707.06347>)
"""

import math

import torch
import torch.nn as nn

N_TRANSFORMER_HEADS = 4


class DiscreteActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(state_dim, hidden_dim)])
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
        return torch.softmax(self.output(x), dim=-1)


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(state_dim, hidden_dim)])
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
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
        use_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(state_dim, hidden_dim)])
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
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
        use_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(state_dim, hidden_dim)])
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.fc_out(x)


class LSTMContinuousActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 1,
        n_fc_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()

        # FC layers before LSTM
        fc_layers = []
        fc_layers.append(nn.Linear(state_dim, hidden_dim))
        if use_layernorm:
            fc_layers.append(nn.LayerNorm(hidden_dim))
        fc_layers.append(nn.ReLU())

        for _ in range(n_fc_layers - 1):
            fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, hidden):
        batch_size, seq_len, _ = state.shape
        x = self.fc_layers(state.view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, -1)
        x, next_hidden = self.lstm(x, hidden)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std, next_hidden


class LSTMContinuousCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        n_fc_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()

        # FC layers before LSTM
        fc_layers = []
        fc_layers.append(nn.Linear(state_dim, hidden_dim))
        if use_layernorm:
            fc_layers.append(nn.LayerNorm(hidden_dim))
        fc_layers.append(nn.ReLU())

        for _ in range(n_fc_layers - 1):
            fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape
        x = self.fc_layers(x.view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, -1)
        x, next_hidden = self.lstm(x, hidden)
        value = self.fc_out(x)
        return value, next_hidden


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerContinuousActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_head: int = 4,
        n_layers: int = 2,
        n_fc_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # FC layers before transformer
        fc_layers = []
        fc_layers.append(nn.Linear(state_dim, hidden_dim))
        if use_layernorm:
            fc_layers.append(nn.LayerNorm(hidden_dim))
        fc_layers.append(nn.ReLU())

        for _ in range(n_fc_layers - 1):
            fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.layernorm = nn.LayerNorm(hidden_dim) if use_layernorm else None

        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=n_layers
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence):
        batch_size, seq_len, _ = state_sequence.shape
        x = self.fc_layers(state_sequence.view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, -1)
        x = self.pos_encoder(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            state_sequence.device
        )
        x = self.transformer_encoder(x, mask=causal_mask)
        if self.layernorm is not None:
            x = self.layernorm(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class TransformerContinuousCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_head: int = 4,
        n_layers: int = 2,
        n_fc_layers: int = 2,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # FC layers before transformer
        fc_layers = []
        fc_layers.append(nn.Linear(state_dim, hidden_dim))
        if use_layernorm:
            fc_layers.append(nn.LayerNorm(hidden_dim))
        fc_layers.append(nn.ReLU())

        for _ in range(n_fc_layers - 1):
            fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                fc_layers.append(nn.LayerNorm(hidden_dim))
            fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=n_layers
        )

        self.layernorm = nn.LayerNorm(hidden_dim) if use_layernorm else None
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_sequence):
        batch_size, seq_len, _ = state_sequence.shape
        x = self.fc_layers(state_sequence.view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, -1)
        x = self.pos_encoder(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            state_sequence.device
        )
        x = self.transformer_encoder(x, mask=causal_mask)
        if self.layernorm is not None:
            x = self.layernorm(x)
        value = self.value_head(x)
        return value
