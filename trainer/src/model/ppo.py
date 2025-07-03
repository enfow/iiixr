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
        use_layernorm: bool = False,
    ):
        super().__init__()
        # Shared layers to process individual states before the LSTM
        layers = [nn.Linear(state_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        # Note: The original code only used one shared layer regardless of n_layers.
        # This implementation can be extended if multi-layer pre-processing is needed.
        self.shared_layers = nn.Sequential(*layers)

        # LSTM layer for processing sequences
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )

        # Output heads for the policy
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, hidden):
        batch_size, seq_len, _ = state.shape
        # Flatten sequence for shared layers, then reshape back
        x = self.shared_layers(state.view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, -1)

        # Pass through LSTM
        x, next_hidden = self.lstm(x, hidden)

        # Get policy parameters
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
        return mean, log_std, next_hidden


class LSTMContinuousCritic(nn.Module):
    """
    A continuous critic model that uses an LSTM layer to estimate the state-value function
    from a sequence of states.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        use_layernorm: bool = False,
    ):
        super().__init__()
        # Layers to process individual states
        layers = [nn.Linear(state_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )

        # Output head for the value function
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape
        # Flatten sequence, process, then reshape
        x = self.hidden_layers(x.view(batch_size * seq_len, -1))
        x = x.view(batch_size, seq_len, -1)

        # Pass through LSTM
        x, next_hidden = self.lstm(x, hidden)

        # Get state value
        value = self.fc_out(x)
        return value, next_hidden


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input sequence for a Transformer model.
    """

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
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerContinuousActor(nn.Module):
    """
    A continuous actor model using a Transformer encoder to process sequences of states.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_head: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(state_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=n_layers
        )

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence):
        seq_len = state_sequence.size(1)

        # Embed input and add positional encoding
        x = self.input_embed(state_sequence)
        x = self.pos_encoder(x)

        # Create a causal mask to prevent attending to future states
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            state_sequence.device
        )

        # Pass through transformer encoder
        x = self.transformer_encoder(x, mask=causal_mask)

        # Get policy parameters
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class TransformerContinuousCritic(nn.Module):
    """
    A continuous critic model using a Transformer encoder to estimate state-value.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_head: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(state_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=n_layers
        )

        # Output head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_sequence):
        seq_len = state_sequence.size(1)

        # Embed and add positional encoding
        x = self.input_embed(state_sequence)
        x = self.pos_encoder(x)

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            state_sequence.device
        )

        # Pass through transformer
        x = self.transformer_encoder(x, mask=causal_mask)

        # Get state value
        value = self.value_head(x)
        return value
