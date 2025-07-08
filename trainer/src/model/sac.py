import torch
import torch.nn as nn


class SACPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.hidden_layers(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Sample with reparameterization
        action = torch.tanh(z)  # Squash to [-1, 1]

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class SACQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.hidden_layers(x)
        return self.fc_out(x)


class SACValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.hidden_layers(state)
        return self.fc_out(x)


class LSTMSACPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.shared_layers = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence, hidden_state=None):
        if state_sequence.dim() == 2:
            state_sequence = state_sequence.unsqueeze(1)
        batch_size, seq_len, _ = state_sequence.shape

        embedded_state = self.shared_layers(
            state_sequence.reshape(batch_size * seq_len, -1)
        )
        embedded_state = embedded_state.view(batch_size, seq_len, -1)
        lstm_out, next_hidden = self.lstm(embedded_state, hidden_state)

        mean = self.mean_head(lstm_out)
        log_std = self.log_std_head(lstm_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std, next_hidden

    def evaluate(self, state_sequence, hidden_state=None):
        mean, log_std, next_hidden = self.forward(state_sequence, hidden_state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        actions = torch.tanh(z)
        log_probs = normal.log_prob(z) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        return actions, log_probs, next_hidden

    def sample(self, state_sequence, hidden_state=None):
        mean, log_std, next_hidden = self.forward(state_sequence, hidden_state)
        last_mean = mean[:, -1, :]
        last_log_std = log_std[:, -1, :]
        std = last_log_std.exp()
        normal = torch.distributions.Normal(last_mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, next_hidden


class LSTMSACQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()
        self.shared_layers = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_sequence, action_sequence, hidden_state=None):
        if state_sequence.dim() == 2:
            state_sequence = state_sequence.unsqueeze(1)

        batch_size, seq_len, _ = state_sequence.shape

        if action_sequence.dim() == 2:
            action_sequence = action_sequence.unsqueeze(1).repeat(1, seq_len, 1)

        embedded_state = self.shared_layers(
            state_sequence.reshape(batch_size * seq_len, -1)
        )
        embedded_state = embedded_state.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(embedded_state, hidden_state)

        combined = torch.cat([lstm_out, action_sequence], dim=2)

        combined_reshaped = combined.reshape(batch_size * seq_len, -1)
        q_value_reshaped = self.q_head(combined_reshaped)

        q_value_sequence = q_value_reshaped.view(batch_size, seq_len, 1)

        return q_value_sequence
