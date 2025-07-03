import torch
import torch.nn as nn


class SACPolicy(nn.Module):
    """
    Stochastic policy network that outputs mean and log_std for Gaussian policy.
    Uses reparameterization trick for sampling actions.
    """

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

        # Dynamic hidden layers
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass returns mean and log_std for Gaussian policy
        """
        x = self.hidden_layers(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        """
        Sample action using reparameterization trick and return action + log_prob
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Sample with reparameterization
        action = torch.tanh(z)  # Squash to [-1, 1]

        # Compute log probability with Jacobian correction for tanh
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class SACQNetwork(nn.Module):
    """
    Q-network for continuous actions: Q(s,a) -> R
    Takes state-action pairs as input
    """

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
        """
        Forward pass: concatenate state and action, then process
        """
        x = torch.cat([state, action], dim=1)
        x = self.hidden_layers(x)
        return self.fc_out(x)


class SACValueNetwork(nn.Module):
    """
    Value network: V(s) -> R
    Estimates the state value function
    """

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
        """
        Forward pass: state -> value
        """
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
        """Returns sequences of actions and log-probs for training."""
        mean, log_std, next_hidden = self.forward(state_sequence, hidden_state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        actions = torch.tanh(z)
        log_probs = normal.log_prob(z) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        return actions, log_probs, next_hidden

    def sample(self, state_sequence, hidden_state=None):
        """Returns a single action from the last step for interaction."""
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
        """Processes sequences of states and actions to return a sequence of Q-values."""
        if state_sequence.dim() == 2:
            state_sequence = state_sequence.unsqueeze(1)

        batch_size, seq_len, _ = state_sequence.shape

        # If action_sequence is a single action, repeat it for the whole sequence
        if action_sequence.dim() == 2:
            action_sequence = action_sequence.unsqueeze(1).repeat(1, seq_len, 1)

        embedded_state = self.shared_layers(
            state_sequence.reshape(batch_size * seq_len, -1)
        )
        embedded_state = embedded_state.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(embedded_state, hidden_state)

        # Combine LSTM features with actions at each time step
        combined = torch.cat([lstm_out, action_sequence], dim=2)

        # Reshape and compute Q-values
        combined_reshaped = combined.reshape(batch_size * seq_len, -1)
        q_value_reshaped = self.q_head(combined_reshaped)

        # Reshape Q-values back to sequence format
        q_value_sequence = q_value_reshaped.view(batch_size, seq_len, 1)

        return q_value_sequence
