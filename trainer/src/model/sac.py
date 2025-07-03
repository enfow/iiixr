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

        # Shared MLP to process each state observation individually
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # LSTM layer to process the sequence of embedded states
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # Output heads for policy parameters
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_sequence, hidden_state=None):
        """
        Processes a sequence of states.
        :param state_sequence: Shape (batch_size, seq_len, state_dim)
        :param hidden_state: Initial hidden state for the LSTM
        :return: mean, log_std for each step in the sequence, and the next hidden state.
        """
        batch_size, seq_len, _ = state_sequence.shape

        # Process each state through the shared layers
        embedded_state = self.shared_layers(
            state_sequence.view(batch_size * seq_len, -1)
        )
        embedded_state = embedded_state.view(batch_size, seq_len, -1)

        # Pass the sequence of embedded states through the LSTM
        lstm_out, next_hidden = self.lstm(embedded_state, hidden_state)

        # Get policy parameters from the LSTM's output sequence
        mean = self.mean_head(lstm_out)
        log_std = self.log_std_head(lstm_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, next_hidden

    def sample(self, state_sequence, hidden_state=None):
        mean, log_std, next_hidden = self.forward(state_sequence, hidden_state)

        # Use the last output of the sequence for action selection
        last_mean = mean[:, -1, :]
        last_log_std = log_std[:, -1, :]

        std = last_log_std.exp()
        normal = torch.distributions.Normal(last_mean, std)

        # Reparameterization trick
        z = normal.rsample()
        action = torch.tanh(z)

        # Calculate log_prob with correction for tanh squashing
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
        # Shared MLP to process each state observation
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # LSTM layer to process the sequence of embedded states
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        # Final MLP to compute Q-value from the combined state-feature and action
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_sequence, action, hidden_state=None):
        batch_size, seq_len, _ = state_sequence.shape

        # Process each state through the shared layers
        embedded_state = self.shared_layers(
            state_sequence.view(batch_size * seq_len, -1)
        )
        embedded_state = embedded_state.view(batch_size, seq_len, -1)

        # Pass the sequence of embedded states through the LSTM
        lstm_out, _ = self.lstm(embedded_state, hidden_state)

        # Use the last feature vector from the LSTM output
        last_state_feature = lstm_out[:, -1, :]

        # Combine the final state feature with the action and compute Q-value
        combined = torch.cat([last_state_feature, action], dim=1)
        q_value = self.q_head(combined)

        return q_value
