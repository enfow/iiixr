import torch
import torch.nn as nn
import torch.nn.functional as F


class SACPolicy(nn.Module):
    """
    Stochastic policy network that outputs mean and log_std for Gaussian policy.
    Uses reparameterization trick for sampling actions.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        n_layers=2,
        log_std_min=-20,
        log_std_max=2,
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

    def __init__(self, state_dim, action_dim, hidden_dim=256, n_layers=2):
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

    def __init__(self, state_dim, hidden_dim=256, n_layers=2):
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
