# model/fork.py

import torch
import torch.nn as nn


class SystemNetwork(nn.Module):
    """
    Predicts the next state given the current state and action.
    s_{t+1} = f(s_t, a_t)
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, n_layers: int = 2
    ):
        print(
            f"SystemNetwork: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}, n_layers={n_layers}"
        )
        super().__init__()
        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)


class RewardNetwork(nn.Module):
    """
    Predicts the immediate reward given the current state and action.
    r_t = r(s_t, a_t)
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, n_layers: int = 2
    ):
        print(
            f"RewardNetwork: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}, n_layers={n_layers}"
        )
        super().__init__()
        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)
