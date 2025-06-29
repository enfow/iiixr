"""
Categorical DQN Network(A.K.A C51)

Reference
---------
- [A Distributional Perspective on Reinforcement Learning](<https://arxiv.org/pdf/1707.06887>)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalDQNNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_atoms: int,
        v_min: float,
        v_max: float,
        n_layers: int = 2,
    ):
        """
        Args
        ----
        - n_atoms: the number of atoms(categories) in the categorical distribution
        - v_max: the maximum value of the support (set to maximum return of env)
        - v_min: the minimum value of the support (set to minimum return of env)
        """
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer(
            "support", torch.linspace(self.v_min, self.v_max, self.n_atoms)
        )  # define categorical support
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        q_atoms_logits = self.output_layer(x).view(-1, self.action_dim, self.n_atoms)
        dist = F.softmax(q_atoms_logits, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.forward(x)
        q_values = (dist * self.support).sum(2)
        return q_values
