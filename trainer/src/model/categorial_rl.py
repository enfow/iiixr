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
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        # register the support as a buffer, so it's moved to the correct device
        # but is not considered a model parameter.
        self.register_buffer(
            "support", torch.linspace(self.v_min, self.v_max, self.n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # create the network layers
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.hidden_layers = nn.Sequential(*layers)
        # final layer to output the logits for the categorical distribution
        self.output_layer = nn.Linear(hidden_dim, action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        # get the logits for each atom of each action
        q_atoms_logits = self.output_layer(x).view(-1, self.action_dim, self.n_atoms)
        # apply softmax to get the probability distribution
        dist = F.softmax(q_atoms_logits, dim=-1)
        # clamp the distribution to avoid nans during loss calculation
        dist = dist.clamp(min=1e-3)
        return dist

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.forward(x)
        # q-value is the expectation of the distribution (sum of atom_value * prob)
        q_values = (dist * self.support).sum(2)
        return q_values
