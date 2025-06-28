import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NoisyLinear(nn.Module):
    """
    Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear layer
        out_features (int): output size of linear layer
        std_init (float): initial standard deviation of noisy linear parameters
        weight_mu (nn.Parameter): mean value of weight parameter
        weight_sigma (nn.Parameter): standard deviation of weight parameter
        bias_mu (nn.Parameter): mean value of bias parameter
        bias_sigma (nn.Parameter): standard deviation of bias parameter
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-pass of noisy linear layer."""
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class CategoricalDuelingNetwork(nn.Module):
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
        self.register_buffer(
            "support", torch.linspace(self.v_min, self.v_max, self.n_atoms)
        )  # support tensor, shape: (n_atoms,) from v_min to v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        layers = [NoisyLinear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([NoisyLinear(hidden_dim, hidden_dim), nn.ReLU()])
        self.hidden_layers = nn.Sequential(*layers)

        self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage_layer = NoisyLinear(hidden_dim, action_dim * n_atoms)

        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value_layer = NoisyLinear(hidden_dim, n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)

        adv_hid = F.relu(self.advantage_hidden(x))
        val_hid = F.relu(self.value_hidden(x))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.action_dim, self.n_atoms
        )
        value = self.value_layer(val_hid).view(-1, 1, self.n_atoms)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset noise for all noisy linear layers."""
        self.advantage_hidden.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden.reset_noise()
        self.value_layer.reset_noise()
