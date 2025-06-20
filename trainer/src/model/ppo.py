import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, is_discrete=True):
        super().__init__()
        self.is_discrete = is_discrete
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if is_discrete:
            self.output = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.shared_layers(state)
        if self.is_discrete:
            return torch.softmax(self.output(x), dim=-1)
        else:
            mean = self.mean(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, -20, 2)  # Prevent extreme values
            return mean, log_std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
