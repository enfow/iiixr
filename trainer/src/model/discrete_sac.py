"""
Discrete SAC implementation
- Ref: https://arxiv.org/pdf/1910.07207
"""

import torch.nn as nn
import torch.nn.functional as F


class DiscreteSACQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # [batch_size, action_dim]
