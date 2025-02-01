import torch
import torch.nn as nn
import torch.nn.functional as F

class DeterministicActor(nn.Module):
    """TD3 Actor: Predicts accident score + fixation point (deterministic)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.action_head(x)  # Remove sigmoid and tanh from here
        return actions

class TwinCritic(nn.Module):
    """TD3 Twin Q-Networks"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 Network
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 Network
        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.Q1(x), self.Q2(x)