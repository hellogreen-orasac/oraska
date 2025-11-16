import torch
import torch.nn as nn
from typing import Tuple

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        action = self.policy_head(features)
        value = self.value_head(features)
        return action, value
    
    def select_action(self, state: torch.Tensor, explore: bool = True, noise_scale: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self.forward(state)
            if explore:
                noise = torch.randn_like(action) * noise_scale
                action = torch.clamp(action + noise, -1, 1)
        return action