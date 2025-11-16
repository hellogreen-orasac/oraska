import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim: int = 384, output_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(text_features)
        embedding = F.normalize(embedding, dim=-1)
        return embedding
    
    def predict_reward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.reward_head(embedding).squeeze(-1)
    
    def update(self, text_features: torch.Tensor, actual_rewards: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()
        embeddings = self.forward(text_features)
        predicted_rewards = self.predict_reward(embeddings)
        loss = F.mse_loss(predicted_rewards, actual_rewards)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()