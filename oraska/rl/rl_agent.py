import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple
from oraska.config import config
from oraska.rl.policy_network import PolicyNetwork

class ExperienceBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)

class RLAgent:
    def __init__(self, agent_id: int, state_dim: int, action_dim: int):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.target_net = PolicyNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.stats = {'policy_loss': 0.0, 'value': 0.0, 'updates': 0}
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.policy_net.select_action(state_tensor, explore=explore)
        return action.squeeze(0).numpy()
    
    def action_to_params(self, action: np.ndarray) -> Dict:
        temperature = self._scale_param(action[0], config.TEMPERATURE_MIN, config.TEMPERATURE_MAX)
        top_p = self._scale_param(action[1], config.TOP_P_MIN, config.TOP_P_MAX)
        max_tokens = int(self._scale_param(action[2], config.MAX_TOKENS_MIN, config.MAX_TOKENS_MAX))
        if len(action) > 3:
            model_logits = action[3:]
            model_probs = F.softmax(torch.FloatTensor(model_logits), dim=0).numpy()
            model_idx = np.argmax(model_probs)
        else:
            model_idx = 0
            model_probs = [1.0]
        return {
            'temperature': float(temperature),
            'top_p': float(top_p),
            'max_tokens': max_tokens,
            'model_idx': int(model_idx),
            'model_probs': model_probs.tolist() if len(action) > 3 else [1.0]
        }
    
    def _scale_param(self, action_val: float, min_val: float, max_val: float) -> float:
        normalized = (action_val + 1) / 2
        return min_val + normalized * (max_val - min_val)
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> float:
        _, current_values = self.policy_net(states)
        current_values = current_values.squeeze(-1)
        with torch.no_grad():
            _, next_values = self.target_net(next_states)
            next_values = next_values.squeeze(-1)
            targets = rewards + self.gamma * next_values * (1 - dones)
        value_loss = F.mse_loss(current_values, targets)
        policy_actions, values = self.policy_net(states)
        policy_loss = -values.mean()
        total_loss = value_loss + 0.5 * policy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._soft_update()
        self.stats['policy_loss'] = total_loss.item()
        self.stats['value'] = current_values.mean().item()
        self.stats['updates'] += 1
        return total_loss.item()
    
    def _soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.stats = checkpoint['stats']