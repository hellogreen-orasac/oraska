"""
Oraska v9.3.0 - Centralized Critic for MADDPG
基于论文: Multi-Agent Actor-Critic (arXiv:1706.02275v4)

关键改进:
1. Critic 输入所有 agent 的 actions (centralized training)
2. Actor 只用自己的 observation (decentralized execution)
3. 独立的 Reward 信号支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CentralizedCritic(nn.Module):
    """
    中心化 Critic 网络
    
    输入:
        - global_state: [batch, state_dim] - 全局状态表示
        - all_actions: [batch, num_agents * action_dim] - 所有 agent 的 actions
    
    输出:
        - q_value: [batch, 1] - 该状态-动作对的价值
    
    理论依据 (MADDPG 论文 Equation 6):
        Q^μᵢ(x, a₁, ..., aₙ) = 𝔼[Rᵢ | x, a₁, ..., aₙ]
    """
    def __init__(self, state_dim: int, num_agents: int, action_dim: int):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # 输入维度 = 全局状态 + 所有 actions
        input_dim = state_dim + num_agents * action_dim
        
        # 深层网络以捕捉复杂的多 agent 交互
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 1)  # 单值输出
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, global_state: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            global_state: [batch, state_dim]
            all_actions: [batch, num_agents * action_dim]
        
        Returns:
            q_values: [batch, 1]
        """
        x = torch.cat([global_state, all_actions], dim=-1)
        return self.network(x)


class MADDPGAgent:
    """
    完整的 MADDPG Agent 实现
    
    核心特性:
    1. Decentralized Actor: πᵢ(aᵢ|oᵢ) - 只用自己的观察
    2. Centralized Critic: Q(x, a₁, ..., aₙ) - 用全局信息
    3. 独立 Reward: 每个 agent 有自己的任务目标
    """
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        
        # Actor 网络 (局部决策)
        from oraska.rl.policy_network import PolicyNetwork
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.target_actor = PolicyNetwork(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Critic 网络 (全局评估)
        self.critic = CentralizedCritic(state_dim, num_agents, action_dim)
        self.target_critic = CentralizedCritic(state_dim, num_agents, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=lr_actor,
            weight_decay=1e-5
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), 
            lr=lr_critic,
            weight_decay=1e-5
        )
        
        # 统计信息
        self.stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'q_value': 0.0,
            'updates': 0
        }
    
    def select_action(self, observation: torch.Tensor, explore: bool = True, noise_scale: float = 0.1) -> torch.Tensor:
        """
        选择动作 (只需要局部观察)
        
        Args:
            observation: [state_dim] - 自己的观察
            explore: 是否添加探索噪声
            noise_scale: 噪声强度
        
        Returns:
            action: [action_dim] - 选择的动作
        """
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            
            action, _ = self.actor(observation)
            
            if explore:
                noise = torch.randn_like(action) * noise_scale
                action = torch.clamp(action + noise, -1, 1)
            
            return action.squeeze(0)
    
    def update(
        self,
        local_obs: torch.Tensor,           # [batch, state_dim] 自己的观察
        global_state: torch.Tensor,        # [batch, state_dim] 全局状态
        all_actions: torch.Tensor,         # [batch, num_agents * action_dim] 所有动作
        rewards: torch.Tensor,             # [batch] 自己的奖励
        next_local_obs: torch.Tensor,      # [batch, state_dim]
        next_global_state: torch.Tensor,   # [batch, state_dim]
        next_all_actions: torch.Tensor,    # [batch, num_agents * action_dim]
        dones: torch.Tensor                # [batch]
    ) -> Tuple[float, float]:
        """
        MADDPG 训练更新
        
        Returns:
            (critic_loss, actor_loss)
        """
        # ========== 步骤 1: 更新 Critic ==========
        # 计算 target Q-value
        with torch.no_grad():
            next_q = self.target_critic(next_global_state, next_all_actions).squeeze(-1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 当前 Q-value
        current_q = self.critic(global_state, all_actions).squeeze(-1)
        
        # Critic loss (TD error)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # ========== 步骤 2: 更新 Actor ==========
        # 生成新的动作
        new_actions, _ = self.actor(local_obs)
        
        # 替换 all_actions 中自己的部分
        start_idx = self.agent_id * self.action_dim
        end_idx = start_idx + self.action_dim
        
        mixed_actions = all_actions.clone()
        mixed_actions[:, start_idx:end_idx] = new_actions
        
        # Actor loss: maximize Q(x, a₁, ..., âᵢ, ..., aₙ)
        actor_loss = -self.critic(global_state, mixed_actions).mean()
        
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # ========== 步骤 3: 软更新目标网络 ==========
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        
        # 更新统计
        self.stats['actor_loss'] = actor_loss.item()
        self.stats['critic_loss'] = critic_loss.item()
        self.stats['q_value'] = current_q.mean().item()
        self.stats['updates'] += 1
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络: θ' ← τθ + (1-τ)θ'"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'stats': self.stats
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.stats = checkpoint['stats']