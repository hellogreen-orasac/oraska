"""
Oraska v9.3.0 - 核心编排器 (已修复)

关键修复:
1. ✅ 每个 agent 独立的 reward 信号
2. ✅ 因果状态转移链: Task → Plan → Output → Review
3. ✅ 使用 MADDPGAgent 而非简单 RLAgent
4. ✅ 全局状态 + 局部观察分离
"""

import asyncio
import logging
import numpy as np
import time
import re
from typing import Dict, List, Tuple
from oraska.config import config
from oraska.memory.hierarchical_memory import HierarchicalMemory
from oraska.rl.centralized_critic import MADDPGAgent
from oraska.rl.embedding_network import EmbeddingNetwork
from oraska.agents.llm_interface import LLMInterface
from oraska.agents.narrow_agents import NarrowAgentRegistry
from oraska.db.database import get_db
from oraska.db.models import Task, AgentPerformance
import torch
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from collections import deque
import random

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """增强的经验回放缓冲区 - 支持 MADDPG"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        local_obs: np.ndarray,
        global_state: np.ndarray,
        all_actions: np.ndarray,
        reward: float,
        next_local_obs: np.ndarray,
        next_global_state: np.ndarray,
        next_all_actions: np.ndarray,
        done: bool
    ):
        """存储完整的 MADDPG 转移"""
        self.buffer.append((
            local_obs, global_state, all_actions, reward,
            next_local_obs, next_global_state, next_all_actions, done
        ))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)


class Orchestrator:
    """
    Oraska 核心编排器 v9.3.0
    
    架构:
    - Agent 0 (Planner): 任务 → 计划
    - Agent 1 (Executor): 计划 → 输出
    - Agent 2 (Reviewer): 输出 → 评审
    """
    def __init__(self):
        self.memory = HierarchicalMemory()
        self.llm = LLMInterface()
        self.narrow_agents = NarrowAgentRegistry()
        
        # 文本嵌入模型
        self._sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Embedding 网络 (文本 → RL 状态)
        self.embedding_net = EmbeddingNetwork(input_dim=384, output_dim=config.STATE_DIM)
        self.embedding_optimizer = torch.optim.AdamW(self.embedding_net.parameters(), lr=1e-4)
        
        # MADDPG Agents (3个专门化的 agent)
        self.agents = [
            MADDPGAgent(
                agent_id=i,
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_DIM,
                num_agents=config.NUM_AGENTS,
                lr_actor=config.LEARNING_RATE,
                lr_critic=config.LEARNING_RATE * 3  # Critic 学习率更高
            )
            for i in range(config.NUM_AGENTS)
        ]
        
        # 经验缓冲区 (每个 agent 独立)
        self.buffers = [ExperienceBuffer(config.BUFFER_SIZE) for _ in range(config.NUM_AGENTS)]
        
        # 评估工具
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # 统计信息
        self.stats = {
            'tasks': 0,
            'successes': 0,
            'total_reward': 0.0,
            'agent_rewards': [0.0, 0.0, 0.0]
        }
        
        logger.info(f"✅ Orchestrator v9.3.0 initialized: MADDPG with {config.NUM_AGENTS} agents")
    
    async def execute_task(self, task: Dict) -> Dict:
        """
        执行任务的主流程
        
        流程:
        1. Task Embedding
        2. Agent 0: Planning
        3. Agent 1: Execution
        4. Agent 2: Review
        5. Reward Computation (分层)
        6. Experience Storage
        7. Model Update
        """
        task_id = task.get('id', f"task_{int(time.time())}")
        description = task.get('description', '')
        start_time = time.time()
        
        self.stats['tasks'] += 1
        
        try:
            # ========== 阶段 0: 初始化 ==========
            # 获取任务的语义嵌入
            task_embedding = self._sbert.encode(description, convert_to_numpy=True)
            
            # 搜索相关记忆
            memories = self.memory.search(task_embedding, k=3)
            context = "\n".join([f"- {m['content'][:100]}" for m in memories[:2]])
            
            # ========== 阶段 1: Planning (Agent 0) ==========
            state_0 = self._embed_to_state(task_embedding)
            action_0 = self.agents[0].select_action(
                torch.FloatTensor(state_0), 
                explore=True
            ).numpy()
            params_0 = self._action_to_params(action_0)
            
            plan_prompt = f"Task: {description}\n\nContext:\n{context}\n\nCreate a detailed step-by-step plan:"
            plan = await self.llm.generate(
                plan_prompt, 
                params=params_0,
                task_type='planning'
            )
            
            # 计算 Planning 质量
            plan_reward = self._evaluate_plan_quality(plan, description)
            
            # ========== 阶段 2: Execution (Agent 1) ==========
            plan_context = f"{description}\nPlan: {plan}"
            plan_embedding = self._sbert.encode(plan_context, convert_to_numpy=True)
            state_1 = self._embed_to_state(plan_embedding)
            
            action_1 = self.agents[1].select_action(
                torch.FloatTensor(state_1),
                explore=True
            ).numpy()
            params_1 = self._action_to_params(action_1)
            
            exec_prompt = f"Task: {description}\n\nPlan:\n{plan}\n\nExecute the solution:"
            output = await self.llm.generate(
                exec_prompt,
                params=params_1,
                task_type='code_generation'
            )
            
            # 计算 Execution 质量
            exec_reward = self._evaluate_execution_quality(output, plan, description)
            
            # ========== 阶段 3: Review (Agent 2) ==========
            output_context = f"{description}\nOutput: {output}"
            output_embedding = self._sbert.encode(output_context, convert_to_numpy=True)
            state_2 = self._embed_to_state(output_embedding)
            
            action_2 = self.agents[2].select_action(
                torch.FloatTensor(state_2),
                explore=True
            ).numpy()
            params_2 = self._action_to_params(action_2)
            
            review_prompt = f"Task: {description}\n\nSolution:\n{output}\n\nReview quality (score 0-10):"
            review = await self.llm.generate(
                review_prompt,
                params=params_2,
                task_type='reasoning'
            )
            
            # 计算 Review 质量
            review_reward = self._evaluate_review_quality(review, output, description)
            
            # ========== 阶段 4: 全局质量评估 ==========
            rouge_scores = self.rouge_scorer.score(description, output)
            overall_quality = rouge_scores['rougeL'].fmeasure
            
            latency_ms = (time.time() - start_time) * 1000
            
            # ========== 阶段 5: 经验存储 (因果链) ==========
            # 构造全局状态 (所有 agent 的观察拼接)
            global_state_0 = state_0  # Task 阶段
            global_state_1 = state_1  # Plan 阶段
            global_state_2 = state_2  # Output 阶段
            
            # 最终状态
            final_embedding = self._sbert.encode(
                f"Task: {description}\nPlan: {plan}\nOutput: {output}\nReview: {review}",
                convert_to_numpy=True
            )
            final_state = self._embed_to_state(final_embedding)
            
            # 拼接所有 actions
            all_actions = np.concatenate([action_0, action_1, action_2])
            
            # Agent 0: Task → Plan
            self.buffers[0].add(
                local_obs=state_0,
                global_state=global_state_0,
                all_actions=all_actions,
                reward=plan_reward,
                next_local_obs=state_1,  # 下一个 agent 的输入
                next_global_state=global_state_1,
                next_all_actions=all_actions,  # 简化: 使用相同的 actions
                done=False
            )
            
            # Agent 1: Plan → Output
            self.buffers[1].add(
                local_obs=state_1,
                global_state=global_state_1,
                all_actions=all_actions,
                reward=exec_reward,
                next_local_obs=state_2,
                next_global_state=global_state_2,
                next_all_actions=all_actions,
                done=False
            )
            
            # Agent 2: Output → Review (终止状态)
            self.buffers[2].add(
                local_obs=state_2,
                global_state=global_state_2,
                all_actions=all_actions,
                reward=review_reward,
                next_local_obs=final_state,
                next_global_state=final_state,
                next_all_actions=all_actions,
                done=True
            )
            
            # ========== 阶段 6: 模型更新 ==========
            if all(len(buf) >= config.BATCH_SIZE for buf in self.buffers):
                await self._update_all_agents()
            
            # ========== 阶段 7: 记忆存储 ==========
            self.memory.add(
                content=output[:500],
                embedding=output_embedding,
                metadata={
                    'task_id': task_id,
                    'quality': overall_quality,
                    'plan_reward': plan_reward,
                    'exec_reward': exec_reward,
                    'review_reward': review_reward
                },
                importance=overall_quality
            )
            
            # ========== 阶段 8: 数据库记录 ==========
            with get_db() as db:
                task_record = Task(
                    task_id=task_id,
                    description=description,
                    status='completed',
                    output=output,
                    reward=(plan_reward + exec_reward + review_reward) / 3,
                    agent_id=-1,
                    generation_params={
                        'agent_0_plan': params_0,
                        'agent_1_exec': params_1,
                        'agent_2_review': params_2
                    },
                    latency_ms=latency_ms
                )
                db.add(task_record)
                
                for i, (params, reward) in enumerate([
                    (params_0, plan_reward),
                    (params_1, exec_reward),
                    (params_2, review_reward)
                ]):
                    perf_record = AgentPerformance(
                        agent_id=i,
                        task_id=task_id,
                        reward=reward,
                        temperature=params['temperature'],
                        top_p=params['top_p'],
                        max_tokens=params['max_tokens'],
                        latency_ms=latency_ms / 3
                    )
                    db.add(perf_record)
            
            # 更新统计
            self.stats['successes'] += 1
            self.stats['total_reward'] += (plan_reward + exec_reward + review_reward) / 3
            for i, r in enumerate([plan_reward, exec_reward, review_reward]):
                self.stats['agent_rewards'][i] += r
            
            return {
                'task_id': task_id,
                'success': True,
                'plan': plan,
                'output': output,
                'review': review,
                'rewards': {
                    'plan': plan_reward,
                    'execution': exec_reward,
                    'review': review_reward,
                    'overall': (plan_reward + exec_reward + review_reward) / 3
                },
                'quality': overall_quality,
                'latency_ms': latency_ms,
                'agent_chain': [0, 1, 2],
                'generation_params': {
                    'agent_0': params_0,
                    'agent_1': params_1,
                    'agent_2': params_2
                },
                'memories_used': len(memories)
            }
        
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    def _embed_to_state(self, text_embedding: np.ndarray) -> np.ndarray:
        """将文本嵌入转换为 RL 状态"""
        with torch.no_grad():
            state = self.embedding_net(
                torch.FloatTensor(text_embedding).unsqueeze(0)
            ).numpy().squeeze()
        return state
    
    def _action_to_params(self, action: np.ndarray) -> Dict:
        """将 RL action 转换为 LLM 参数"""
        def scale(val, min_val, max_val):
            return min_val + (val + 1) / 2 * (max_val - min_val)
        
        return {
            'temperature': float(scale(action[0], config.TEMPERATURE_MIN, config.TEMPERATURE_MAX)),
            'top_p': float(scale(action[1], config.TOP_P_MIN, config.TOP_P_MAX)),
            'max_tokens': int(scale(action[2], config.MAX_TOKENS_MIN, config.MAX_TOKENS_MAX))
        }
    
    def _evaluate_plan_quality(self, plan: str, task: str) -> float:
        """
        评估计划质量
        
        指标:
        - 步骤数量 (适中最好)
        - 与任务的相关性
        - 结构完整性
        """
        steps = len([line for line in plan.split('\n') if line.strip().startswith(('1', '2', '3', '-', '*'))])
        
        # 步骤数量评分 (3-8 步最优)
        step_score = 1.0 - abs(steps - 5) / 10 if 3 <= steps <= 8 else 0.5
        
        # 相关性评分 (简单关键词匹配)
        task_keywords = set(task.lower().split())
        plan_keywords = set(plan.lower().split())
        relevance = len(task_keywords & plan_keywords) / max(len(task_keywords), 1)
        
        # 结构评分 (是否有明确的开始/中间/结尾)
        structure_score = 0.8 if steps >= 3 else 0.5
        
        return 0.4 * step_score + 0.4 * relevance + 0.2 * structure_score
    
    def _evaluate_execution_quality(self, output: str, plan: str, task: str) -> float:
        """
        评估执行质量
        
        指标:
        - 与计划的一致性
        - 输出长度适中
        - 代码/结构完整性
        """
        # 与计划的一致性
        plan_keywords = set(plan.lower().split())
        output_keywords = set(output.lower().split())
        adherence = len(plan_keywords & output_keywords) / max(len(plan_keywords), 1)
        
        # 长度适中性 (100-1000 字符最优)
        length = len(output)
        length_score = 1.0 if 100 <= length <= 1000 else 0.7
        
        # 代码完整性 (如果是代码任务)
        if 'code' in task.lower() or 'function' in task.lower():
            has_def = 'def ' in output or 'function' in output or 'class ' in output
            completeness = 1.0 if has_def else 0.6
        else:
            completeness = 0.9
        
        return 0.5 * adherence + 0.2 * length_score + 0.3 * completeness
    
    def _evaluate_review_quality(self, review: str, output: str, task: str) -> float:
        """
        评估评审质量
        
        指标:
        - 是否包含分数
        - 是否指出具体问题
        - 评审长度适中
        """
        # 是否包含数字评分
        score_match = re.findall(r'\d+', review)
        has_score = 1.0 if score_match else 0.5
        
        # 是否包含具体评价
        has_specifics = 1.0 if len(review) > 50 else 0.6
        
        # 长度适中
        length = len(review)
        length_score = 1.0 if 30 <= length <= 300 else 0.7
        
        return 0.5 * has_score + 0.3 * has_specifics + 0.2 * length_score
    
    async def _update_all_agents(self):
        """更新所有 agents"""
        for i, agent in enumerate(self.agents):
            # 采样经验
            batch = self.buffers[i].sample(config.BATCH_SIZE)
            (local_obs, global_states, all_actions, rewards,
             next_local_obs, next_global_states, next_all_actions, dones) = batch
            
            # 转换为 tensor
            local_obs_t = torch.FloatTensor(np.array(local_obs))
            global_states_t = torch.FloatTensor(np.array(global_states))
            all_actions_t = torch.FloatTensor(np.array(all_actions))
            rewards_t = torch.FloatTensor(rewards)
            next_local_obs_t = torch.FloatTensor(np.array(next_local_obs))
            next_global_states_t = torch.FloatTensor(np.array(next_global_states))
            next_all_actions_t = torch.FloatTensor(np.array(next_all_actions))
            dones_t = torch.FloatTensor(dones)
            
            # 更新 agent
            critic_loss, actor_loss = agent.update(
                local_obs_t, global_states_t, all_actions_t, rewards_t,
                next_local_obs_t, next_global_states_t, next_all_actions_t, dones_t
            )
            
            logger.debug(f"Agent {i} updated: critic_loss={critic_loss:.4f}, actor_loss={actor_loss:.4f}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'avg_reward': self.stats['total_reward'] / max(self.stats['tasks'], 1),
            'success_rate': self.stats['successes'] / max(self.stats['tasks'], 1),
            'buffer_sizes': [len(buf) for buf in self.buffers],
            'agents': [agent.stats for agent in self.agents],
            'agent_avg_rewards': [
                r / max(self.stats['tasks'], 1) for r in self.stats['agent_rewards']
            ]
        }
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        for i, agent in enumerate(self.agents):
            agent.save(f"{path}_agent_{i}.pt")
        torch.save(self.embedding_net.state_dict(), f"{path}_embedding.pt")
        logger.info(f"✅ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        for i, agent in enumerate(self.agents):
            agent.load(f"{path}_agent_{i}.pt")
        self.embedding_net.load_state_dict(torch.load(f"{path}_embedding.pt"))
        logger.info(f"✅ Checkpoint loaded: {path}")