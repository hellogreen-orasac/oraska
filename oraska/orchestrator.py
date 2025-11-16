import asyncio
import logging
import numpy as np
import time
from typing import Dict, List
from oraska.config import config
from oraska.memory.hierarchical_memory import HierarchicalMemory
from oraska.rl.rl_agent import RLAgent, ExperienceBuffer
from oraska.rl.embedding_network import EmbeddingNetwork
from oraska.agents.llm_interface import LLMInterface
from oraska.agents.narrow_agents import NarrowAgentRegistry
from oraska.db.database import get_db
from oraska.db.models import Task, AgentPerformance
import torch
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import re

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.memory = HierarchicalMemory()
        self.llm = LLMInterface()
        self.narrow_agents = NarrowAgentRegistry()
        self._sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_net = EmbeddingNetwork(input_dim=384, output_dim=config.STATE_DIM)
        self.embedding_optimizer = torch.optim.AdamW(self.embedding_net.parameters(), lr=1e-4)
        self.rl_agents = [RLAgent(i, config.STATE_DIM, config.ACTION_DIM) for i in range(config.NUM_AGENTS)]
        self.experience_buffer = ExperienceBuffer(config.BUFFER_SIZE)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.stats = {'tasks': 0, 'successes': 0, 'total_reward': 0.0}
        logger.info(f"Orchestrator v9.2.2 initialized: {config.NUM_AGENTS} RL agents, REAL embeddings")

    async def execute_task(self, task: Dict) -> Dict:
        task_id = task.get('id') if task and isinstance(task, dict) else f"task_{int(time.time())}"
        description = task.get('description', '') if task and isinstance(task, dict) else ''
        start_time = time.time()
        self.stats['tasks'] += 1

        try:
            # Step 0: initial embedding and memory search
            real_embedding = self._sbert.encode(description, convert_to_numpy=True)
            initial_state = self.embedding_net(torch.FloatTensor(real_embedding).unsqueeze(0)).detach().numpy().squeeze()
            query_emb = real_embedding
            memories = self.memory.search(query_emb, k=3)
            context = "\n".join([f"- {m['content'][:100]}" for m in memories[:2]])

            # Step 1: Planning
            state_0 = initial_state
            action_0 = self.rl_agents[0].select_action(state_0, explore=True)
            params_0 = self.rl_agents[0].action_to_params(action_0)
            plan_prompt = f"Task: {description}\n\nContext:\n{context}\n\nCreate a step-by-step plan:"
            plan = await self.llm.generate(plan_prompt, params=params_0,
                                           provider='openai' if 'openai' in self.llm.providers else list(self.llm.providers.keys())[0])
            plan_context = f"{description}\nPlan: {plan}"
            plan_embedding = self._sbert.encode(plan_context, convert_to_numpy=True)
            
            # Step 2: Execution
            state_1 = self.embedding_net(torch.FloatTensor(plan_embedding).unsqueeze(0)).detach().numpy().squeeze()
            action_1 = self.rl_agents[1].select_action(state_1, explore=True)
            params_1 = self.rl_agents[1].action_to_params(action_1)
            exec_prompt = f"Task: {description}\n\nPlan:\n{plan}\n\nExecute the solution:"
            output = await self.llm.generate(exec_prompt, params=params_1,
                                             provider='openai' if 'openai' in self.llm.providers else list(self.llm.providers.keys())[0])

            # Step 3: Review
            review_context = f"{description}\nOutput: {output}"
            review_embedding = self._sbert.encode(review_context, convert_to_numpy=True)
            state_2 = self.embedding_net(torch.FloatTensor(review_embedding).unsqueeze(0)).detach().numpy().squeeze()
            action_2 = self.rl_agents[2].select_action(state_2, explore=True)
            params_2 = self.rl_agents[2].action_to_params(action_2)
            review_prompt = f"Task: {description}\n\nSolution:\n{output}\n\nReview quality (score 0-10):"
            review = await self.llm.generate(review_prompt, params=params_2,
                                             provider='openai' if 'openai' in self.llm.providers else list(self.llm.providers.keys())[0])

            # Compute quality using rouge and review score
            rouge_scores = self.rouge_scorer.score(description, output)
            quality = rouge_scores['rougeL'].fmeasure
            try:
                review_match = re.findall(r'\d+', review)
                if review_match:
                    review_score = float(review_match[0]) / 10.0
                    quality = (quality + review_score) / 2
            except:
                pass

            latency_ms = (time.time() - start_time) * 1000
            reward = self._compute_reward(quality, latency_ms, len(memories))

            # Update memory
            self.memory.add(content=output[:500], embedding=query_emb, metadata={'task_id': task_id, 'quality': quality}, importance=quality)

            # Update experience buffer
            final_embedding = self._sbert.encode(output, convert_to_numpy=True)
            done = True
            self.experience_buffer.add(real_embedding, action_0, reward, final_embedding, done)
            self.experience_buffer.add(plan_embedding, action_1, reward, final_embedding, done)
            self.experience_buffer.add(review_embedding, action_2, reward, final_embedding, done)

            if len(self.experience_buffer) >= config.BATCH_SIZE:
                await self._update_agents()

            # Save to DB
            with get_db() as db:
                task_record = Task(
                    task_id=task_id,
                    description=description,
                    status='completed',
                    output=output,
                    reward=reward,
                    agent_id=-1,
                    generation_params={'agent_0_plan': params_0, 'agent_1_exec': params_1, 'agent_2_review': params_2},
                    latency_ms=latency_ms
                )
                db.add(task_record)
                for i, (params, action) in enumerate([(params_0, action_0), (params_1, action_1), (params_2, action_2)]):
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

            self.stats['successes'] += 1
            self.stats['total_reward'] += reward

            return {
                'task_id': task_id,
                'success': True,
                'plan': plan,
                'output': output,
                'review': review,
                'reward': reward,
                'quality': quality,
                'latency_ms': latency_ms,
                'agent_chain': [0, 1, 2],
                'generation_params': {'agent_0': params_0, 'agent_1': params_1, 'agent_2': params_2},
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

    def _compute_reward(self, quality: float, latency_ms: float, memories_used: int) -> float:
        r_quality = quality
        r_latency = np.exp(-latency_ms / 15000)
        r_memory = min(memories_used / 3.0, 1.0)
        return 0.5 * r_quality + 0.3 * r_latency + 0.2 * r_memory

    async def _update_agents(self):
        raw_states, actions, rewards, raw_next_states, dones = self.experience_buffer.sample(config.BATCH_SIZE)
        text_features = torch.stack([torch.FloatTensor(s) for s in raw_states])
        next_text_features = torch.stack([torch.FloatTensor(s) for s in raw_next_states])

        self.embedding_net.update(text_features, torch.FloatTensor(rewards), self.embedding_optimizer)

        with torch.no_grad():
            states = self.embedding_net(text_features).numpy()
            next_states = self.embedding_net(next_text_features).numpy()

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        for agent in self.rl_agents:
            agent.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'avg_reward': self.stats['total_reward'] / max(self.stats['tasks'], 1),
            'success_rate': self.stats['successes'] / max(self.stats['tasks'], 1),
            'buffer_size': len(self.experience_buffer),
            'agents': [agent.stats for agent in self.rl_agents]
        }

    def save_checkpoint(self, path: str):
        for i, agent in enumerate(self.rl_agents):
            agent.save(f"{path}_agent_{i}.pt")
        torch.save(self.embedding_net.state_dict(), f"{path}_embedding.pt")
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        for i, agent in enumerate(self.rl_agents):
            agent.load(f"{path}_agent_{i}.pt")
        self.embedding_net.load_state_dict(torch.load(f"{path}_embedding.pt"))
        logger.info(f"Checkpoint loaded: {path}")
