"""
Oraska v9.3.0 - 完整 LLMInterface
功能:
1. 分层记忆系统 (STM + LTM + 因果过滤 + 时间衰减)
2. LLM 生成接口 (OpenAI / Anthropic)
3. 兼容旧 Oraska 交互接口
"""

import numpy as np
import faiss
import logging
from typing import List, Dict, Optional
from oraska.config import config
from oraska.db.database import get_db, get_redis
from oraska.db.models import MemoryEntry
import json
import time
from datetime import datetime
import openai  # 确保已安装 openai
import anthropic  # 确保已安装 anthropic

logger = logging.getLogger(__name__)


class LLMInterface:
    def __init__(self):
        # ===== Memory System =====
        self.dim = config.MEMORY_DIM
        self.stm_capacity = config.STM_CAPACITY
        self.redis = get_redis()
        self.ltm_index = faiss.IndexFlatL2(self.dim)
        self.ltm_ids = []
        self.causal_stats = {}
        self._rebuild_index()
        logger.info(f"✅ LLMInterface initialized: {len(self.ltm_ids)} LTM entries")

        # ===== LLM API clients =====
        self.openai_key = config.OPENAI_API_KEY
        self.anthropic_key = config.ANTHROPIC_API_KEY
        if self.openai_key:
            openai.api_key = self.openai_key
        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        else:
            self.anthropic_client = None

    # ---------------- Memory Methods ----------------
    def _rebuild_index(self):
        with get_db() as db:
            entries = db.query(MemoryEntry).filter(MemoryEntry.tier == 'LTM').all()
            if entries:
                embeddings = np.array([json.loads(e.embedding) for e in entries], dtype='float32')
                self.ltm_index.add(embeddings)
                self.ltm_ids = [e.id for e in entries]
                logger.info(f"Rebuilt FAISS index with {len(entries)} LTM entries")

    def add(self, content: str, embedding: np.ndarray, metadata: Dict, importance: float = 1.0):
        stm_size = self.redis.llen('stm_entries')
        if importance > 0.7 and stm_size < self.stm_capacity:
            entry_data = {
                'content': content[:500],
                'embedding': embedding.tolist(),
                'metadata': metadata,
                'importance': importance,
                'timestamp': time.time(),
                'id': f"stm_{int(time.time() * 1000)}"
            }
            self.redis.lpush('stm_entries', json.dumps(entry_data))
            self.redis.ltrim('stm_entries', 0, self.stm_capacity - 1)
        else:
            with get_db() as db:
                entry = MemoryEntry(
                    content=content,
                    embedding=json.dumps(embedding.tolist()),
                    meta_data=metadata,
                    importance=importance,
                    tier='LTM'
                )
                db.add(entry)
                db.flush()
                self.ltm_index.add(embedding.reshape(1, -1))
                self.ltm_ids.append(entry.id)
                self.causal_stats[entry.id] = {'uses': 0, 'successes': 0}

    def search(self, query_embedding: np.ndarray, k: int = 5,
               use_causal_filter: bool = True, task_type: Optional[str] = None) -> List[Dict]:
        results = []

        # ---- STM ----
        stm_entries = self.redis.lrange('stm_entries', 0, -1)
        for entry_json in stm_entries:
            entry = json.loads(entry_json)
            emb = np.array(entry['embedding'], dtype='float32')
            similarity = 1.0 / (1.0 + np.linalg.norm(query_embedding - emb))
            results.append({
                'content': entry['content'],
                'metadata': entry['metadata'],
                'similarity': float(similarity),
                'source': 'STM',
                'importance': entry['importance'],
                'causal_score': 1.0,
                'id': entry['id']
            })

        # ---- LTM ----
        if len(self.ltm_ids) > 0:
            k_ltm = min(k * 3, len(self.ltm_ids))
            distances, indices = self.ltm_index.search(query_embedding.reshape(1, -1), k_ltm)
            with get_db() as db:
                for dist, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(self.ltm_ids):
                        entry_id = self.ltm_ids[idx]
                        entry = db.query(MemoryEntry).get(entry_id)
                        if entry:
                            similarity = 1.0 / (1.0 + dist)
                            causal_score = self._compute_causal_score(entry_id, entry.meta_data, task_type) \
                                if use_causal_filter else 1.0
                            time_decay = self._compute_time_decay(entry.created_at)
                            results.append({
                                'content': entry.content,
                                'metadata': entry.meta_data,
                                'similarity': float(similarity),
                                'source': 'LTM',
                                'importance': entry.importance,
                                'causal_score': causal_score,
                                'time_decay': time_decay,
                                'id': entry_id
                            })
                            entry.access_count += 1
                            entry.last_accessed = datetime.utcnow()
                            db.add(entry)

        # ---- 排序 ----
        if use_causal_filter:
            for r in results:
                r['final_score'] = (
                        0.5 * r['similarity'] +
                        0.3 * r.get('causal_score', 1.0) +
                        0.2 * r.get('time_decay', 1.0)
                )
            results.sort(key=lambda x: x['final_score'], reverse=True)
        else:
            results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def _compute_causal_score(self, memory_id: int, metadata: Dict, task_type: Optional[str]) -> float:
        stats = self.causal_stats.get(memory_id, {'uses': 0, 'successes': 0})
        alpha, beta = 1.0, 2.0
        base_score = (stats['successes'] + alpha) / (stats['uses'] + beta)
        type_bonus = 0.2 if task_type and metadata.get('task_type') == task_type else 0.0
        return min(base_score + type_bonus, 1.0)

    def _compute_time_decay(self, created_at: datetime) -> float:
        delta_days = (datetime.utcnow() - created_at).total_seconds() / 86400
        return np.exp(-0.01 * delta_days)

    def update_causal_feedback(self, memory_ids: List[int], task_success: bool):
        for mid in memory_ids:
            if mid not in self.causal_stats:
                self.causal_stats[mid] = {'uses': 0, 'successes': 0}
            self.causal_stats[mid]['uses'] += 1
            if task_success:
                self.causal_stats[mid]['successes'] += 1

    def get_stats(self) -> Dict:
        stm_size = self.redis.llen('stm_entries')
        return {'stm_size': stm_size, 'ltm_size': len(self.ltm_ids),
                'total': stm_size + len(self.ltm_ids), 'causal_stats_count': len(self.causal_stats)}

    # ---------------- LLM Methods ----------------
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """生成文本: 支持 OpenAI / Anthropic / 占位返回"""
        if self.openai_key:
            try:
                resp = openai.Completion.create(
                    model="gpt-4",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return resp.choices[0].text.strip()
            except Exception as e:
                logger.error(f"OpenAI generate failed: {e}")
                return "[OpenAI API error]"
        elif self.anthropic_client:
            try:
                resp = self.anthropic_client.completions.create(
                    model="claude-v1",
                    prompt=prompt,
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                return resp.completion.strip()
            except Exception as e:
                logger.error(f"Anthropic generate failed: {e}")
                return "[Anthropic API error]"
        else:
            return f"[LLM placeholder for prompt: {prompt}]"
