"""
Oraska v9.3.0 - 分层记忆系统 (增强版)

新增功能:
1. ✅ 因果相关性过滤
2. ✅ 基于任务类型的记忆聚类
3. ✅ 动态重要性衰减
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
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HierarchicalMemory:
    """
    双层记忆架构:
    - STM (Short-Term Memory): Redis 缓存, 50条高优先级
    - LTM (Long-Term Memory): PostgreSQL + FAISS, 50,000条持久化
    
    新增因果过滤:
    - 基于历史成功率的因果推断
    - 任务类型聚类
    - 时间衰减
    """
    def __init__(self):
        self.dim = config.MEMORY_DIM
        self.stm_capacity = config.STM_CAPACITY
        self.redis = get_redis()
        
        # FAISS 索引
        self.ltm_index = faiss.IndexFlatL2(self.dim)
        self.ltm_ids = []
        
        # 因果统计 (记忆ID → 使用该记忆后的成功率)
        self.causal_stats = {}  # {memory_id: {'uses': int, 'successes': int}}
        
        self._rebuild_index()
        logger.info(f"✅ Hierarchical Memory initialized: {len(self.ltm_ids)} LTM entries")
    
    def _rebuild_index(self):
        """重建 FAISS 索引"""
        with get_db() as db:
            entries = db.query(MemoryEntry).filter(MemoryEntry.tier == 'LTM').all()
            if entries:
                embeddings = np.array([json.loads(e.embedding) for e in entries], dtype='float32')
                self.ltm_index.add(embeddings)
                self.ltm_ids = [e.id for e in entries]
                logger.info(f"Rebuilt FAISS index with {len(entries)} LTM entries")
    
    def add(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Dict,
        importance: float = 1.0
    ):
        """
        添加记忆
        
        策略:
        - importance > 0.7 → STM (快速访问)
        - 否则 → LTM (长期存储)
        """
        stm_size = self.redis.llen('stm_entries')
        
        if importance > 0.7 and stm_size < self.stm_capacity:
            # 添加到 STM
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
            # 添加到 LTM
            with get_db() as db:
                entry = MemoryEntry(
                    content=content,
                    embedding=json.dumps(embedding.tolist()),
                    meta_data=metadata,  # 修正: 使用 meta_data 而非 metadata
                    importance=importance,
                    tier='LTM'
                )
                db.add(entry)
                db.flush()
                
                # 更新 FAISS 索引
                self.ltm_index.add(embedding.reshape(1, -1))
                self.ltm_ids.append(entry.id)
                
                # 初始化因果统计
                self.causal_stats[entry.id] = {'uses': 0, 'successes': 0}
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        use_causal_filter: bool = True,
        task_type: Optional[str] = None
    ) -> List[Dict]:
        """
        智能记忆检索
        
        步骤:
        1. 向量相似度检索 (召回 top-k*3)
        2. 因果相关性过滤
        3. 时间衰减调整
        4. 重新排序返回 top-k
        
        Args:
            query_embedding: 查询向量
            k: 返回数量
            use_causal_filter: 是否使用因果过滤
            task_type: 任务类型 (用于聚类)
        
        Returns:
            排序后的记忆列表
        """
        results = []
        
        # ===== 阶段 1: STM 检索 =====
        stm_entries = self.redis.lrange('stm_entries', 0, -1)
        for entry_json in stm_entries:
            entry = json.loads(entry_json)
            emb = np.array(entry['embedding'], dtype='float32')
            
            # 向量相似度
            similarity = 1.0 / (1.0 + np.linalg.norm(query_embedding - emb))
            
            results.append({
                'content': entry['content'],
                'metadata': entry['metadata'],
                'similarity': float(similarity),
                'source': 'STM',
                'importance': entry['importance'],
                'causal_score': 1.0,  # STM 默认高因果分
                'id': entry['id']
            })
        
        # ===== 阶段 2: LTM 检索 =====
        if len(self.ltm_ids) > 0:
            # 召回 top-k*3 候选
            k_ltm = min(k * 3, len(self.ltm_ids))
            distances, indices = self.ltm_index.search(query_embedding.reshape(1, -1), k_ltm)
            
            with get_db() as db:
                for dist, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(self.ltm_ids):
                        entry_id = self.ltm_ids[idx]
                        entry = db.query(MemoryEntry).get(entry_id)
                        
                        if entry:
                            # 向量相似度
                            similarity = 1.0 / (1.0 + dist)
                            
                            # 因果相关性
                            causal_score = self._compute_causal_score(
                                entry_id,
                                entry.meta_data,
                                task_type
                            ) if use_causal_filter else 1.0
                            
                            # 时间衰减
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
                            
                            # 更新访问统计
                            entry.access_count += 1
                            entry.last_accessed = datetime.utcnow()
                            db.add(entry)
        
        # ===== 阶段 3: 重新排序 =====
        if use_causal_filter:
            # 综合评分: semantic + causal + time
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
    
    def _compute_causal_score(
        self,
        memory_id: int,
        metadata: Dict,
        task_type: Optional[str]
    ) -> float:
        """
        计算因果相关性分数
        
        基于历史数据: P(success | use_this_memory)
        
        公式:
        causal_score = (successes + α) / (uses + β)
        其中 α=1, β=2 (拉普拉斯平滑)
        """
        stats = self.causal_stats.get(memory_id, {'uses': 0, 'successes': 0})
        
        # 拉普拉斯平滑
        alpha, beta = 1.0, 2.0
        base_score = (stats['successes'] + alpha) / (stats['uses'] + beta)
        
        # 任务类型匹配加成
        if task_type and metadata.get('task_type') == task_type:
            type_bonus = 0.2
        else:
            type_bonus = 0.0
        
        return min(base_score + type_bonus, 1.0)
    
    def _compute_time_decay(self, created_at: datetime) -> float:
        """
        时间衰减因子
        
        公式: decay = exp(-λ * Δt)
        其中 λ = 0.01 (天^-1), Δt 是天数
        """
        now = datetime.utcnow()
        delta_days = (now - created_at).total_seconds() / 86400
        lambda_decay = 0.01
        return np.exp(-lambda_decay * delta_days)
    
    def update_causal_feedback(
        self,
        memory_ids: List[int],
        task_success: bool
    ):
        """
        更新因果统计
        
        当任务完成后,记录使用的记忆是否带来成功
        
        Args:
            memory_ids: 本次使用的记忆 ID 列表
            task_success: 任务是否成功
        """
        for mid in memory_ids:
            if mid not in self.causal_stats:
                self.causal_stats[mid] = {'uses': 0, 'successes': 0}
            
            self.causal_stats[mid]['uses'] += 1
            if task_success:
                self.causal_stats[mid]['successes'] += 1
        
        logger.debug(f"Updated causal stats for {len(memory_ids)} memories")
    
    def get_stats(self) -> Dict:
        """获取记忆系统统计"""
        stm_size = self.redis.llen('stm_entries')
        ltm_size = len(self.ltm_ids)
        
        return {
            'stm_size': stm_size,
            'ltm_size': ltm_size,
            'total': stm_size + ltm_size,
            'causal_stats_count': len(self.causal_stats)
        }