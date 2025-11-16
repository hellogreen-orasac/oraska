import numpy as np
import faiss
import logging
from typing import List, Dict
from oraska.config import config
from oraska.db.database import get_db, get_redis
from oraska.db.models import MemoryEntry
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class HierarchicalMemory:
    def __init__(self):
        self.dim = config.MEMORY_DIM
        self.stm_capacity = config.STM_CAPACITY
        self.redis = get_redis()
        self.ltm_index = faiss.IndexFlatL2(self.dim)
        self.ltm_ids = []
        self._rebuild_index()
    
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
                'timestamp': time.time()
            }
            self.redis.lpush('stm_entries', json.dumps(entry_data))
            self.redis.ltrim('stm_entries', 0, self.stm_capacity - 1)
        else:
            with get_db() as db:
                entry = MemoryEntry(
                    content=content,
                    embedding=json.dumps(embedding.tolist()),
                    metadata=metadata,
                    importance=importance,
                    tier='LTM'
                )
                db.add(entry)
                db.flush()
                self.ltm_index.add(embedding.reshape(1, -1))
                self.ltm_ids.append(entry.id)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        results = []
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
                'importance': entry['importance']
            })
        if len(self.ltm_ids) > 0:
            k_ltm = min(k * 2, len(self.ltm_ids))
            distances, indices = self.ltm_index.search(query_embedding.reshape(1, -1), k_ltm)
            with get_db() as db:
                for dist, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(self.ltm_ids):
                        entry_id = self.ltm_ids[idx]
                        entry = db.query(MemoryEntry).get(entry_id)
                        if entry:
                            similarity = 1.0 / (1.0 + dist)
                            results.append({
                                'content': entry.content,
                                'metadata': entry.metadata,
                                'similarity': float(similarity),
                                'source': 'LTM',
                                'importance': entry.importance
                            })
                            entry.access_count += 1
                            entry.last_accessed = datetime.utcnow()
                            db.add(entry)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]