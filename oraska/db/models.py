from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)
    task_id = Column(String(100), unique=True, index=True)
    description = Column(Text)
    context = Column(JSON)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    output = Column(Text, nullable=True)
    reward = Column(Float, nullable=True)
    agent_id = Column(Integer, nullable=True)
    generation_params = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=True)

class MemoryEntry(Base):
    __tablename__ = 'memory_entries'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(JSON)
    meta_data = Column(JSON)  # 改名，原来叫 metadata
    importance = Column(Float, default=1.0)
    tier = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)

class AgentPerformance(Base):
    __tablename__ = 'agent_performance'
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, index=True)
    task_id = Column(String(100), index=True)
    reward = Column(Float)
    temperature = Column(Float)
    top_p = Column(Float)
    max_tokens = Column(Integer)
    latency_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class NarrowAgent(Base):
    __tablename__ = 'narrow_agents'
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(50), unique=True)
    agent_type = Column(String(50))
    endpoint = Column(String(500), nullable=True)
    capabilities = Column(JSON)
    config = Column(JSON)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
