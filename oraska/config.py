"""
ORASKA v9.3.0 Configuration - Docker 修复版
移除所有 World Model 相关配置
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Config(BaseSettings):
    # ========== 数据库配置 ==========
    DATABASE_URL: str = "postgresql://oraska:oraska_secret@postgres:5432/oraska"
    REDIS_URL: str = "redis://redis:6379"
    
    # ========== API Keys ==========
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # ========== RL Agent 配置 ==========
    NUM_AGENTS: int = 3
    STATE_DIM: int = 256
    ACTION_DIM: int = 32
    LEARNING_RATE: float = 3e-4
    GAMMA: float = 0.95
    TAU: float = 0.01
    BUFFER_SIZE: int = 10000
    BATCH_SIZE: int = 32
    
    # ========== 记忆系统配置 ==========
    MEMORY_DIM: int = 384
    STM_CAPACITY: int = 50
    LTM_CAPACITY: int = 50000
    
    # ========== LLM 生成参数范围 ==========
    TEMPERATURE_MIN: float = 0.0
    TEMPERATURE_MAX: float = 1.5
    TOP_P_MIN: float = 0.5
    TOP_P_MAX: float = 1.0
    MAX_TOKENS_MIN: int = 256
    MAX_TOKENS_MAX: int = 2048
    
    # ========== API 服务配置 ==========
    API_PORT: int = 8000
    CHECKPOINT_DIR: str = "checkpoints"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

config = Config()