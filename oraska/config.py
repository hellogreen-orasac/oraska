from pydantic_settings import BaseSettings
from typing import Optional

class Config(BaseSettings):
    DATABASE_URL: str = "postgresql://oraska:oraska_secret@localhost:5432/oraska"
    REDIS_URL: str = "redis://localhost:6379"
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    NUM_AGENTS: int = 3
    STATE_DIM: int = 256
    ACTION_DIM: int = 32
    LEARNING_RATE: float = 3e-4
    GAMMA: float = 0.95
    TAU: float = 0.01
    BUFFER_SIZE: int = 10000
    BATCH_SIZE: int = 32
    MEMORY_DIM: int = 384
    STM_CAPACITY: int = 50
    LTM_CAPACITY: int = 50000
    TEMPERATURE_MIN: float = 0.0
    TEMPERATURE_MAX: float = 1.5
    TOP_P_MIN: float = 0.5
    TOP_P_MAX: float = 1.0
    MAX_TOKENS_MIN: int = 256
    MAX_TOKENS_MAX: int = 2048
    API_PORT: int = 8000
    CHECKPOINT_DIR: str = "checkpoints"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

class Config(BaseSettings):
    # ... 现有配置 ...
    
    # 新增: RL 训练超参数
    CRITIC_LR_MULTIPLIER: float = 3.0  # Critic 学习率是 Actor 的3倍
    EXPLORATION_NOISE: float = 0.1      # 探索噪声
    
    # 新增: 记忆系统配置
    CAUSAL_FILTER_ENABLED: bool = True  # 启用因果过滤
    TIME_DECAY_LAMBDA: float = 0.01     # 时间衰减系数
    
    # 新增: 模型路由配置
    ENABLE_MODEL_ENSEMBLE: bool = False  # 是否启用模型集成
    UCB_EXPLORATION_COEF: float = 1.0    # UCB 探索系数