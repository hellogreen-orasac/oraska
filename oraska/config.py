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

config = Config()