from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from oraska.config import config
from oraska.db.models import Base
import redis
import logging

logger = logging.getLogger(__name__)

engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")

@contextmanager
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_redis():
    return redis_client