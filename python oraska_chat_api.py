# 保存为 oraska_chat_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from oraska.orchestrator import Orchestrator
from oraska.config import config
from oraska.db.database import init_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Oraska Chat API", version="9.2.2")
orchestrator: Optional[Orchestrator] = None

class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = {}

@app.on_event("startup")
async def startup():
    global orchestrator
    logger.info("Initializing Oraska with chat support...")
    init_db()
    orchestrator = Orchestrator()
    logger.info("Oraska ready for chat")

@app.post("/chat")
async def chat(req: ChatRequest):
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    
    try:
        # 使用 orchestrator 执行任务模拟聊天
        # 如果 Orchestrator 有 llm.chat 或类似方法可以调用
        if hasattr(orchestrator, 'llm') and hasattr(orchestrator.llm, 'chat'):
            response = await orchestrator.llm.chat(req.message, context=req.context)
        else:
            # fallback：用 execute_task 包装输入
            task = {
                "description": req.message,
                "context": req.context
            }
            response = await orchestrator.execute_task(task)
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(500, f"Chat failed: {e}")

@app.get("/health")
async def health():
    return {"status": "healthy", "orchestrator_initialized": orchestrator is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)
