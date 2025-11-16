from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from oraska.orchestrator import Orchestrator
from oraska.db.database import init_db
from oraska.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Oraska v9.2.2", version="9.2.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

orchestrator: Optional[Orchestrator] = None

class TaskRequest(BaseModel):
    id: Optional[str] = None
    description: str
    context: Optional[Dict] = {}

class NarrowAgentRequest(BaseModel):
    agent_id: str
    agent_type: str
    endpoint: Optional[str] = None
    capabilities: List[str]
    config: Dict = {}

@app.on_event("startup")
async def startup():
    global orchestrator
    logger.info("Initializing Oraska v9.2.2...")
    init_db()
    orchestrator = Orchestrator()
    logger.info("Oraska v9.2.2 ready")

@app.post("/tasks/execute")
async def execute_task(task: TaskRequest):
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    result = await orchestrator.execute_task(task.dict())
    return result

@app.get("/metrics")
async def get_metrics():
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    return orchestrator.get_stats()

@app.post("/agents/narrow/register")
async def register_narrow_agent(agent: NarrowAgentRequest):
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    from oraska.db.database import get_db
    from oraska.db.models import NarrowAgent as NarrowAgentModel
    with get_db() as db:
        db_agent = NarrowAgentModel(agent_id=agent.agent_id, agent_type=agent.agent_type, endpoint=agent.endpoint, capabilities=agent.capabilities, config=agent.config)
        db.add(db_agent)
    return {"status": "success", "agent_id": agent.agent_id}

@app.get("/agents/narrow/list")
async def list_narrow_agents():
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    return orchestrator.narrow_agents.list_agents()

@app.post("/memory/search")
async def search_memory(query: str, k: int = 5):
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    query_emb = await orchestrator.llm.embed(query)
    results = orchestrator.memory.search(query_emb, k=k)
    return {"results": results}

@app.post("/checkpoint/save")
async def save_checkpoint(path: str = "checkpoint"):
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    full_path = f"{config.CHECKPOINT_DIR}/{path}"
    orchestrator.save_checkpoint(full_path)
    return {"status": "success", "path": full_path}

@app.post("/checkpoint/load")
async def load_checkpoint(path: str):
    if not orchestrator:
        raise HTTPException(500, "Orchestrator not initialized")
    full_path = f"{config.CHECKPOINT_DIR}/{path}"
    orchestrator.load_checkpoint(full_path)
    return {"status": "success", "path": full_path}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "9.2.2", "orchestrator": orchestrator is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)