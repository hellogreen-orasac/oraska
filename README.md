# Oraska v9.2.2 - All Critical Bugs Fixed

## Critical Fixes Applied

1. **Embedding net trains on real tasks** - Uses SBERT embeddings, not random noise
2. **next_state computed from output** - Proper TD learning
3. **Each agent gets different state** - Context-aware specialization
4. **Actions from agent-specific states** - True multi-agent coordination
5. **Experience buffer stores raw embeddings** - Semantic learning preserved

## Quick Start

bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start services
docker-compose up -d

# 3. Wait for initialization (30 seconds)
sleep 30

# 4. Run test task
curl -X POST http://localhost:8000/tasks/execute \
  -H \"Content-Type: application/json\" \
  -d '{\"description\": \"Design JWT authentication API\"}'

# 5. Check metrics
curl http://localhost:8000/metrics | jq


## Expected Learning Curve

- **Tasks 1-20**: Random exploration, reward ~0.4
- **Tasks 50-100**: Specialization emerging, reward ~0.6
- **Tasks 200+**: Converged, reward >0.75
  - Agent 0 (Planner): temp 0.75-0.85
  - Agent 1 (Executor): temp 0.25-0.35
  - Agent 2 (Reviewer): temp 0.95-1.15

## Architecture


Orchestrator
├── Agent 0 (Planner) → high temperature
├── Agent 1 (Executor) → low temperature
└── Agent 2 (Reviewer) → high temperature

Memory
├── STM (Redis cache)
└── LTM (PostgreSQL + FAISS)

RL Loop
├── Real SBERT embeddings
├── Agent-specific states
└── ROUGE-L quality rewards


## Validation

bash
# Run 100 tasks
for i in {1..100}; do
  curl -X POST http://localhost:8000/tasks/execute \
    -H \"Content-Type: application/json\" \
    -d '{\"description\": \"Design login API with JWT\"}'
  sleep 2
done

# Check results
curl http://localhost:8000/metrics | jq '{
  tasks: .tasks,
  avg_reward: .avg_reward,
  success_rate: .success_rate,
  agents: .agents
}'


## Success Criteria

- Avg reward > 0.75 after 100 tasks
- Agent temperatures diverge (0.3, 0.8, 1.0)
- Embedding loss < 0.1
- Memory relevance > 80%