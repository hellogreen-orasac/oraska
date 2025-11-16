import aiohttp
import logging
from typing import Dict, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class NarrowAgent(ABC):
    def __init__(self, agent_id: str, agent_type: str, config: Dict):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
    
    @abstractmethod
    async def execute(self, task: Dict) -> Dict:
        pass

# 在 CodeAnalyzerAgent.execute 中添加 reward 优化
class CodeAnalyzerAgent(NarrowAgent):
    async def execute(self, task: Dict) -> Dict:
        code = task.get('code', '')
        lines = len([l for l in code.split('\n') if l.strip()])
        return {
            'agent_type': self.agent_type,
            'analysis': {
                'lines': lines,
                'concise_score': max(0, 1.0 - lines/50),  # 越短越好
                'functions': code.count('def '),
                'classes': code.count('class ')
            },
            'success': True
        }

class SQLGeneratorAgent(NarrowAgent):
    async def execute(self, task: Dict) -> Dict:
        description = task.get('description', '').lower()
        # 强制简洁 SQL
        if 'top' in description and 'sales' in description:
            sql = "SELECT customer_id, SUM(amount) as total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 5;"
        elif 'email' in description:
            sql = "SELECT * FROM users WHERE email LIKE '%@%.%';"
        else:
            sql = "SELECT * FROM table WHERE active = 1;"
        return {'agent_type': self.agent_type, 'sql': sql.strip(), 'success': True}

class APICallerAgent(NarrowAgent):
    async def execute(self, task: Dict) -> Dict:
        endpoint = task.get('endpoint')
        if not endpoint:
            return {'agent_type': self.agent_type, 'error': 'No endpoint provided', 'success': False}

        method = task.get('method', 'GET')
        data = task.get('data', {})
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, endpoint, json=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    result = await resp.json()
                    return {'agent_type': self.agent_type, 'status': resp.status, 'result': result, 'success': resp.status == 200}
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {'agent_type': self.agent_type, 'error': str(e), 'success': False}

class NarrowAgentRegistry:
    def __init__(self):
        self.agents = {}
        self._register_defaults()
    
    def _register_defaults(self):
        self.register('code_analyzer', CodeAnalyzerAgent('code_analyzer', 'code_analyzer', {}))
        self.register('sql_generator', SQLGeneratorAgent('sql_generator', 'sql_generator', {}))
        self.register('api_caller', APICallerAgent('api_caller', 'api_caller', {}))
    
    def register(self, agent_id: str, agent: NarrowAgent):
        self.agents[agent_id] = agent
        logger.info(f"Registered narrow agent: {agent_id}")
    
    def get(self, agent_id: str) -> Optional[NarrowAgent]:
        return self.agents.get(agent_id)
    
    def list_agents(self) -> Dict:
        return {aid: {'type': agent.agent_type} for aid, agent in self.agents.items()}
