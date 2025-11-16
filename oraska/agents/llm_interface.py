import aiohttp
import asyncio
import logging
import numpy as np
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from oraska.config import config
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self):
        self.providers = self._init_providers()
        self.stats = {name: {'calls': 0, 'errors': 0} for name in self.providers}
        self._sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _init_providers(self) -> Dict:
        providers = {}
        if config.OPENAI_API_KEY:
            providers['openai'] = {
                'url': 'https://api.openai.com/v1/chat/completions',
                'headers': {'Authorization': f'Bearer {config.OPENAI_API_KEY}'},
                'model': 'gpt-4-turbo-preview'
            }
        if config.ANTHROPIC_API_KEY:
            providers['anthropic'] = {
                'url': 'https://api.anthropic.com/v1/messages',
                'headers': {
                    'x-api-key': config.ANTHROPIC_API_KEY,
                    'anthropic-version': '2023-06-01'
                },
                'model': 'claude-3-sonnet-20240229'
            }
        return providers
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, prompt: str, params: Dict, provider: str = 'openai') -> str:
        if provider not in self.providers:
            provider = list(self.providers.keys())[0] if self.providers else None
        if not provider:
            raise ValueError("No LLM providers configured")
        provider_config = self.providers[provider]
        self.stats[provider]['calls'] += 1
        try:
            if provider == 'openai':
                return await self._call_openai(prompt, params, provider_config)
            elif provider == 'anthropic':
                return await self._call_anthropic(prompt, params, provider_config)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            self.stats[provider]['errors'] += 1
            logger.error(f"LLM call failed ({provider}): {e}")
            raise
    
    async def _call_openai(self, prompt: str, params: Dict, config: Dict) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config['url'],
                headers=config['headers'],
                json={
                    'model': config['model'],
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': params.get('temperature', 0.7),
                    'top_p': params.get('top_p', 1.0),
                    'max_tokens': params.get('max_tokens', 1024)
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"OpenAI API error {resp.status}: {text}")
                data = await resp.json()
                return data['choices'][0]['message']['content']
    
    async def _call_anthropic(self, prompt: str, params: Dict, config: Dict) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config['url'],
                headers={**config['headers'], 'content-type': 'application/json'},
                json={
                    'model': config['model'],
                    'max_tokens': params.get('max_tokens', 1024),
                    'temperature': params.get('temperature', 0.7),
                    'top_p': params.get('top_p', 1.0),
                    'messages': [{'role': 'user', 'content': prompt}]
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Anthropic API error {resp.status}: {text}")
                data = await resp.json()
                return data['content'][0]['text']
    
    async def embed(self, text: str) -> np.ndarray:
        return self._sbert.encode(text, convert_to_numpy=True).astype('float32')