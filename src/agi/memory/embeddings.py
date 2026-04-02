from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import lru_cache
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_cache: dict[str, list[float]] = {}
_CACHE_MAX = 2048


async def get_embedding(text: str, model: str) -> list[float] | None:
    """Get embedding vector. Returns None if embedding service unavailable."""
    key = hashlib.md5(f"{model}:{text}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]

    try:
        vec = await _fetch_embedding(text, model)
    except Exception as e:
        logger.debug("Embedding failed for model %s: %s", model, e)
        return None

    if vec:
        if len(_cache) >= _CACHE_MAX:
            # Evict oldest entry
            _cache.pop(next(iter(_cache)))
        _cache[key] = vec
    return vec


async def _fetch_embedding(text: str, model: str) -> list[float]:
    # model format: "ollama/nomic-embed-text" or "openai/text-embedding-3-small"
    provider, _, model_name = model.partition("/")

    if provider == "ollama":
        return await _ollama_embed(text, model_name)
    elif provider in ("openai", "azure"):
        return await _openai_embed(text, model_name)
    else:
        # Try litellm as fallback
        return await _litellm_embed(text, model)


async def _ollama_embed(text: str, model: str) -> list[float]:
    import os
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    return data["embedding"]


async def _openai_embed(text: str, model: str) -> list[float]:
    import os
    import litellm
    response = await litellm.aembedding(model=f"openai/{model}", input=[text])
    return response.data[0]["embedding"]


async def _litellm_embed(text: str, model: str) -> list[float]:
    import litellm
    response = await litellm.aembedding(model=model, input=[text])
    return response.data[0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
