from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from agi.config import MemoryConfig
from agi.memory.embeddings import get_embedding
from agi.memory.hybrid import hybrid_search
from agi.storage.db import memory_insert, memory_delete
from agi.types import MemoryEntry

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


class MemoryManager:
    def __init__(self, db: aiosqlite.Connection, cfg: MemoryConfig) -> None:
        self._db = db
        self._cfg = cfg

    async def add(
        self,
        agent_id: str,
        peer_id: str,
        content: str,
        scope: str = "user",
        source_file: str | None = None,
    ) -> list[int]:
        """Chunk, embed, and store content. Returns list of entry IDs."""
        chunks = _chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        ids = []
        for chunk in chunks:
            embedding = await get_embedding(chunk, self._cfg.embedding_model)
            entry_id = await memory_insert(self._db, {
                "agent_id": agent_id,
                "peer_id": peer_id,
                "scope": scope,
                "content": chunk,
                "source_file": source_file,
                "embedding": embedding,
            })
            ids.append(entry_id)
        return ids

    async def search(
        self,
        agent_id: str,
        user_id: str,
        chat_id: str = "",
        thread_id: str = "",
        query: str = "",
        rerank_model: str | None = None,
    ) -> list[MemoryEntry]:
        return await hybrid_search(
            self._db, agent_id, user_id, chat_id, thread_id, query, self._cfg, rerank_model
        )

    async def build_context(
        self,
        agent_id: str,
        user_id: str,
        chat_id: str = "",
        thread_id: str = "",
        query: str = "",
        top_k: int | None = None,
        rerank_model: str | None = None,
    ) -> str:
        """Return formatted memory context string for injection into system prompt."""
        cfg = self._cfg
        if top_k is not None:
            from copy import deepcopy
            cfg = deepcopy(cfg)
            cfg.top_k = top_k

        entries = await hybrid_search(self._db, agent_id, user_id, chat_id, thread_id, query, cfg, rerank_model)
        if not entries:
            return ""

        lines = []
        for e in entries:
            snippet = e.content.strip()[:500]
            lines.append(f"- {snippet}")
        return "\n".join(lines)

    async def delete(self, entry_id: int) -> None:
        await memory_delete(self._db, entry_id)

    async def delete_by_query(
        self,
        agent_id: str,
        user_id: str,
        chat_id: str = "",
        thread_id: str = "",
        query: str = "",
    ) -> int:
        """Delete entries matching query. Returns count deleted.

        Only deletes from writable scopes (user, chat, thread, agent_memory).
        Never touches global, agent_kb, or user_kb.
        """
        _DELETABLE = {"user", "chat", "thread", "agent_memory"}
        entries = await self.search(agent_id, user_id, chat_id, thread_id, query)
        count = 0
        for e in entries:
            if e.scope in _DELETABLE:
                await memory_delete(self._db, e.id)
                count += 1
        return count


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at paragraph
        para_break = text.rfind("\n\n", start, end)
        if para_break > start + size // 2:
            end = para_break
        else:
            # Try sentence boundary
            for sep in (". ", "! ", "? ", "\n"):
                pos = text.rfind(sep, start + size // 2, end)
                if pos > start:
                    end = pos + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c.strip()]
