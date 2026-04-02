from __future__ import annotations

"""Sync markdown memory files into SQLite — seven-scope hierarchy.

Directory layout under config_dir/memory/:
  global/                      → scope='global'        (all agents, all users)
  agents/{agent_id}/kb/        → scope='agent_kb'      (agent knowledge base, manual)
  agents/{agent_id}/memory/    → scope='agent_memory'  (agent runtime memory, auto)
  users/{peer_id}/kb/          → scope='user_kb'       (user personal KB, manual)
  users/{peer_id}/flush/summary/ → scope='user'        (auto-generated conversation summaries)
  users/{peer_id}/flush/prefs/   → scope='user'        (auto-detected user preferences)
  chats/{chat_id}/             → scope='chat'          (chat-space shared context)
  threads/{thread_id}/         → scope='thread'        (topic/task-specific context)

Note: agents/{agent_id}/AGENT.md and agents/{agent_id}/skills/ are NOT indexed here —
AGENT.md is injected directly into the system prompt; skills are loaded by SkillManager.

LLM writes memories to markdown files → we detect changes and re-index
into memory_entries + memory_fts for hybrid search.
"""

import logging
import time
from pathlib import Path

import aiosqlite

from agi.config import MemoryConfig
from agi.memory.embeddings import get_embedding
from agi.storage.db import memory_insert, memory_delete

logger = logging.getLogger(__name__)


async def sync_memory_root(
    db: aiosqlite.Connection,
    memory_root: Path,
    agents: list,
    cfg: MemoryConfig,
) -> int:
    """Sync all seven scopes. Returns total count of files synced."""
    total = 0

    # 1. Global knowledge base
    global_dir = memory_root / "global"
    if global_dir.exists():
        total += await _sync_scope_dir(
            db, global_dir, scope="global", agent_id="", peer_id="", cfg=cfg
        )

    # 2. Per-agent KB + memory (AGENT.md and skills/ are not indexed)
    agents_dir = memory_root / "agents"
    for agent in agents:
        agent_dir = agents_dir / agent.id
        kb_dir = agent_dir / "kb"
        if kb_dir.exists():
            total += await _sync_scope_dir(
                db, kb_dir, scope="agent_kb", agent_id=agent.id, peer_id="", cfg=cfg
            )
        mem_dir = agent_dir / "memory"
        if mem_dir.exists():
            total += await _sync_scope_dir(
                db, mem_dir, scope="agent_memory", agent_id=agent.id, peer_id="", cfg=cfg
            )

    # 3. Per-user KB (manual) + flush (auto)
    users_dir = memory_root / "users"
    if users_dir.exists():
        for peer_dir in sorted(users_dir.iterdir()):
            if not peer_dir.is_dir():
                continue
            peer_id = peer_dir.name
            kb_dir = peer_dir / "kb"
            if kb_dir.exists():
                total += await _sync_scope_dir(
                    db, kb_dir, scope="user_kb", agent_id="", peer_id=peer_id, cfg=cfg
                )
            flush_dir = peer_dir / "flush"
            if flush_dir.exists():
                total += await _sync_scope_dir(
                    db, flush_dir, scope="user", agent_id="", peer_id=peer_id, cfg=cfg
                )

    # 4. Per-chat context
    chats_dir = memory_root / "chats"
    if chats_dir.exists():
        for chat_dir in sorted(chats_dir.iterdir()):
            if chat_dir.is_dir():
                total += await _sync_scope_dir(
                    db, chat_dir, scope="chat", agent_id="", peer_id=chat_dir.name, cfg=cfg
                )

    # 5. Per-thread context
    threads_dir = memory_root / "threads"
    if threads_dir.exists():
        for thread_dir in sorted(threads_dir.iterdir()):
            if thread_dir.is_dir():
                total += await _sync_scope_dir(
                    db, thread_dir, scope="thread", agent_id="", peer_id=thread_dir.name, cfg=cfg
                )

    if total:
        logger.info("Memory sync: indexed %d changed file(s) total", total)
    return total


async def sync_user_flush(
    db: aiosqlite.Connection,
    memory_root: Path,
    peer_id: str,
    cfg: MemoryConfig,
) -> int:
    """Targeted sync of one user's flush dir after a memory flush."""
    flush_dir = memory_root / "users" / peer_id / "flush"
    return await _sync_scope_dir(
        db, flush_dir, scope="user", agent_id="", peer_id=peer_id, cfg=cfg
    )


async def _sync_scope_dir(
    db: aiosqlite.Connection,
    directory: Path,
    scope: str,
    agent_id: str,
    peer_id: str,
    cfg: MemoryConfig,
) -> int:
    """Sync all .md files in a directory with the given scope/agent_id/peer_id."""
    if not directory.exists() or not directory.is_dir():
        return 0

    candidates = sorted(directory.glob("**/*.md"))
    synced = 0
    for path in candidates:
        try:
            mtime_ms = int(path.stat().st_mtime * 1000)
            stored = await _get_index_entry(db, str(path))
            if stored and stored["mtime_ms"] == mtime_ms:
                continue
            await _reindex_file(db, agent_id, peer_id, scope, path, mtime_ms, cfg)
            synced += 1
        except Exception as e:
            logger.warning("Failed to sync memory file %s: %s", path, e)

    if synced:
        logger.info(
            "Memory sync: indexed %d file(s) [scope=%s agent=%s peer=%s]",
            synced, scope, agent_id or "-", peer_id or "-",
        )
    return synced


async def _reindex_file(
    db: aiosqlite.Connection,
    agent_id: str,
    peer_id: str,
    scope: str,
    path: Path,
    mtime_ms: int,
    cfg: MemoryConfig,
) -> None:
    """Delete old chunks for this file and re-index."""
    path_str = str(path)

    # Delete existing chunks from this file
    async with db.execute(
        "SELECT id FROM memory_entries WHERE source_file=? AND scope=? AND agent_id=? AND peer_id=?",
        (path_str, scope, agent_id, peer_id)
    ) as cur:
        rows = await cur.fetchall()
    for row in rows:
        await memory_delete(db, row[0])

    # Read and chunk the file
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning("Cannot read memory file %s: %s", path, e)
        return

    if not content:
        await _upsert_index_entry(db, path_str, agent_id, peer_id, scope, mtime_ms)
        return

    chunks = _chunk_markdown(content)
    for chunk in chunks:
        embedding = await get_embedding(chunk, cfg.embedding_model)
        await memory_insert(db, {
            "agent_id": agent_id,
            "peer_id": peer_id,
            "scope": scope,
            "content": chunk,
            "source_file": path_str,
            "embedding": embedding,
        })

    await _upsert_index_entry(db, path_str, agent_id, peer_id, scope, mtime_ms)
    logger.debug("Indexed %d chunks from %s [scope=%s]", len(chunks), path.name, scope)


def _chunk_markdown(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split markdown text into chunks, preferring section boundaries."""
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

        # Prefer markdown heading boundary
        heading = _rfind_heading(text, start + size // 2, end)
        if heading > start:
            end = heading
        else:
            # Paragraph boundary
            para = text.rfind("\n\n", start, end)
            if para > start + size // 2:
                end = para
            else:
                for sep in (". ", "! ", "? ", "\n"):
                    pos = text.rfind(sep, start + size // 2, end)
                    if pos > start:
                        end = pos + len(sep)
                        break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c.strip()]


def _rfind_heading(text: str, lo: int, hi: int) -> int:
    """Find the last markdown heading (# ...) in text[lo:hi]."""
    pos = hi
    while pos > lo:
        pos = text.rfind("\n#", lo, pos)
        if pos == -1:
            return -1
        return pos + 1
    return -1


async def _get_index_entry(db: aiosqlite.Connection, path: str) -> dict | None:
    async with db.execute(
        "SELECT path, agent_id, peer_id, scope, mtime_ms, indexed_at_ms FROM memory_file_index WHERE path=?",
        (path,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def _upsert_index_entry(
    db: aiosqlite.Connection,
    path: str,
    agent_id: str,
    peer_id: str,
    scope: str,
    mtime_ms: int,
) -> None:
    now = int(time.time() * 1000)
    await db.execute("""
        INSERT INTO memory_file_index (path, agent_id, peer_id, scope, mtime_ms, indexed_at_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            agent_id=excluded.agent_id,
            peer_id=excluded.peer_id,
            scope=excluded.scope,
            mtime_ms=excluded.mtime_ms,
            indexed_at_ms=excluded.indexed_at_ms
    """, (path, agent_id, peer_id, scope, mtime_ms, now))
    await db.commit()
