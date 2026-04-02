from __future__ import annotations

import math
import struct
import time
import logging
from typing import Any

import aiosqlite

from agi.config import MemoryConfig
from agi.memory.embeddings import get_embedding, cosine_similarity
from agi.memory.mmr import mmr_rerank
from agi.memory.query import expand_query_for_fts
from agi.storage.db import memory_fts_search, _decode_floats
from agi.types import MemoryEntry

logger = logging.getLogger(__name__)


async def hybrid_search(
    db: aiosqlite.Connection,
    agent_id: str,
    user_id: str,
    chat_id: str,
    thread_id: str,
    query: str,
    cfg: MemoryConfig,
    rerank_model: str | None = None,
) -> list[MemoryEntry]:
    limit_inner = min(cfg.top_k * 4, 60)

    # 1. Sparse (BM25 via FTS5)
    fts_query = expand_query_for_fts(query)
    fts_rows = await memory_fts_search(db, agent_id, user_id, chat_id, thread_id, fts_query, limit=limit_inner)
    fts_by_id: dict[int, float] = {}
    if fts_rows:
        raw = [-float(r["bm25_score"]) for r in fts_rows]
        max_s = max(raw) if raw else 1.0
        for row, s in zip(fts_rows, raw):
            fts_by_id[int(row["id"])] = s / max_s if max_s else 0.0

    # 2. Dense (sqlite-vec cosine)
    vec_by_id: dict[int, float] = {}
    query_vec = await get_embedding(query, cfg.embedding_model)
    if query_vec:
        pairs = await _dense_search(db, agent_id, user_id, chat_id, thread_id, query_vec, limit_inner)
        for eid, sim in pairs:
            vec_by_id[eid] = sim

    # 3. RRF fusion
    all_ids = set(fts_by_id) | set(vec_by_id)
    if not all_ids:
        return []

    scored: list[tuple[int, float]] = []
    for eid in all_ids:
        score = (
            cfg.vector_weight * vec_by_id.get(eid, 0.0)
            + cfg.text_weight * fts_by_id.get(eid, 0.0)
        )
        scored.append((eid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [eid for eid, _ in scored[:limit_inner]]

    # 4. Fetch full rows
    ph = ",".join("?" * len(top_ids))
    async with db.execute(
        f"SELECT * FROM memory_entries WHERE id IN ({ph})", top_ids
    ) as cur:
        rows = await cur.fetchall()

    id_to_score = dict(scored)
    now_ms = int(time.time() * 1000)
    entries: list[MemoryEntry] = []

    for row in rows:
        r = dict(row)
        score = id_to_score.get(r["id"], 0.0)
        # 5. Temporal decay
        score = _temporal_decay(score, r["updated_at_ms"], now_ms, cfg.half_life_days)
        emb = _decode_floats(r["embedding"]) if r.get("embedding") else None
        entries.append(MemoryEntry(
            id=r["id"],
            agent_id=r.get("agent_id", ""),
            peer_id=r.get("peer_id", ""),
            scope=r.get("scope", "agent"),
            content=r["content"],
            source_file=r.get("source_file"),
            embedding=emb,
            score=score,
            created_at_ms=r["created_at_ms"],
            updated_at_ms=r["updated_at_ms"],
        ))

    entries.sort(key=lambda e: e.score, reverse=True)

    # 6. Rerank
    if cfg.reranker == "llm" and rerank_model:
        from agi.memory.rerank import llm_rerank
        entries = await llm_rerank(entries, query, rerank_model, cfg.top_k)
    elif cfg.reranker == "mmr":
        entries = mmr_rerank(entries, cfg.top_k, cfg.mmr_lambda)
    else:
        entries = entries[:cfg.top_k]

    return entries


async def _dense_search(
    db: aiosqlite.Connection,
    agent_id: str,
    user_id: str,
    chat_id: str,
    thread_id: str,
    query_vec: list[float],
    limit: int,
) -> list[tuple[int, float]]:
    try:
        blob = struct.pack(f"{len(query_vec)}f", *query_vec)
        sql = """
            SELECT e.id, vec_distance_cosine(e.embedding, ?) AS dist
            FROM memory_entries e
            WHERE (
                e.scope = 'global'
                OR (e.scope IN ('agent_kb', 'agent_memory') AND e.agent_id = ?)
                OR (e.scope IN ('user_kb', 'user') AND e.peer_id = ?)
                OR (e.scope = 'chat' AND e.peer_id = ?)
                OR (e.scope = 'thread' AND e.peer_id = ? AND ? != '')
            ) AND e.embedding IS NOT NULL
            ORDER BY dist
            LIMIT ?
        """
        async with db.execute(sql, (blob, agent_id, user_id, chat_id, thread_id, thread_id, limit)) as cur:
            rows = await cur.fetchall()
        return [(r[0], 1.0 - float(r[1])) for r in rows]
    except Exception:
        return []


def _temporal_decay(score: float, updated_at_ms: int, now_ms: int, half_life_days: float) -> float:
    if half_life_days <= 0:
        return score
    age_days = max(0.0, (now_ms - updated_at_ms) / 86_400_000)
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    return score * decay
