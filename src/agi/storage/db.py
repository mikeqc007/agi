from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_key     TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    channel         TEXT NOT NULL,
    account_id      TEXT,
    peer_kind       TEXT NOT NULL,
    peer_id         TEXT NOT NULL,
    thread_id       TEXT,
    history         TEXT NOT NULL DEFAULT '[]',
    meta            TEXT NOT NULL DEFAULT '{}',
    created_at_ms   INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT NOT NULL DEFAULT '',
    peer_id         TEXT NOT NULL DEFAULT '',
    scope           TEXT NOT NULL DEFAULT 'agent',
    content         TEXT NOT NULL,
    source_file     TEXT,
    embedding       BLOB,
    created_at_ms   INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    agent_id UNINDEXED,
    content='memory_entries',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT NOT NULL,
    session_key     TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at_ms   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cron_jobs (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    peer_kind       TEXT NOT NULL,
    peer_id         TEXT NOT NULL,
    schedule        TEXT NOT NULL,
    task            TEXT NOT NULL,
    output          TEXT NOT NULL DEFAULT '',
    notify          TEXT NOT NULL DEFAULT '',
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at_ms   INTEGER NOT NULL,
    last_run_ms     INTEGER
);

CREATE TABLE IF NOT EXISTS memory_file_index (
    path            TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL DEFAULT '',
    peer_id         TEXT NOT NULL DEFAULT '',
    scope           TEXT NOT NULL DEFAULT 'agent',
    mtime_ms        INTEGER NOT NULL,
    indexed_at_ms   INTEGER NOT NULL
);
"""


async def open_db(path: str | Path) -> aiosqlite.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.executescript(SCHEMA)
    await db.commit()
    # Migrations for existing DBs (column additions are idempotent via try/except)
    for col_sql in [
        "ALTER TABLE cron_jobs ADD COLUMN notify TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE cron_jobs ADD COLUMN output TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE memory_entries ADD COLUMN peer_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE memory_entries ADD COLUMN scope TEXT NOT NULL DEFAULT 'agent'",
        "ALTER TABLE memory_file_index ADD COLUMN peer_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE memory_file_index ADD COLUMN scope TEXT NOT NULL DEFAULT 'agent'",
    ]:
        try:
            await db.execute(col_sql)
            await db.commit()
        except Exception:
            pass  # column already exists
    await _load_sqlite_vec(db)
    return db


async def _load_sqlite_vec(db: aiosqlite.Connection) -> None:
    try:
        import sqlite_vec
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
    except Exception:
        pass  # sqlite-vec optional; dense search disabled


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

async def session_get(db: aiosqlite.Connection, key: str) -> dict | None:
    async with db.execute("SELECT * FROM sessions WHERE session_key=?", (key,)) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    return _session_row(row)


async def session_upsert(db: aiosqlite.Connection, rec: dict) -> None:
    await db.execute("""
        INSERT INTO sessions
            (session_key, agent_id, channel, account_id, peer_kind, peer_id,
             thread_id, history, meta, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(session_key) DO UPDATE SET
            history=excluded.history,
            meta=excluded.meta,
            updated_at_ms=excluded.updated_at_ms
    """, (
        rec["session_key"], rec["agent_id"], rec["channel"],
        rec.get("account_id"), rec["peer_kind"], rec["peer_id"],
        rec.get("thread_id"),
        json.dumps(rec.get("history") or [], ensure_ascii=False),
        json.dumps(rec.get("meta") or {}, ensure_ascii=False),
        rec.get("created_at_ms") or int(time.time() * 1000),
        int(time.time() * 1000),
    ))
    await db.commit()


async def session_delete(db: aiosqlite.Connection, key: str) -> None:
    await db.execute("DELETE FROM sessions WHERE session_key=?", (key,))
    await db.commit()


async def session_reap(db: aiosqlite.Connection, max_age_days: int,
                       cron_max_age_days: int = 1) -> int:
    now = time.time()
    cutoff = int((now - max_age_days * 86400) * 1000)
    cron_cutoff = int((now - cron_max_age_days * 86400) * 1000)
    async with db.execute(
        "DELETE FROM sessions WHERE (channel != 'cron' AND updated_at_ms < ?) "
        "OR (channel = 'cron' AND updated_at_ms < ?)",
        (cutoff, cron_cutoff)
    ) as cur:
        n = cur.rowcount
    await db.commit()
    return n


def _session_row(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["history"] = json.loads(d["history"])
    d["meta"] = json.loads(d["meta"])
    return d


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

async def memory_insert(db: aiosqlite.Connection, entry: dict) -> int:
    now = int(time.time() * 1000)
    emb_blob = _encode_floats(entry["embedding"]) if entry.get("embedding") else None
    scope = entry.get("scope", "agent")
    peer_id = entry.get("peer_id", "")
    agent_id = entry.get("agent_id", "")
    async with db.execute("""
        INSERT INTO memory_entries (agent_id, peer_id, scope, content, source_file, embedding, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?,?,?)
    """, (agent_id, peer_id, scope, entry["content"], entry.get("source_file"),
          emb_blob, now, now)) as cur:
        row_id = cur.lastrowid
    await db.execute(
        "INSERT INTO memory_fts(rowid, content, agent_id) VALUES (?,?,?)",
        (row_id, entry["content"], agent_id)
    )
    await db.commit()
    return row_id


async def memory_delete(db: aiosqlite.Connection, entry_id: int) -> None:
    await db.execute("DELETE FROM memory_entries WHERE id=?", (entry_id,))
    await db.execute("DELETE FROM memory_fts WHERE rowid=?", (entry_id,))
    await db.commit()


async def memory_fts_search(
    db: aiosqlite.Connection,
    agent_id: str,
    user_id: str,
    chat_id: str,
    thread_id: str,
    query: str,
    limit: int = 30,
) -> list[dict]:
    try:
        sql = """
            SELECT e.id, e.agent_id, e.peer_id, e.scope, e.content, e.source_file,
                   e.embedding, e.created_at_ms, e.updated_at_ms,
                   bm25(memory_fts) AS bm25_score
            FROM memory_fts
            JOIN memory_entries e ON e.id = memory_fts.rowid
            WHERE memory_fts MATCH ?
              AND (
                e.scope = 'global'
                OR (e.scope IN ('agent_kb', 'agent_memory') AND e.agent_id = ?)
                OR (e.scope IN ('user_kb', 'user') AND e.peer_id = ?)
                OR (e.scope = 'chat' AND e.peer_id = ?)
                OR (e.scope = 'thread' AND e.peer_id = ? AND ? != '')
              )
            ORDER BY bm25_score
            LIMIT ?
        """
        async with db.execute(sql, (query, agent_id, user_id, chat_id, thread_id, thread_id, limit)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


async def memory_get_by_ids(db: aiosqlite.Connection, ids: list[int]) -> list[dict]:
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT * FROM memory_entries WHERE id IN ({ph})", ids
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

async def usage_record(
    db: aiosqlite.Connection,
    agent_id: str,
    session_key: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    await db.execute("""
        INSERT INTO usage (agent_id, session_key, model, prompt_tokens, completion_tokens, created_at_ms)
        VALUES (?,?,?,?,?,?)
    """, (agent_id, session_key, model, prompt_tokens, completion_tokens,
          int(time.time() * 1000)))
    await db.commit()


# ---------------------------------------------------------------------------
# Cron
# ---------------------------------------------------------------------------

async def cron_list(db: aiosqlite.Connection) -> list[dict]:
    async with db.execute("SELECT * FROM cron_jobs ORDER BY created_at_ms") as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def cron_upsert(db: aiosqlite.Connection, job: dict) -> None:
    await db.execute("""
        INSERT INTO cron_jobs (id, agent_id, peer_kind, peer_id, schedule, task, output, notify, enabled, created_at_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            schedule=excluded.schedule, task=excluded.task,
            output=excluded.output, notify=excluded.notify, enabled=excluded.enabled
    """, (job["id"], job["agent_id"], job["peer_kind"], job["peer_id"],
          job["schedule"], job["task"],
          job.get("output") or "", job.get("notify") or "",
          int(job.get("enabled", 1)),
          job.get("created_at_ms") or int(time.time() * 1000)))
    await db.commit()


async def cron_delete(db: aiosqlite.Connection, job_id: str) -> None:
    await db.execute("DELETE FROM cron_jobs WHERE id=?", (job_id,))
    await db.commit()


async def cron_touch(db: aiosqlite.Connection, job_id: str) -> None:
    await db.execute(
        "UPDATE cron_jobs SET last_run_ms=? WHERE id=?",
        (int(time.time() * 1000), job_id)
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_floats(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_floats(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))
