from __future__ import annotations

import asyncio
import time
from typing import Any

import aiosqlite

from agi.storage.db import session_get, session_upsert, session_delete, session_reap
from agi.types import SessionRecord


class SessionStore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._cache: dict[str, SessionRecord] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def get(self, key: str) -> SessionRecord | None:
        if key in self._cache:
            return self._cache[key]
        row = await session_get(self._db, key)
        if row is None:
            return None
        rec = _row_to_record(row)
        self._cache[key] = rec
        return rec

    async def get_or_create(
        self,
        key: str,
        agent_id: str,
        channel: str,
        peer_kind: str,
        peer_id: str,
        account_id: str | None = None,
        thread_id: str | None = None,
    ) -> SessionRecord:
        rec = await self.get(key)
        if rec is not None:
            return rec
        now = int(time.time() * 1000)
        rec = SessionRecord(
            session_key=key,
            agent_id=agent_id,
            channel=channel,
            peer_kind=peer_kind,
            peer_id=peer_id,
            account_id=account_id,
            thread_id=thread_id,
            history=[],
            meta={},
            created_at_ms=now,
            updated_at_ms=now,
        )
        await self.save(rec)
        return rec

    async def save(self, rec: SessionRecord) -> None:
        self._cache[rec.session_key] = rec
        await session_upsert(self._db, _record_to_row(rec))

    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)
        self._locks.pop(key, None)
        await session_delete(self._db, key)

    async def patch_meta(self, key: str, updates: dict[str, Any]) -> None:
        rec = await self.get(key)
        if rec is None:
            return
        rec.meta.update(updates)
        await self.save(rec)

    async def replace_history(self, key: str, history: list[dict]) -> None:
        rec = await self.get(key)
        if rec is None:
            return
        rec.history = history
        await self.save(rec)

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def start_reaper(self, interval_hours: int = 24, max_age_days: int = 90) -> None:
        asyncio.create_task(self._reap_loop(interval_hours, max_age_days))

    async def _reap_loop(self, interval_hours: int, max_age_days: int) -> None:
        while True:
            await asyncio.sleep(interval_hours * 3600)
            try:
                await session_reap(self._db, max_age_days)
            except Exception:
                pass


def _row_to_record(row: dict) -> SessionRecord:
    return SessionRecord(
        session_key=row["session_key"],
        agent_id=row["agent_id"],
        channel=row["channel"],
        peer_kind=row["peer_kind"],
        peer_id=row["peer_id"],
        account_id=row.get("account_id"),
        thread_id=row.get("thread_id"),
        history=row.get("history") or [],
        meta=row.get("meta") or {},
        created_at_ms=row.get("created_at_ms") or 0,
        updated_at_ms=row.get("updated_at_ms") or 0,
    )


def _record_to_row(rec: SessionRecord) -> dict:
    return {
        "session_key": rec.session_key,
        "agent_id": rec.agent_id,
        "channel": rec.channel,
        "account_id": rec.account_id,
        "peer_kind": rec.peer_kind,
        "peer_id": rec.peer_id,
        "thread_id": rec.thread_id,
        "history": rec.history,
        "meta": rec.meta,
        "created_at_ms": rec.created_at_ms,
        "updated_at_ms": rec.updated_at_ms,
    }
