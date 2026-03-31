from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class _Item:
    message: Any
    future: asyncio.Future


@dataclass
class _SessionQueue:
    items: list[_Item] = field(default_factory=list)
    processing: bool = False
    active_msg_id: str | None = None  # id of message currently being processed


class MessageQueue:
    """Per-session message queue with drop and collect modes.

    drop:    only the newest message is kept while processing
    collect: accumulate up to cap messages, flush together
    """

    def __init__(self, process_fn: Callable, mode: str = "drop", cap: int = 5) -> None:
        self._process_fn = process_fn
        self._mode = mode
        self._cap = cap
        self._queues: dict[str, _SessionQueue] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def configure(self, session_meta: dict) -> tuple[str, int]:
        mode = str(session_meta.get("queue_mode") or self._mode)
        if mode not in {"drop", "collect"}:
            mode = self._mode
        cap = int(session_meta.get("queue_cap") or self._cap)
        return mode, max(1, cap)

    async def enqueue(self, key: str, msg: Any, session_meta: dict | None = None) -> str:
        meta = session_meta or {}
        mode, cap = self.configure(meta)

        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        q = self._queues.setdefault(key, _SessionQueue())

        # Deduplicate by msg.id — also check the currently-processing message
        msg_id = str(getattr(msg, "id", None))
        if msg_id and msg_id != "-1" and msg_id != "None":
            in_queue = any(str(getattr(item.message, "id", None)) == msg_id for item in q.items)
            if in_queue or q.active_msg_id == msg_id:
                fut.set_result("(duplicate)")
                return await fut

        item = _Item(message=msg, future=fut)

        if mode == "drop":
            # Cancel existing pending items
            for old in q.items:
                if not old.future.done():
                    old.future.set_result("(dropped)")
            q.items = [item]
        else:
            q.items.append(item)
            if len(q.items) > cap:
                dropped = q.items[:-cap]
                q.items = q.items[-cap:]
                for d in dropped:
                    if not d.future.done():
                        d.future.set_result("(dropped)")

        if not q.processing:
            asyncio.create_task(self._drain(key))

        return await fut

    async def _drain(self, key: str) -> None:
        q = self._queues.get(key)
        if not q:
            return
        q.processing = True
        async with self._lock(key):
            try:
                while True:
                    q = self._queues.get(key)
                    if not q or not q.items:
                        break
                    item = q.items.pop(0)
                    q.active_msg_id = str(getattr(item.message, "id", None))
                    try:
                        result = await self._process_fn(item.message)
                        if not item.future.done():
                            item.future.set_result(result or "")
                    except Exception as e:
                        logger.exception("Queue drain error for %s", key)
                        if not item.future.done():
                            item.future.set_exception(e)
                    finally:
                        q.active_msg_id = None
            finally:
                q = self._queues.get(key)
                if q:
                    q.processing = False
