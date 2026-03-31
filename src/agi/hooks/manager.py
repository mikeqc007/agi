from __future__ import annotations

"""Hook system — event-driven extensibility, mirroring openclaw's internal-hooks.

Event keys:  type (e.g. "message")  or  type:action (e.g. "message:received")

Event types and actions:
  message : received | sent
  agent   : start | end
  tool    : call | result
  gateway : startup | shutdown

Drop Python files in ~/.agi/hooks/ — loaded at startup.
Each file calls register_hook() to subscribe.

Example (~/.agi/hooks/logger.py):
    from agi.hooks.manager import register_hook, HookEvent

    async def on_msg(event: HookEvent) -> None:
        print(f"[hook] {event.type}:{event.action}  session={event.session_key}")
        print(f"       context={event.context}")

    register_hook("message", on_msg)
"""

import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Handler type: sync or async callable
HookHandler = Callable[["HookEvent"], "Coroutine[Any, Any, None] | None"]


class HookEvent:
    """Event object passed to every hook handler."""

    __slots__ = ("type", "action", "session_key", "context", "messages", "timestamp")

    def __init__(
        self,
        type: str,
        action: str,
        session_key: str = "",
        context: dict | None = None,
    ) -> None:
        import time
        self.type = type
        self.action = action
        self.session_key = session_key
        self.context: dict = context or {}
        self.messages: list[str] = []   # handlers can append user-facing strings
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return f"HookEvent({self.type!r}:{self.action!r} session={self.session_key!r})"


# ------------------------------------------------------------------
# Global registry  key → [handler, ...]
# ------------------------------------------------------------------

_handlers: dict[str, list[HookHandler]] = {}


def register_hook(event_key: str, handler: HookHandler) -> None:
    """Register handler for event_key (e.g. "message" or "message:received")."""
    _handlers.setdefault(event_key, []).append(handler)


def unregister_hook(event_key: str, handler: HookHandler) -> None:
    bucket = _handlers.get(event_key, [])
    if handler in bucket:
        bucket.remove(handler)
    if not bucket:
        _handlers.pop(event_key, None)


def clear_hooks() -> None:
    _handlers.clear()


def registered_keys() -> list[str]:
    return list(_handlers.keys())


# ------------------------------------------------------------------
# Trigger
# ------------------------------------------------------------------

async def trigger_hook(event: HookEvent) -> None:
    """Fire all handlers for event.type and event.type:event.action (in order)."""
    type_handlers = list(_handlers.get(event.type, []))
    specific_handlers = list(_handlers.get(f"{event.type}:{event.action}", []))

    for handler in type_handlers + specific_handlers:
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(
                "Hook error [%s:%s]: %s", event.type, event.action, e
            )


def make_event(
    type: str,
    action: str,
    session_key: str = "",
    **context: Any,
) -> HookEvent:
    """Convenience constructor."""
    return HookEvent(type=type, action=action, session_key=session_key, context=dict(context))


# ------------------------------------------------------------------
# HookManager — loads scripts from disk
# ------------------------------------------------------------------

class HookManager:
    """Loads .py hook files from directory and manages lifecycle events."""

    def __init__(self, hooks_dir: Path | None = None) -> None:
        self._dir = hooks_dir or (Path.home() / ".agi" / "hooks")
        self._loaded: list[str] = []

    def load_all(self) -> None:
        """Import all *.py files from hooks directory."""
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.py")):
            self._load_file(path)

    def _load_file(self, path: Path) -> None:
        try:
            spec = importlib.util.spec_from_file_location(f"_hook_{path.stem}", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                self._loaded.append(path.name)
                logger.info("Hook loaded: %s", path.name)
        except Exception as e:
            logger.error("Hook load failed (%s): %s", path.name, e)

    async def startup(self, app: Any) -> None:
        self.load_all()
        await trigger_hook(make_event("gateway", "startup", app=app))

    async def shutdown(self) -> None:
        await trigger_hook(make_event("gateway", "shutdown"))

    @property
    def loaded(self) -> list[str]:
        return list(self._loaded)
