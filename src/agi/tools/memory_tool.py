from __future__ import annotations

import datetime
from pathlib import Path

from agi.tools.registry import tool
from agi.types import ToolContext


def _state_root(ctx: ToolContext) -> Path | None:
    app = ctx.app
    if not app:
        return None
    cfg = getattr(app, "cfg", None)
    if not cfg or not cfg.config_dir:
        return None
    return Path(cfg.config_dir) / cfg.memory.memory_dir


def _append_to_file(path: Path, content: str) -> None:
    """Append a bullet entry to a markdown file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    entry = f"- [{date_str}] {content.strip()}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


@tool
async def remember(ctx: ToolContext, content: str, scope: str = "user") -> str:
    """Save information to long-term memory.

    content: The information to remember
    scope: Where to store it — 'user' (cross-chat preferences/facts), 'chat' (this chat's context), 'thread' (current topic/task), 'agent_memory' (shared across all users of this agent)
    """
    try:
        if scope == "chat":
            entity_id = ctx.peer_id
        elif scope == "thread":
            if not ctx.thread_id:
                scope = "chat"
                entity_id = ctx.peer_id
            else:
                entity_id = ctx.thread_id
        elif scope == "agent_memory":
            entity_id = ""
        else:
            scope = "user"
            entity_id = ctx.sender or ctx.peer_id

        # Write to SQLite
        ids = await ctx.app.memory_manager.add(ctx.agent_id, entity_id, content, scope=scope)

        # Also write to corresponding md file
        root = _state_root(ctx)
        if root:
            md_path: Path | None = None
            safe = lambda s: s.replace("/", "_").replace("..", "_") if s else "default"
            if scope == "user":
                md_path = root / "users" / safe(entity_id) / "kb" / "manual.md"
            elif scope == "agent_memory":
                md_path = root / "agents" / ctx.agent_id / "kb" / "manual.md"
            elif scope == "chat":
                md_path = root / "chats" / safe(entity_id) / "manual.md"
            elif scope == "thread":
                md_path = root / "threads" / safe(entity_id) / "manual.md"
            if md_path:
                try:
                    _append_to_file(md_path, content)
                except Exception:
                    pass  # md write failure is non-fatal

        return f"Saved to memory ({len(ids)} chunk(s))."
    except Exception as e:
        return f"Error saving memory: {e}"


@tool
async def recall(ctx: ToolContext, query: str) -> str:
    """Search long-term memory for relevant information.

    query: What to search for
    """
    try:
        context = await ctx.app.memory_manager.build_context(
            ctx.agent_id,
            user_id=ctx.sender or ctx.peer_id,
            chat_id=ctx.peer_id,
            thread_id=ctx.thread_id or "",
            query=query,
            top_k=5,
        )
        return context if context else "Nothing relevant found in memory."
    except Exception as e:
        return f"Error recalling memory: {e}"


@tool
async def forget(ctx: ToolContext, query: str) -> str:
    """Delete memory entries matching a query.

    query: Description of what to forget
    """
    try:
        n = await ctx.app.memory_manager.delete_by_query(
            ctx.agent_id,
            user_id=ctx.sender or ctx.peer_id,
            chat_id=ctx.peer_id,
            thread_id=ctx.thread_id or "",
            query=query,
        )
        return f"Deleted {n} memory entry(ies) matching '{query}'."
    except Exception as e:
        return f"Error: {e}"
