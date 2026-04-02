from __future__ import annotations

from typing import Literal

from agi.tools.registry import tool
from agi.types import ToolContext

_todos: dict[str, list[dict]] = {}  # session_key -> list of todo items


def _get_todos(session_key: str) -> list[dict]:
    return _todos.setdefault(session_key, [])


def _render(todos: list[dict]) -> str:
    if not todos:
        return "(no todos)"
    icons = {"pending": "○", "in_progress": "◉", "completed": "✓"}
    lines = []
    for item in todos:
        icon = icons.get(item["status"], "○")
        lines.append(f"  {icon} {item['content']}")
    return "\n".join(lines)


@tool
async def todo(
    ctx: ToolContext,
    action: str,
    content: str = "",
    id: int = -1,
) -> str:
    """Manage a task list to track progress on multi-step work.

    action: One of: add, complete, in_progress, delete, list
    content: Task description (required for add)
    id: Task index (required for complete, in_progress, delete)
    """
    todos = _get_todos(ctx.session_key)

    if action == "add":
        if not content:
            return "Error: content required for add"
        todos.append({"content": content, "status": "pending"})
        result = f"Added: {content}"

    elif action == "in_progress":
        if id < 0 or id >= len(todos):
            return f"Error: invalid id {id}"
        todos[id]["status"] = "in_progress"
        result = f"In progress: {todos[id]['content']}"

    elif action == "complete":
        if id < 0 or id >= len(todos):
            return f"Error: invalid id {id}"
        todos[id]["status"] = "completed"
        result = f"Completed: {todos[id]['content']}"

    elif action == "delete":
        if id < 0 or id >= len(todos):
            return f"Error: invalid id {id}"
        removed = todos.pop(id)
        result = f"Deleted: {removed['content']}"

    elif action == "list":
        result = None

    else:
        return f"Error: unknown action '{action}'"

    rendered = _render(todos)
    output = f"\n\033[1mTasks:\033[0m\n{rendered}\n"
    if ctx.on_text:
        ctx.on_text(output)

    return rendered
