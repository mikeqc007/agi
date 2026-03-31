from __future__ import annotations

import os
from pathlib import Path

from agi.tools.registry import tool
from agi.types import ToolContext

MAX_READ = 50_000


@tool
async def read_file(ctx: ToolContext, path: str) -> str:
    """Read a file and return its contents.

    path: Absolute or relative file path
    """
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {path}"
        if p.is_dir():
            return f"Error: {path} is a directory, use list_dir"
        size = p.stat().st_size
        with open(p, "r", errors="replace") as f:
            content = f.read(MAX_READ)
        if size > MAX_READ:
            content += f"\n[...truncated, file is {size} bytes]"
        return content
    except Exception as e:
        return f"Error: {e}"


@tool
async def write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Write content to a file (creates or overwrites).

    path: File path to write
    content: Content to write
    """
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
async def list_dir(ctx: ToolContext, path: str = ".") -> str:
    """List files and directories at path.

    path: Directory path (default: current directory)
    """
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: path not found: {path}"
        if not p.is_dir():
            return f"Error: {path} is not a directory"

        entries = []
        for item in sorted(p.iterdir()):
            if item.is_dir():
                entries.append(f"[DIR]  {item.name}/")
            else:
                size = item.stat().st_size
                entries.append(f"[FILE] {item.name} ({size} bytes)")

        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Error: {e}"
