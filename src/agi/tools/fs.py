from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from agi.tools.registry import tool
from agi.types import ToolContext

MAX_READ = 50_000


@tool
async def read_file(ctx: ToolContext, path: str, offset: int = 0, limit: int = 0) -> str:
    """Read a file and return its contents.

    path: Absolute or relative file path
    offset: Line number to start reading from (0 = beginning)
    limit: Number of lines to read (0 = all)
    """
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {path}"
        if p.is_dir():
            return f"Error: {path} is a directory, use list_dir"
        with open(p, "r", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(0, offset)
        end = (start + limit) if limit > 0 else total
        selected = lines[start:end]
        content = "".join(
            f"{start + i + 1}\t{line}" for i, line in enumerate(selected)
        )
        if len(content) > MAX_READ:
            content = content[:MAX_READ] + f"\n[...truncated]"
        if offset > 0 or limit > 0:
            content = f"[Lines {start + 1}-{min(end, total)} of {total}]\n" + content
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
async def edit_file(ctx: ToolContext, path: str, old_str: str, new_str: str) -> str:
    """Edit a file by replacing an exact string with new content.

    path: File path to edit
    old_str: Exact string to find and replace (must be unique in the file)
    new_str: Replacement string
    """
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {path}"
        content = p.read_text(errors="replace")
        count = content.count(old_str)
        if count == 0:
            return "Error: old_str not found in file"
        if count > 1:
            return f"Error: old_str found {count} times, must be unique"
        p.write_text(content.replace(old_str, new_str, 1))
        return "OK"
    except Exception as e:
        return f"Error: {e}"


@tool
async def grep(ctx: ToolContext, pattern: str, path: str = ".", recursive: bool = True) -> str:
    """Search for a regex pattern in files.

    pattern: Regular expression to search for
    path: File or directory to search in (default: current directory)
    recursive: Search recursively in directories (default: true)
    """
    try:
        p = Path(path).expanduser()
        regex = re.compile(pattern)
        results = []
        MAX_RESULTS = 200

        def search_file(fp: Path) -> None:
            try:
                lines = fp.read_text(errors="replace").splitlines()
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        results.append(f"{fp}:{i}: {line}")
                        if len(results) >= MAX_RESULTS:
                            return
            except Exception:
                pass

        if p.is_file():
            search_file(p)
        elif p.is_dir():
            files = p.rglob("*") if recursive else p.glob("*")
            for f in sorted(files):
                if f.is_file():
                    search_file(f)
                if len(results) >= MAX_RESULTS:
                    results.append(f"[truncated at {MAX_RESULTS} results]")
                    break
        else:
            return f"Error: path not found: {path}"

        return "\n".join(results) if results else "(no matches)"
    except re.error as e:
        return f"Error: invalid regex: {e}"
    except Exception as e:
        return f"Error: {e}"


@tool
async def glob(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern.

    pattern: Glob pattern (e.g. **/*.py, src/*.ts)
    path: Root directory to search from (default: current directory)
    """
    try:
        p = Path(path).expanduser()
        if not p.is_dir():
            return f"Error: path not found or not a directory: {path}"
        matches = sorted(p.glob(pattern))
        if not matches:
            return "(no matches)"
        return "\n".join(str(m) for m in matches[:500])
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
