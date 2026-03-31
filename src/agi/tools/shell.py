from __future__ import annotations

import asyncio
import logging
import os
import shlex

from agi.tools.registry import tool
from agi.types import ToolContext

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_OUTPUT = 8000

# Commands always blocked regardless of policy
_BLOCKED = {"rm -rf /", "mkfs", "dd if=/dev/zero", ":(){:|:&};:"}


@tool
async def shell(ctx: ToolContext, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute a shell command and return stdout+stderr.

    command: Shell command to execute
    timeout: Timeout in seconds (default 30)
    """
    for blocked in _BLOCKED:
        if blocked in command:
            return f"Error: command blocked for safety: {blocked}"

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "TERM": "dumb"},
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=max(1, int(timeout))
            )
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: command timed out after {timeout}s"

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        combined = out
        if err:
            combined += f"\nSTDERR:\n{err}" if out else err

        if len(combined) > MAX_OUTPUT:
            combined = combined[:MAX_OUTPUT] + f"\n[...truncated]"

        rc = proc.returncode or 0
        if rc != 0:
            combined += f"\n[exit code: {rc}]"
        return combined or "(no output)"

    except Exception as e:
        return f"Error: {e}"
