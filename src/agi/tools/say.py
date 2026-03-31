from __future__ import annotations

from agi.tools.registry import tool
from agi.types import ToolContext


@tool
async def say(ctx: ToolContext, message: str) -> str:
    """Send a status update or progress message to the user.

    message: The message to display to the user.
    """
    if ctx.on_text:
        ctx.on_text(message + "\n")
    return ""
