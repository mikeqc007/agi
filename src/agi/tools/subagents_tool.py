from __future__ import annotations

import json

from agi.tools.registry import tool
from agi.types import ToolContext


@tool
async def spawn_agent(
    ctx: ToolContext,
    task: str,
    label: str = "",
    agent_id: str = "",
    model: str = "",
    tool_profile: str = "",
) -> str:
    """Spawn a subagent to handle a task concurrently.

    task: Detailed task description for the subagent
    label: Short label for this subagent (optional)
    agent_id: Which agent config to use (default: same as parent)
    model: Override model for this subagent (optional)
    tool_profile: Tool policy: default/safe/minimal (optional)
    """
    mgr = getattr(ctx.app, "subagent_manager", None)
    if not mgr:
        return "Error: subagent manager not available"

    cfg = ctx.app.cfg
    # Inherit parent's primary model if caller didn't specify one
    effective_agent_id = agent_id or ctx.agent_id
    if not model:
        model = cfg.agent(effective_agent_id).model.primary
    result = await mgr.spawn(
        task=task,
        label=label or None,
        agent_id=effective_agent_id,
        session_key=ctx.session_key,
        channel=ctx.channel,
        peer_kind=ctx.peer_kind,
        peer_id=ctx.peer_id,
        account_id=ctx.account_id,
        thread_id=ctx.thread_id,
        max_depth=cfg.max_subagent_depth,
        max_children=cfg.max_subagent_children,
        timeout=cfg.subagent_timeout_seconds,
        model_override=model,
        tool_profile=tool_profile or None,
        on_text=ctx.on_text,
    )
    return json.dumps(result, ensure_ascii=False)


@tool
async def list_subagents(ctx: ToolContext) -> str:
    """List all subagents spawned by the current session."""
    mgr = getattr(ctx.app, "subagent_manager", None)
    if not mgr:
        return "[]"
    runs = mgr.list_runs(ctx.session_key)
    return json.dumps(runs, ensure_ascii=False, indent=2)


@tool
async def kill_subagent(ctx: ToolContext, run_id: str) -> str:
    """Kill a running subagent.

    run_id: The run_id returned by spawn_agent
    """
    mgr = getattr(ctx.app, "subagent_manager", None)
    if not mgr:
        return "Error: subagent manager not available"
    ok = mgr.kill(run_id)
    return f"Killed {run_id}" if ok else f"Could not kill {run_id} (not found or already done)"
