from __future__ import annotations

import re

from agi.tools.registry import tool
from agi.types import ToolContext

# Same pattern used in _build_cron_prompt to strip output/notify leakage
_STRIP_OUTPUT_RE = re.compile(
    r"[，,]?\s*(并\s*)?(输出|写入|保存|记录|存储|log|notify|通知)\s*(到|至)?\s*\S+",
    re.IGNORECASE,
)


def _clean_task(task: str) -> str:
    """Strip output/notify instructions leaked into task by the model."""
    cleaned = _STRIP_OUTPUT_RE.sub("", task)
    return cleaned.strip("，, \t") or task.strip()


def _task_slug(task: str) -> str:
    """Turn task description into a safe filename stem."""
    # Keep Chinese chars, ASCII letters, digits — drop punctuation/spaces
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "", task.strip())
    if not cleaned:
        return "task"
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return cleaned[:20]
    words = re.findall(r"[a-zA-Z0-9]+", cleaned)[:4]
    return "_".join(w.lower() for w in words)[:40]


@tool
async def cron_add(ctx: ToolContext, schedule: str, task: str,
                   output: str = "", notify: str = "cli") -> str:
    """Add a scheduled task (cron job).

    schedule: Cron expression, e.g. '0 8 * * *' for daily at 8am,
              '*/30 * * * *' for every 30 min, '* * * * *' for every minute
    task: ONLY the work to perform, e.g. '搜索白银价格'. Do NOT include
          output path or notify instructions — those are handled separately.
    output: Log file path. Leave empty to auto-generate from task name.
    notify: 'cli' (default), 'telegram:CHAT_ID', 'discord:CHAN_ID', or '' for none.
    """
    from pathlib import Path

    p = Path(output) if output else None
    if p and p.is_absolute():
        # User explicitly provided an absolute path — respect it.
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Relative paths (model-generated like 'logs/metals.log') are ignored.
        # Always auto-generate from the clean task name.
        slug = _task_slug(_clean_task(task))
        output = f"logs/{slug}.log"

    svc = getattr(ctx.app, "cron_service", None)
    if not svc:
        return "Error: cron service not available"
    result = await svc.add(
        agent_id=ctx.agent_id,
        peer_kind=ctx.peer_kind,
        peer_id=ctx.peer_id,
        schedule=schedule,
        task=task,
        output=output,
        notify=notify,
    )
    parts = [f"Scheduled: {result['id']}", f"Schedule: {schedule}",
             f"Task: {task}", f"Output: {output}", f"Notify: {notify}"]
    return "\n".join(parts)


@tool
async def cron_list(ctx: ToolContext) -> str:
    """List all scheduled tasks."""
    svc = getattr(ctx.app, "cron_service", None)
    if not svc:
        return "Error: cron service not available"
    jobs = await svc.list_jobs()
    if not jobs:
        return "No scheduled tasks."
    lines = []
    for j in jobs:
        status = "✓" if j.get("enabled") else "✗"
        lines.append(f"[{status}] {j['id']}: {j['schedule']} → {j['task'][:60]}")
    return "\n".join(lines)


@tool
async def cron_delete(ctx: ToolContext, job_id: str) -> str:
    """Delete a scheduled task.

    job_id: The job ID returned by cron_add
    """
    svc = getattr(ctx.app, "cron_service", None)
    if not svc:
        return "Error: cron service not available"
    ok = await svc.remove(job_id)
    return f"Deleted job {job_id}" if ok else f"Job {job_id} not found"
