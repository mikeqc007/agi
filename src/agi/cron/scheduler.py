from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agi.storage.db import cron_list, cron_upsert, cron_delete, cron_touch
from agi.types import InboundMessage

logger = logging.getLogger(__name__)


def _build_cron_prompt(job: dict) -> str:
    import datetime
    import re
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Strip output/notify instructions the main agent may have leaked into the task
    task = job["task"]
    task = re.sub(r"[，,]?\s*(并\s*)?(输出|写入|保存|记录|存储|log|notify|通知)\s*(到|至)?\s*\S+", "", task, flags=re.IGNORECASE)
    task = task.strip("，, \t")
    return f"[Scheduled task | {now}]\n{task}"



class CronService:
    def __init__(self, db: Any, submit_fn: Any,
                 notify_fn: Any = None, on_text_fn: Any = None) -> None:
        self._db = db
        self._submit = submit_fn
        self._notify_fn = notify_fn   # async (notify: str, reply: str) -> None
        self._on_text_fn = on_text_fn  # sync (notify: str) -> Callable | None
        self._scheduler = AsyncIOScheduler()
        self._running_tasks: dict[str, set[asyncio.Task]] = {}

    async def start(self) -> None:
        self._scheduler.start()
        # Load persisted jobs
        jobs = await cron_list(self._db)
        for job in jobs:
            if job.get("enabled"):
                self._schedule(job)
        logger.info("CronService started, loaded %d jobs", len([j for j in jobs if j.get("enabled")]))

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)

    async def add(
        self,
        agent_id: str,
        peer_kind: str,
        peer_id: str,
        schedule: str,
        task: str,
        output: str = "",
        notify: str = "",
    ) -> dict:
        job_id = f"cron-{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "agent_id": agent_id,
            "peer_kind": peer_kind,
            "peer_id": peer_id,
            "schedule": schedule,
            "task": task,
            "output": output,
            "notify": notify,
            "enabled": 1,
            "created_at_ms": int(time.time() * 1000),
        }
        await cron_upsert(self._db, job)
        self._schedule(job)
        return {"id": job_id, "schedule": schedule, "task": task,
                "output": output, "notify": notify}

    async def remove(self, job_id: str) -> bool:
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass
        running = list(self._running_tasks.get(job_id, set()))
        for task in running:
            if not task.done():
                task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        await cron_delete(self._db, job_id)
        return True

    async def list_jobs(self) -> list[dict]:
        return await cron_list(self._db)

    def _schedule(self, job: dict) -> None:
        try:
            trigger = CronTrigger.from_crontab(job["schedule"])
            self._scheduler.add_job(
                self._run_job,
                trigger=trigger,
                id=job["id"],
                args=[job],
                replace_existing=True,
                misfire_grace_time=60,
            )
        except Exception as e:
            logger.warning("Failed to schedule job %s: %s", job["id"], e)

    async def _run_job(self, job: dict) -> None:
        job_id = str(job["id"])
        runner = asyncio.create_task(self._run_job_once(job))
        self._running_tasks.setdefault(job_id, set()).add(runner)
        try:
            await runner
        except asyncio.CancelledError:
            logger.info("Cron job %s cancelled", job_id)
            raise
        finally:
            tasks = self._running_tasks.get(job_id)
            if tasks:
                tasks.discard(runner)
                if not tasks:
                    self._running_tasks.pop(job_id, None)

    async def _run_job_once(self, job: dict) -> None:
        logger.info("Running cron job %s: %s", job["id"], job["task"][:60])
        output = job.get("output") or ""
        notify = job.get("notify") or ""
        try:
            # For "cli" notify, inject on_text so streaming output goes to terminal live.
            on_text = None
            if notify == "cli" and self._on_text_fn:
                on_text = self._on_text_fn("cli")
                if on_text:
                    on_text(f"\n\033[93m[cron {job['id']}]\033[0m\n")

            meta: dict[str, Any] = {
                "force_agent_id": job["agent_id"],
                "cron_job_id": job["id"],
                "tools_deny": ["cron_add", "cron_delete"],
                # Clear stale cron session history older than 6 hours to avoid
                # accumulated bad tool calls polluting future runs
                "session_max_age_hours": 6,
            }
            if on_text is not None:
                meta["on_text"] = on_text

            inbound = InboundMessage(
                id=-1,
                channel="cron",
                peer_kind=job["peer_kind"],
                peer_id=f"{job['peer_id']}:{job['id']}",
                sender=f"cron:{job['id']}",
                content=_build_cron_prompt(job),
                created_at_ms=int(time.time() * 1000),
                metadata=meta,
            )
            reply = await self._submit(inbound)
            await cron_touch(self._db, job["id"])

            if self._notify_fn:
                # Write full result to output destination (e.g. log file)
                if output and output != "log":
                    await self._notify_fn(output, reply or "")
                # Send notification (e.g. cli, telegram)
                if notify and notify != "log":
                    await self._notify_fn(notify, reply or "")
        except asyncio.CancelledError:
            logger.info("Cron job %s execution cancelled", job["id"])
            raise
        except Exception as e:
            logger.error("Cron job %s failed: %s", job["id"], e)
