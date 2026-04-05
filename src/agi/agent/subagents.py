from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from agi.types import InboundMessage
from agi.agent import tracer as _tracer_mod

logger = logging.getLogger(__name__)


def _make_prefixed_on_text(label: str) -> Any:
    """Logs subagent output to log file only — not forwarded to terminal."""
    line_buf: list[str] = []

    def emit(text: str) -> None:
        if not text:
            return
        for ch in text:
            line_buf.append(ch)
            if ch == "\n":
                line = "".join(line_buf).strip()
                if line:
                    logger.info("[%s] %s", label, line)
                line_buf.clear()

    return emit


class SubagentManager:
    def __init__(
        self,
        submit_fn: Any,
        max_concurrent: int = 4,
        default_timeout: int = 300,
    ) -> None:
        self._submit = submit_fn
        self._sem = asyncio.Semaphore(max(1, max_concurrent))
        self._default_timeout = default_timeout
        self._runs: dict[str, dict] = {}
        self._children: dict[str, set[str]] = {}  # parent_key -> {run_id}
        # pending asyncio.Tasks for each run_id
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def spawn(
        self,
        *,
        task: str,
        label: str | None = None,
        agent_id: str,
        session_key: str,
        channel: str,
        peer_kind: str,
        peer_id: str,
        account_id: str | None = None,
        thread_id: str | None = None,
        max_depth: int = 3,
        max_children: int = 5,
        timeout: int | None = None,
        model_override: str | None = None,
        tool_profile: str | None = None,
        tools_allow: list[str] | None = None,
        tools_deny: list[str] | None = None,
        on_text: Any = None,
    ) -> dict:
        depth = _depth(session_key)
        if depth >= max_depth:
            return {"status": "forbidden", "error": "max spawn depth reached"}

        active = sum(
            1 for rid in self._children.get(session_key, set())
            if self._runs.get(rid, {}).get("status") == "running"
        )
        if active >= max_children:
            return {"status": "forbidden", "error": "max children reached"}

        run_id = f"sub-{uuid.uuid4().hex[:8]}"
        child_key = f"{session_key}:subagent:{run_id}"

        meta_patch: dict[str, Any] = {"spawn_depth": depth + 1, "spawned_by": session_key}
        if model_override:
            meta_patch["model_override"] = model_override
        if tool_profile:
            meta_patch["tool_profile"] = tool_profile
        if tools_allow:
            meta_patch["tools_allow"] = tools_allow
        if tools_deny:
            meta_patch["tools_deny"] = tools_deny

        run: dict[str, Any] = {
            "run_id": run_id,
            "label": label or task[:60],
            "status": "running",
            "agent_id": agent_id,
            "session_key": child_key,
            "parent_key": session_key,
            "task_desc": task,
            "started_at_ms": int(time.time() * 1000),
            "trajectory": [],  # list of {tool, args, result}
        }
        self._runs[run_id] = run
        self._children.setdefault(session_key, set()).add(run_id)

        _tracer_mod.emit(
            "subagent.spawn",
            session_key=session_key,
            run_id=run_id,
            child_key=child_key,
            label=run["label"],
            agent_id=agent_id,
            depth=depth + 1,
            task=task[:200],
        )
        logger.info(
            "Subagent spawned run_id=%s label=%s parent=%s agent=%s",
            run_id, run["label"], session_key, agent_id,
        )

        # Start as background task — returns immediately so LLM can spawn more
        t = asyncio.create_task(
            self._job(run, task, agent_id, child_key, channel, peer_kind, peer_id,
                      account_id, thread_id, meta_patch, session_key,
                      timeout or self._default_timeout)
        )
        self._tasks[run_id] = t

        return {
            "status": "pending",
            "run_id": run_id,
            "label": run["label"],
            "note": "Subagent started. Call wait_subagents to collect all results.",
        }

    async def wait_all(self, session_key: str, on_text: Any = None) -> list[dict]:
        """Wait for all pending subagents under session_key and return their results."""
        run_ids = list(self._children.get(session_key, set()))
        if not run_ids:
            return []

        pending_tasks = [
            (rid, self._tasks[rid])
            for rid in run_ids
            if rid in self._tasks and not self._tasks[rid].done()
        ]

        if pending_tasks:
            if on_text:
                labels = [self._runs[rid].get("label", rid) for rid, _ in pending_tasks]
                on_text(f"\033[90m[等待子任务: {', '.join(labels)}]\033[0m\n")
            await asyncio.gather(*[t for _, t in pending_tasks], return_exceptions=True)

        results = []
        for rid in run_ids:
            run = self._runs.get(rid)
            if not run:
                continue
            if on_text:
                label_str = run.get("label") or rid[:8]
                if run["status"] == "ok":
                    on_text(f"\033[92m[{label_str}]\033[0m 完成\n")
                else:
                    on_text(f"\033[91m[{label_str}]\033[0m {run['status']}: {run.get('error', '')}\n")
            results.append({
                "run_id": rid,
                "label": run.get("label", ""),
                "status": run["status"],
                "result": run.get("result", ""),
                "error": run.get("error", ""),
                "elapsed_ms": run.get("ended_at_ms", 0) - run.get("started_at_ms", 0),
                "trajectory": run.get("trajectory", []),
            })
            self._tasks.pop(rid, None)

        self._children.pop(session_key, None)
        return results

    async def steer(self, run_id: str, message: str) -> bool:
        run = self._runs.get(run_id)
        if not run or run.get("status") != "running":
            return False
        inbound = InboundMessage(
            id=-1, channel="system",
            peer_kind=str(run.get("peer_kind", "direct")),
            peer_id=str(run.get("peer_id", "")),
            sender="steer",
            content=f"[steer] {message}",
            created_at_ms=int(time.time() * 1000),
            metadata={"force_session_key": run["session_key"]},
        )
        asyncio.create_task(self._submit(inbound))
        return True

    def kill(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run:
            return False
        t = self._tasks.get(run_id)
        if t and not t.done():
            t.cancel()
            return True
        return False

    def list_runs(self, parent_key: str) -> list[dict]:
        out = []
        for run in self._runs.values():
            if run.get("parent_key") == parent_key or run.get("session_key") == parent_key:
                out.append({
                    "run_id": run["run_id"],
                    "label": run.get("label", ""),
                    "status": run.get("status", ""),
                    "agent_id": run.get("agent_id", ""),
                    "session_key": run.get("session_key", ""),
                    "started_at_ms": run.get("started_at_ms"),
                    "ended_at_ms": run.get("ended_at_ms"),
                    "error": run.get("error"),
                })
        return sorted(out, key=lambda x: x.get("started_at_ms") or 0, reverse=True)

    def has_pending(self, session_key: str) -> bool:
        """True if this session has any subagents still running."""
        return any(
            self._runs.get(rid, {}).get("status") == "running"
            for rid in self._children.get(session_key, set())
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _job(
        self, run: dict, task: str, agent_id: str, child_key: str,
        channel: str, peer_kind: str, peer_id: str,
        account_id: str | None, thread_id: str | None,
        meta_patch: dict, parent_key: str, timeout: int,
    ) -> None:
        async with self._sem:
            logger.info(
                "Subagent started run_id=%s child=%s parent=%s timeout=%ss",
                run.get("run_id"), child_key, parent_key, timeout,
            )
            try:
                label = run.get("label") or run["run_id"][:8]
                child_on_text = _make_prefixed_on_text(label)

                def _record_tool(tool: str, args: dict, result: Any,
                                  _run: dict = run) -> None:
                    _run["trajectory"].append({"tool": tool, "args": args, "result": result})

                meta: dict[str, Any] = {
                    "force_agent_id": agent_id,
                    "force_session_key": child_key,
                    "session_meta_patch": meta_patch,
                    "run_id": run["run_id"],
                    "on_text": child_on_text,
                    "on_tool_result": _record_tool,
                }
                inbound = InboundMessage(
                    id=-1, channel=channel,
                    peer_kind=peer_kind, peer_id=peer_id,
                    account_id=account_id, thread_id=thread_id,
                    sender="subagent",
                    content=f"[Subagent task]\n{task}",
                    created_at_ms=int(time.time() * 1000),
                    metadata=meta,
                )
                result = await asyncio.wait_for(self._submit(inbound), timeout=timeout)
                run["status"] = "ok"
                run["result"] = str(result or "")[:4000]
            except asyncio.TimeoutError:
                run["status"] = "timeout"
                run["error"] = f"Timed out after {timeout}s"
            except asyncio.CancelledError:
                run["status"] = "killed"
                raise
            except Exception as e:
                run["status"] = "error"
                run["error"] = str(e)
            finally:
                run["ended_at_ms"] = int(time.time() * 1000)
                elapsed_ms = int(run["ended_at_ms"] - run.get("started_at_ms", run["ended_at_ms"]))
                logger.info(
                    "Subagent finished run_id=%s status=%s elapsed_ms=%d",
                    run.get("run_id"), run.get("status"), elapsed_ms,
                )
                _tracer_mod.emit(
                    "subagent.done",
                    session_key=parent_key,
                    run_id=run.get("run_id"),
                    child_key=child_key,
                    status=run.get("status"),
                    elapsed_ms=elapsed_ms,
                    result=run.get("result") or "",
                    error=run.get("error"),
                )


def _depth(session_key: str) -> int:
    return session_key.lower().count(":subagent:")
