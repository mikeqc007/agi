from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from agi.types import InboundMessage

logger = logging.getLogger(__name__)


def _make_prefixed_on_text(label: str, _parent_fn: Any) -> Any:
    """Return a streaming callback that logs subagent output to the log file only.

    Subagent intermediate output is intentionally NOT forwarded to the terminal
    to keep the main interface clean. Final results are delivered via _announce().
    """
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
        submit_fn: Any,             # async (InboundMessage) -> str
        max_concurrent: int = 4,
        default_timeout: int = 300,
    ) -> None:
        self._submit = submit_fn
        self._sem = asyncio.Semaphore(max(1, max_concurrent))
        self._default_timeout = default_timeout
        self._runs: dict[str, dict] = {}
        self._children: dict[str, set[str]] = {}  # parent_key -> {run_id}

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
            "started_at_ms": int(time.time() * 1000),
            "parent_on_text": on_text,  # used by _announce to route output in CLI mode
        }
        self._runs[run_id] = run
        self._children.setdefault(session_key, set()).add(run_id)
        logger.info(
            "Subagent spawned run_id=%s label=%s parent=%s child=%s agent=%s",
            run_id,
            run["label"],
            session_key,
            child_key,
            agent_id,
        )

        run["task"] = asyncio.create_task(
            self._job(run, task, agent_id, child_key, channel, peer_kind, peer_id,
                      account_id, thread_id, meta_patch, session_key,
                      timeout or self._default_timeout, on_text)
        )

        return {
            "status": "accepted",
            "run_id": run_id,
            "session_key": child_key,
            "note": "Subagent started; result will be announced when done.",
        }

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
        t = run.get("task")
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _job(
        self, run: dict, task: str, agent_id: str, child_key: str,
        channel: str, peer_kind: str, peer_id: str,
        account_id: str | None, thread_id: str | None,
        meta_patch: dict, parent_key: str, timeout: int,
        on_text: Any = None,
    ) -> None:
        async with self._sem:
            logger.info(
                "Subagent started run_id=%s child=%s parent=%s timeout=%ss",
                run.get("run_id"),
                child_key,
                parent_key,
                timeout,
            )
            try:
                label = run.get("label") or run["run_id"][:8]
                child_on_text = _make_prefixed_on_text(label, on_text) if on_text else None
                meta: dict[str, Any] = {
                    "force_agent_id": agent_id,
                    "force_session_key": child_key,
                    "session_meta_patch": meta_patch,
                    "run_id": run["run_id"],
                }
                if child_on_text:
                    meta["on_text"] = child_on_text
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
                run["result"] = str(result or "")[:2000]
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
                self._children.get(parent_key, set()).discard(run["run_id"])
                elapsed_ms = int(run["ended_at_ms"] - run.get("started_at_ms", run["ended_at_ms"]))
                logger.info(
                    "Subagent finished run_id=%s status=%s elapsed_ms=%d error=%s",
                    run.get("run_id"),
                    run.get("status"),
                    elapsed_ms,
                    (run.get("error") or "")[:200],
                )
                await self._announce(run, parent_key, channel, peer_kind, peer_id,
                                     account_id, thread_id, agent_id)

    async def _announce(
        self, run: dict, parent_key: str, channel: str,
        peer_kind: str, peer_id: str, account_id: str | None,
        thread_id: str | None, agent_id: str,
    ) -> None:
        status = run.get("status", "")
        if status == "ok":
            text = f"[subagent:{run['run_id']}] done\n{run.get('result', '')[:1200]}"
        elif status == "error":
            text = f"[subagent:{run['run_id']}] error: {run.get('error', '')}"
        else:
            text = f"[subagent:{run['run_id']}] {status}"

        try:
            logger.info(
                "Subagent announce run_id=%s status=%s -> parent=%s",
                run.get("run_id"),
                status,
                parent_key,
            )
            parent_on_text = run.get("parent_on_text")

            # In CLI/streaming mode: print the result immediately so the
            # terminal doesn't block on a full parent-agent LLM round-trip.
            if parent_on_text:
                label = run.get("label") or run["run_id"][:8]
                if status == "ok":
                    parent_on_text(f"\n\033[92m[{label}]\033[0m 完成:\n{run.get('result', '')}\n")
                elif status == "error":
                    parent_on_text(f"\n\033[91m[{label}]\033[0m 失败: {run.get('error', '')}\n")
                else:
                    parent_on_text(f"\n[{label}] {status}\n")

            # Submit the announce to update the parent session's history.
            # Always fire-and-forget: we never need the return value, and
            # awaiting would hold the session lock (blocking user input).
            inbound = InboundMessage(
                id=-1, channel=channel,
                peer_kind=peer_kind, peer_id=peer_id,
                account_id=account_id, thread_id=thread_id,
                sender="subagent_result",
                content=text,
                created_at_ms=int(time.time() * 1000),
                metadata={
                    "force_agent_id": agent_id,
                    "force_session_key": parent_key,
                    "run_id": f"{run['run_id']}-announce",
                },
            )
            asyncio.create_task(self._submit(inbound))
        except Exception as e:
            logger.warning("Subagent announce failed: %s", e)


def _depth(session_key: str) -> int:
    return session_key.lower().count(":subagent:")
