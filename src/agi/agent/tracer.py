"""Session execution tracer.

Each turn produces one JSON file: logs/traces/{date}/{session_key}/{turn_id}.json

Structure:
{
  "meta": { timestamp, session_key, turn_id },
  "turn": {
    "query": "...",
    "main": {
      "agent": "...",
      "model": "...",
      "trajectory": [
        {
          "iter": 1,
          "think": "...",           # LLM text before tool call
          "action": {
            "type": "tool_call",
            "tool": "web_search",
            "args": { "query": "..." }
          },
          "tool_result": { ... }    # actual tool output (null if finish)
        },
        {
          "iter": 2,
          "think": "...",
          "action": { "type": "finish", "result": "..." },
          "tool_result": null
        }
      ]
    },
    "subagents": [
      {
        "run_id": "sub-abc",
        "label": "...",
        "agent": "...",
        "assigned_task": "...",
        "trajectory": [ ... ],      # same structure as main.trajectory
        "final_result": "...",
        "status": "ok",
        "elapsed_ms": 3200,
        "error": null
      }
    ],
    "answer": "..."
  },
  "diagnostics": {
    "duration_ms": 6500,
    "total_tool_calls": 3,
    "subagent_count": 2,
    "no_tool_subagents": [],
    "error_subagents": []
  }
}
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_tracer: "Tracer | None" = None


def init(logs_dir: Path) -> "Tracer":
    global _tracer
    _tracer = Tracer(logs_dir)
    return _tracer


def get() -> "Tracer | None":
    return _tracer


def emit(event_type: str, **kwargs: Any) -> None:
    if _tracer:
        _tracer.emit(event_type, **kwargs)


# ---------------------------------------------------------------------------
# Per-agent execution state (shared by main and each subagent)
# ---------------------------------------------------------------------------

class _AgentExec:
    """Tracks one agent's trajectory: a sequence of (think, action, tool_result) iters."""

    def __init__(self) -> None:
        self.trajectory: list[dict] = []
        # pending state for the current iter
        self._think: str = ""
        self._action: dict | None = None
        self._tool_result: Any = None

    def record_think(self, think: str) -> None:
        self._think = think

    def record_tool_call(self, tool: str, args: dict) -> None:
        """Called when LLM decides to call a tool."""
        self._action = {"type": "tool_call", "tool": tool, "args": args}

    def record_tool_result(self, result: Any) -> None:
        """Called after the tool returns — flushes the current iter."""
        self._tool_result = result
        self._flush()

    def record_finish(self, result: str) -> None:
        """Called when LLM produces a final answer (no tool call)."""
        self._action = {"type": "finish", "result": result}
        self._tool_result = None
        self._flush()

    def _flush(self) -> None:
        if self._action is None:
            return
        self.trajectory.append({
            "iter": len(self.trajectory) + 1,
            "think": self._think,
            "action": self._action,
            "tool_result": self._tool_result,
        })
        self._think = ""
        self._action = None
        self._tool_result = None


# ---------------------------------------------------------------------------
# Per-turn state
# ---------------------------------------------------------------------------

class _TurnState:
    def __init__(self, session_key: str, agent_id: str, query: str, model: str) -> None:
        self.turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        self.session_key = session_key
        self.agent_id = agent_id
        self.query = query
        self.model = model
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_ms = int(time.time() * 1000)

        self.main = _AgentExec()
        self.subagents: dict[str, dict] = {}   # run_id -> subagent meta + _AgentExec
        self.answer: str = ""

        self.total_tool_calls: int = 0
        self.no_tool_subagents: list[str] = []
        self.error_subagents: list[str] = []

    # -- main agent --

    def main_think(self, think: str) -> None:
        self.main.record_think(think)

    def main_tool_call(self, tool: str, args: dict) -> None:
        self.main.record_tool_call(tool, args)

    def main_tool_result(self, result: Any) -> None:
        self.main.record_tool_result(result)
        self.total_tool_calls += 1

    def main_finish(self, result: str) -> None:
        self.main.record_finish(result)

    # -- subagents --

    def spawn_subagent(self, run_id: str, label: str, task: str,
                       child_key: str, agent_id: str, depth: int) -> None:
        self.subagents[run_id] = {
            "run_id": run_id,
            "label": label,
            "agent": agent_id,
            "child_key": child_key,
            "depth": depth,
            "assigned_task": task,
            "exec": _AgentExec(),
            "final_result": None,
            "status": "running",
            "elapsed_ms": 0,
            "error": None,
        }

    def subagent_think(self, run_id: str, think: str) -> None:
        sub = self.subagents.get(run_id)
        if sub:
            sub["exec"].record_think(think)

    def subagent_tool_call(self, run_id: str, tool: str, args: dict) -> None:
        sub = self.subagents.get(run_id)
        if sub:
            sub["exec"].record_tool_call(tool, args)

    def subagent_tool_result(self, run_id: str, result: Any) -> None:
        sub = self.subagents.get(run_id)
        if sub:
            sub["exec"].record_tool_result(result)
            self.total_tool_calls += 1

    def subagent_done(self, run_id: str, status: str, result: str,
                      error: str | None, elapsed_ms: int) -> None:
        sub = self.subagents.get(run_id)
        if not sub:
            return
        sub["exec"].record_finish(result if status == "ok" else (error or ""))
        sub["status"] = status
        sub["final_result"] = result if status == "ok" else None
        sub["elapsed_ms"] = elapsed_ms
        sub["error"] = error
        if status != "ok":
            self.error_subagents.append(run_id)
        if not sub["exec"].trajectory or all(
            not t.get("action") or t["action"].get("type") == "finish"
            for t in sub["exec"].trajectory
            if t.get("action", {}).get("type") != "finish"
        ):
            # subagent that called no tools
            tool_iters = [t for t in sub["exec"].trajectory
                          if t.get("action", {}).get("type") == "tool_call"]
            if not tool_iters:
                self.no_tool_subagents.append(run_id)

    def to_dict(self, duration_ms: int) -> dict:
        subagents_out = []
        for sub in self.subagents.values():
            subagents_out.append({
                "run_id": sub["run_id"],
                "label": sub["label"],
                "agent": sub["agent"],
                "assigned_task": sub["assigned_task"],
                "depth": sub["depth"],
                "trajectory": sub["exec"].trajectory,
                "final_result": sub["final_result"],
                "status": sub["status"],
                "elapsed_ms": sub["elapsed_ms"],
                "error": sub["error"],
            })

        return {
            "meta": {
                "timestamp": self.started_at,
                "session_key": self.session_key,
                "turn_id": self.turn_id,
            },
            "turn": {
                "query": self.query,
                "main": {
                    "agent": self.agent_id,
                    "model": self.model,
                    "trajectory": self.main.trajectory,
                },
                "subagents": subagents_out,
                "answer": self.answer,
            },
            "diagnostics": {
                "duration_ms": duration_ms,
                "total_tool_calls": self.total_tool_calls,
                "subagent_count": len(self.subagents),
                "no_tool_subagents": self.no_tool_subagents,
                "error_subagents": self.error_subagents,
            },
        }


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    def __init__(self, logs_dir: Path) -> None:
        self._base = logs_dir / "traces"
        self._turns: dict[str, _TurnState] = {}       # session_key -> turn
        self._child_to_run: dict[str, str] = {}        # child_key -> run_id
        self._child_to_parent: dict[str, str] = {}     # child_key -> parent session_key

    def emit(self, event_type: str, **kwargs: Any) -> None:
        asyncio.ensure_future(self._handle(event_type, kwargs))

    async def _handle(self, event_type: str, data: dict) -> None:
        try:
            h = {
                "turn.start":        self._on_turn_start,
                "turn.think":        self._on_turn_think,
                "tool.call":         self._on_tool_call,
                "tool.result":       self._on_tool_result,
                "subagent.spawn":    self._on_subagent_spawn,
                "subagent.done":     self._on_subagent_done,
                "turn.end":          self._on_turn_end,
            }.get(event_type)
            if h:
                r = h(data)
                if asyncio.iscoroutine(r):
                    await r
        except Exception as e:
            logger.debug("Tracer error [%s]: %s", event_type, e)

    def _resolve(self, session_key: str) -> tuple[_TurnState | None, str | None]:
        """Return (turn, run_id). run_id is None for main agent."""
        if session_key in self._child_to_parent:
            parent_key = self._child_to_parent[session_key]
            run_id = self._child_to_run[session_key]
            return self._turns.get(parent_key), run_id
        return self._turns.get(session_key), None

    def _on_turn_start(self, d: dict) -> None:
        sk = d["session_key"]
        self._turns[sk] = _TurnState(
            session_key=sk,
            agent_id=d.get("agent_id", ""),
            query=d.get("content", ""),
            model=d.get("model", ""),
        )

    def _on_turn_think(self, d: dict) -> None:
        sk = d["session_key"]
        think = d.get("think", "")
        turn, run_id = self._resolve(sk)
        if not turn:
            return
        if run_id:
            turn.subagent_think(run_id, think)
        else:
            turn.main_think(think)

    def _on_tool_call(self, d: dict) -> None:
        sk = d["session_key"]
        tool = d.get("tool", "")
        args = d.get("args", {})
        turn, run_id = self._resolve(sk)
        if not turn:
            return
        if run_id:
            turn.subagent_tool_call(run_id, tool, args)
        else:
            turn.main_tool_call(tool, args)

    def _on_tool_result(self, d: dict) -> None:
        sk = d["session_key"]
        result = d.get("result")
        turn, run_id = self._resolve(sk)
        if not turn:
            return
        if run_id:
            turn.subagent_tool_result(run_id, result)
        else:
            turn.main_tool_result(result)

    def _on_subagent_spawn(self, d: dict) -> None:
        sk = d["session_key"]
        run_id = d["run_id"]
        child_key = d["child_key"]
        turn = self._turns.get(sk)
        if turn:
            turn.spawn_subagent(
                run_id=run_id,
                label=d.get("label", ""),
                task=d.get("task", ""),
                child_key=child_key,
                agent_id=d.get("agent_id", ""),
                depth=d.get("depth", 1),
            )
        self._child_to_run[child_key] = run_id
        self._child_to_parent[child_key] = sk

    def _on_subagent_done(self, d: dict) -> None:
        sk = d["session_key"]
        run_id = d["run_id"]
        child_key = d.get("child_key", "")
        turn = self._turns.get(sk)
        if turn:
            turn.subagent_done(
                run_id=run_id,
                status=d.get("status", "error"),
                result=d.get("result", ""),
                error=d.get("error"),
                elapsed_ms=d.get("elapsed_ms", 0),
            )
        self._child_to_run.pop(child_key, None)
        self._child_to_parent.pop(child_key, None)

    async def _on_turn_end(self, d: dict) -> None:
        sk = d["session_key"]
        turn = self._turns.pop(sk, None)
        if not turn:
            return
        turn.answer = d.get("reply", "")
        turn.main_finish(turn.answer)
        duration_ms = d.get("duration_ms", 0)
        doc = turn.to_dict(duration_ms)
        await self._write(sk, turn.turn_id, doc)

    async def _write(self, session_key: str, turn_id: str, doc: dict) -> None:
        from datetime import date
        day = date.today().isoformat()
        safe_sk = session_key.replace(":", "_").replace("/", "_")[:80]
        d = self._base / day / safe_sk
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{turn_id}.json"
        try:
            path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("Tracer write error: %s", e)

    def close(self) -> None:
        pass
