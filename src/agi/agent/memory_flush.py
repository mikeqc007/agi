from __future__ import annotations

"""Pre-compaction memory flush — openclaw style.

Before compacting the context window, ask the LLM to write memories to three files:
  1. users/{peer_id}/flush/summary/summary-YYYY-MM-DD.md   — conversation summary
  2. users/{peer_id}/flush/prefs/prefs-YYYY-MM-DD.md       — user preferences/facts
  3. agents/{agent_id}/memory/YYYY-MM-DD.md        — agent business knowledge

For group chats (peer_kind=group):
  4. chats/{chat_id}/flush-YYYY-MM-DD.md           — shared chat context

For threaded conversations (thread_id set):
  5. threads/{thread_id}/flush-YYYY-MM-DD.md       — topic/task memory
"""

import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)

FLUSH_THRESHOLD = 0.70  # trigger at 70% of max_tokens (before compaction's 75%)

_FLUSH_PROMPT = """\
Pre-compaction memory flush. Review the conversation and write to these files using write_file:

1. {summary_file}
   Conversation summary: what was discussed, decisions made, tasks completed.
   Write a concise chronological summary in bullet points.

2. {prefs_file}
   User preferences and facts: persistent info about this user (name, preferences, ongoing projects, constraints).
   Only store genuinely useful long-term facts. Skip trivial details.

3. {agent_file}
   Agent knowledge: patterns or domain insights this agent should remember across all users.
   E.g. "users asking about X usually mean Y", effective response styles, domain facts learned.
   Skip if nothing agent-level was learned.

Rules:
- For each file: if it exists, preserve existing content and append new entries only.
- Write concise bullet points, not full conversation transcripts.
- If a file has nothing new to add, skip it (don't write an empty file).
- Reply [SKIP] only if ALL three files have nothing to write.
"""

_EXTRA_CHAT_PROMPT = """
4. {chat_file}
   Group chat context: shared context for this chat/channel (who is involved, ongoing topics, group decisions).
"""

_EXTRA_THREAD_PROMPT = """
5. {thread_file}
   Thread/task memory: specific context for this topic or task thread (progress, intermediate results, conclusions).
"""


def should_run_memory_flush(
    history: list[dict],
    max_tokens: int,
    threshold: float = FLUSH_THRESHOLD,
) -> bool:
    if len(history) < 10:
        return False
    estimated = sum(len(str(m.get("content") or "")) for m in history) // 4
    return estimated > max_tokens * threshold


async def run_memory_flush(
    history: list[dict[str, Any]],
    models: list[str],
    max_tokens: int,
    config_dir: str,
    state_dir: str,
    agent_id: str,
    user_id: str = "",
    peer_kind: str = "direct",
    chat_id: str = "",
    thread_id: str = "",
    custom_prompt: str = "",
) -> None:
    """Run a silent LLM turn to flush important memories to disk."""
    import json
    from pathlib import Path

    date_str = datetime.date.today().strftime("%Y-%m-%d")
    state_root = Path(config_dir) / state_dir

    safe_uid = user_id.replace("/", "_").replace("..", "_") if user_id else "default"
    user_dir = state_root / "users" / safe_uid
    (user_dir / "kb").mkdir(parents=True, exist_ok=True)
    (user_dir / "flush" / "summary").mkdir(parents=True, exist_ok=True)
    (user_dir / "flush" / "prefs").mkdir(parents=True, exist_ok=True)

    summary_file = str(user_dir / "flush" / "summary" / f"summary-{date_str}.md")
    prefs_file = str(user_dir / "flush" / "prefs" / f"prefs-{date_str}.md")
    agent_file = str(state_root / "agents" / agent_id / "memory" / f"{date_str}.md")
    Path(agent_file).parent.mkdir(parents=True, exist_ok=True)

    # Build prompt
    prompt = (custom_prompt or _FLUSH_PROMPT).format(
        summary_file=summary_file,
        prefs_file=prefs_file,
        agent_file=agent_file,
    )

    # Optional: group chat
    chat_file = ""
    if peer_kind == "group" and chat_id:
        safe_cid = chat_id.replace("/", "_").replace("..", "_")
        chat_dir = state_root / "chats" / safe_cid
        chat_dir.mkdir(parents=True, exist_ok=True)
        chat_file = str(chat_dir / f"flush-{date_str}.md")
        prompt += _EXTRA_CHAT_PROMPT.format(chat_file=chat_file)

    # Optional: thread
    thread_file = ""
    if thread_id:
        safe_tid = thread_id.replace("/", "_").replace("..", "_")
        thread_dir = state_root / "threads" / safe_tid
        thread_dir.mkdir(parents=True, exist_ok=True)
        thread_file = str(thread_dir / f"flush-{date_str}.md")
        prompt += _EXTRA_THREAD_PROMPT.format(thread_file=thread_file)

    # Pre-read existing files and inject into prompt so LLM doesn't need read_file
    existing: dict[str, str] = {}
    for label, fpath in [
        ("summary", summary_file),
        ("prefs", prefs_file),
        ("agent", agent_file),
        ("chat", chat_file),
        ("thread", thread_file),
    ]:
        if not fpath:
            continue
        try:
            p = Path(fpath)
            if p.exists():
                existing[label] = p.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    if existing:
        prompt += "\n\nExisting file contents (preserve and append only):\n"
        for label, content in existing.items():
            prompt += f"\n[{label}]\n{content}\n"

    # Build condensed history (text only, no images)
    condensed: list[dict] = []
    for m in history[-40:]:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if content and str(content).strip():
            condensed.append({"role": role, "content": str(content)[:500]})

    if not condensed:
        return

    from agi.tools.registry import get_all_schemas
    allowed = {"write_file"}
    tools = [s for s in get_all_schemas() if s.get("function", {}).get("name") in allowed]

    flush_messages = [
        {"role": "system", "content": "You are a memory assistant. Write important information to disk or reply [SKIP]."},
        *condensed,
        {"role": "user", "content": "/no_think\n" + prompt},
    ]

    try:
        from agi.agent.model_fallback import complete_with_fallback
        from agi.tools.registry import dispatch
        from agi.types import ToolContext

        ctx = ToolContext(
            agent_id=agent_id,
            session_key=f"{agent_id}:memory_flush",
            channel="internal",
            peer_kind="system",
            peer_id=user_id or "flush",
            app=None,
        )

        response = await complete_with_fallback(
            models=models,
            messages=flush_messages,
            tools=tools if tools else None,
            max_tokens=min(1024, max_tokens // 4),
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()

        if "[SKIP]" in text and not getattr(choice.message, "tool_calls", None):
            logger.debug("Memory flush: nothing to store (agent=%s)", agent_id)
            return

        tool_calls = getattr(choice.message, "tool_calls", None) or []
        for tc in tool_calls:
            name = tc.function.name
            if name not in allowed:
                continue
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                continue
            try:
                await dispatch(name, ctx, args)
            except Exception as e:
                logger.warning("Memory flush tool %s failed: %s", name, e)

        logger.info("Memory flush completed for agent=%s user=%s", agent_id, safe_uid)

    except Exception as e:
        logger.warning("Memory flush failed (agent=%s): %s", agent_id, e)
