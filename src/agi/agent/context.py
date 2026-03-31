from __future__ import annotations

from typing import Any

from agi.config import AgentConfig


MAX_HISTORY_TOKENS_ESTIMATE = 1500  # chars / 4 ≈ tokens, keep conservatively


def build_messages(
    agent_cfg: AgentConfig,
    history: list[dict[str, Any]],
    user_message: dict[str, Any] | None,
    memory_context: str = "",
    skill_context: str = "",
    agent_md: str = "",
) -> list[dict[str, Any]]:
    """Assemble the message list sent to the LLM."""
    system = _build_system(agent_cfg, memory_context, skill_context, agent_md)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    # Trim history to avoid context overflow (compaction handles this normally,
    # but as a safety net we hard-trim oldest messages)
    trimmed = _trim_history(history)
    messages.extend(trimmed)

    if user_message is not None:
        messages.append(user_message)

    # Sanitize to fix structural issues that cause API 400 errors (e.g. GLM
    # "messages 参数非法"): consecutive user messages, orphaned tool results, etc.
    messages = _sanitize_messages(messages)

    return messages


def _build_system(agent_cfg: AgentConfig, memory_context: str, skill_context: str = "", agent_md: str = "") -> str:
    import datetime
    today = datetime.date.today().strftime("%Y-%m-%d")
    parts = [agent_cfg.system_prompt.strip(), f"\nToday's date: {today}"]

    if agent_md:
        parts.append("\n" + agent_md)

    if skill_context:
        parts.append(
            "\n## Skills\n"
            "Before replying, check if any skill below matches the user's request. "
            "If one matches, you MUST call `read_file` with the skill's `path` value to get the full instructions, "
            "then follow those instructions exactly.\n\n"
            + skill_context
        )

    if memory_context:
        parts.append(
            "\n## Relevant memories\n"
            + memory_context
            + "\n(Use these as context; do not mention them unless relevant.)"
        )

    parts.append(
        "\n## Guidelines\n"
        "- Be concise. Plain text only. No markdown, no emoji.\n"
        "- Use tools when needed; you may call multiple tools in one turn.\n"
        "- For long tasks, spawn subagents to parallelize work.\n"
        "- Before EVERY tool call, use the say() tool to briefly state what you are about to do.\n"
        "- Write files using relative paths (e.g. report.md), never absolute paths.\n"
    )

    return "\n".join(parts)


def _trim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove oldest messages if total character length is too large.

    Drops whole turns (from one user message up to the next) to avoid leaving
    orphaned tool/assistant messages at the start of the history.
    """
    def _char_len(msgs: list) -> int:
        total = 0
        for m in msgs:
            c = m.get("content")
            if isinstance(c, list):
                total += sum(len(str(b)) for b in c)
            else:
                total += len(str(c or ""))
        return total

    if _char_len(history) <= MAX_HISTORY_TOKENS_ESTIMATE * 4:
        return history

    trimmed = list(history)
    while len(trimmed) > 1 and _char_len(trimmed) > MAX_HISTORY_TOKENS_ESTIMATE * 4:
        # Find the first user message and skip everything up to (and including) the
        # entire turn that starts with it. A turn starts at a user message and ends
        # just before the next user message.
        first_user = next(
            (i for i, m in enumerate(trimmed) if m.get("role") == "user"), None
        )
        if first_user is None:
            # No user message found — drop the first message as last resort
            trimmed = trimmed[1:]
            break
        # Find the start of the NEXT turn (next user message after first_user)
        next_turn_start = next(
            (i for i, m in enumerate(trimmed) if i > first_user and m.get("role") == "user"),
            None,
        )
        if next_turn_start is None:
            # Only one turn remains — keep it regardless of size
            break
        trimmed = trimmed[next_turn_start:]

    # Ensure we never start with a non-user role (orphaned assistant/tool messages)
    while trimmed and trimmed[0].get("role") not in ("user", "system"):
        trimmed = trimmed[1:]

    return trimmed


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix structural issues in the message list that cause API 400 errors.

    - Merge consecutive user messages (GLM rejects them)
    - Remove orphaned tool results (no preceding assistant with tool_calls)
    - Remove empty-content user messages
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")

        # Drop user messages with empty string content (GLM rejects content: "")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, str) and not content.strip():
                continue

        # Merge consecutive user messages into one
        if role == "user" and result and result[-1].get("role") == "user":
            prev = result[-1]
            prev_c = prev.get("content", "")
            curr_c = msg.get("content", "")
            if isinstance(prev_c, str) and isinstance(curr_c, str):
                prev["content"] = f"{prev_c}\n{curr_c}".strip()
                continue
            # Mixed types (one is list) — just keep both; merging is unsafe
            result.append(msg)
            continue

        # Drop orphaned tool results: must be preceded by assistant with tool_calls
        if role == "tool":
            # Walk back to find the previous non-tool message
            prev_non_tool = next(
                (m for m in reversed(result) if m.get("role") != "tool"), None
            )
            if not prev_non_tool or not prev_non_tool.get("tool_calls"):
                continue

        result.append(msg)

    return result


def build_user_message(content: str, attachments: list | None = None) -> dict[str, Any]:
    """Build user message, with optional image attachments."""
    import base64
    images = [a for a in (attachments or []) if a.kind in ("image",)]
    if not images:
        return {"role": "user", "content": content}

    parts: list[dict] = []
    if content:
        parts.append({"type": "text", "text": content})
    for att in images:
        if att.url:
            parts.append({"type": "image_url", "image_url": {"url": att.url}})
        elif att.data and att.mime_type:
            b64 = base64.b64encode(att.data).decode()
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{att.mime_type};base64,{b64}"},
            })
    return {"role": "user", "content": parts}
