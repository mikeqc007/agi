from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Compact when estimated token count exceeds this fraction of model max_tokens
COMPACTION_THRESHOLD = 0.75
# Always keep this many recent messages intact
KEEP_RECENT = 12
# Minimum messages to trigger compaction (no point compacting 4 messages)
MIN_MESSAGES_TO_COMPACT = 20


def needs_compaction(history: list[dict], max_tokens: int, threshold: float = COMPACTION_THRESHOLD) -> bool:
    if len(history) < MIN_MESSAGES_TO_COMPACT:
        return False
    estimated = _estimate_tokens(history)
    return estimated > max_tokens * threshold


async def compact(
    history: list[dict[str, Any]],
    model: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Summarize old messages and replace with a single summary message."""
    if len(history) <= KEEP_RECENT:
        return history

    old = history[:-KEEP_RECENT]
    recent = _clean_recent(history[-KEEP_RECENT:])

    summary = await _summarize(old, model, max_tokens)
    logger.info("Compacted %d messages into summary (%d chars)", len(old), len(summary))

    summary_msg = {
        "role": "user",
        "content": f"[Earlier conversation summary]\n{summary}",
    }
    # Add a minimal assistant ack so the history stays well-formed
    ack_msg = {"role": "assistant", "content": "Understood, continuing from there."}

    return [summary_msg, ack_msg] + recent


async def _summarize(messages: list[dict], model: str, max_tokens: int) -> str:
    try:
        from agi.agent.model_fallback import complete_with_fallback

        summary_prompt = (
            "Summarize the following conversation history concisely, "
            "preserving all important facts, decisions, and context:\n\n"
        )
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            summary_prompt += f"{role.upper()}: {content}\n"

        response = await complete_with_fallback(
            models=[model],
            messages=[{"role": "user", "content": "/no_think\n" + summary_prompt}],
            max_tokens=min(1024, max_tokens // 4),
        )
        return response.choices[0].message.content or "(no summary)"
    except Exception as e:
        logger.warning("Compaction summarize failed: %s", e)
        # Fallback: just truncate to text
        lines = []
        for m in messages[-10:]:
            role = m.get("role", "")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            lines.append(f"{role}: {str(content)[:200]}")
        return "\n".join(lines)


def _clean_recent(messages: list[dict]) -> list[dict]:
    """Ensure recent messages start at a clean boundary (user or standalone assistant).
    Avoids starting with a tool message that has no preceding assistant+tool_calls."""
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == "user":
            return messages[i:]
        if role == "assistant" and not msg.get("tool_calls"):
            return messages[i:]
    return messages


def _estimate_tokens(messages: list[dict]) -> int:
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    return total_chars // 4
