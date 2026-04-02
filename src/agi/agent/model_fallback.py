from __future__ import annotations

import asyncio
import json
import logging
import os
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# HTTP status codes that warrant trying the next model
RETRYABLE_CODES = {429, 500, 502, 503, 504}


def normalize_messages_for_model(model: str, messages: list[dict]) -> list[dict]:
    """Provider-specific message normalization for stricter OpenAI-compatible endpoints.

    Ollama is handled via a separate direct-HTTP path and is never passed here.
    For all other providers (GLM, OpenRouter, Gemini, etc.):
      - assistant + tool_calls with content=None → content="" (providers reject null/absent)
    GLM-specific additional fixes applied only when model starts with "openai/glm-".
    """
    is_glm = model.startswith("openai/glm-")

    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue

        m = dict(msg)
        role = m.get("role")

        # Most non-Ollama providers reject assistant tool-call messages with
        # null or absent content. (Ollama is the opposite — it requires null.)
        if role == "assistant" and m.get("tool_calls") and m.get("content") is None:
            m["content"] = ""

        # GLM-specific: tool result messages must not carry a "name" field,
        # and content must always be a non-null string.
        if is_glm and role == "tool":
            m.pop("name", None)
            content = m.get("content")
            if content is None:
                m["content"] = ""
            elif not isinstance(content, str):
                m["content"] = json.dumps(content, ensure_ascii=False)

        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Ollama direct helpers (bypasses litellm to avoid thinking-model parse bug)
# ---------------------------------------------------------------------------

def _think_params(model: str, think_level: str) -> dict:
    """Return provider-specific extra kwargs for thinking/reasoning mode."""
    if not think_level or think_level == "off":
        return {}
    BUDGETS = {"minimal": 1024, "low": 4096, "medium": 8192, "high": 16000}
    budget = BUDGETS.get(think_level, 4096)
    if model.startswith("openrouter/"):
        effort = think_level if think_level != "minimal" else "low"
        return {"extra_body": {"reasoning": {"effort": effort}}}
    if "anthropic" in model or model.startswith("claude"):
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if model.startswith("gemini/"):
        level_map = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
        return {"extra_body": {"thinkingConfig": {"thinkingMode": level_map.get(think_level, "LOW")}}}
    return {}


def _is_ollama(model: str) -> bool:
    return model.startswith("ollama/") or model.startswith("ollama_chat/")


def _ollama_name(model: str) -> str:
    return model.split("/", 1)[-1]


def _ollama_base() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _parse_ollama_response(data: dict) -> Any:
    """Parse Ollama /v1/chat/completions JSON into a litellm-compatible object."""
    choice = data["choices"][0]
    msg = choice["message"]

    # Build tool_calls list
    tc_list = None
    if msg.get("tool_calls"):
        tc_list = []
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args)
            tc_list.append(SimpleNamespace(
                id=tc.get("id", "call_0"),
                type="function",
                function=SimpleNamespace(name=fn.get("name", ""), arguments=args),
            ))

    usage_data = data.get("usage", {})
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=msg.get("content") or "",
                tool_calls=tc_list,
                role="assistant",
            ),
            finish_reason=choice.get("finish_reason", "stop"),
        )],
        usage=SimpleNamespace(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        ),
    )


async def ollama_complete(
    model_name: str,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
    think: bool = False,
) -> Any:
    """Call Ollama /v1/chat/completions directly (non-streaming)."""
    import httpx

    payload: dict[str, Any] = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        think=think,
    )
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{_ollama_base()}/v1/chat/completions", json=payload)
        if not resp.is_success:
            logger.error("Ollama %d error body: %s", resp.status_code, resp.text[:800])
        resp.raise_for_status()
        return _parse_ollama_response(resp.json())


async def ollama_stream(
    model_name: str,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
    on_text: Any,
    think: bool = False,
) -> tuple[str, list, str, Any]:
    """Stream from Ollama /v1/chat/completions directly.
    Returns (full_text, tool_calls, stop_reason, usage).
    """
    import httpx

    payload: dict[str, Any] = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        think=think,
    )
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    text_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    stop_reason = "stop"
    usage = None

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST", f"{_ollama_base()}/v1/chat/completions", json=payload
        ) as resp:
            if not resp.is_success:
                body = await resp.aread()
                logger.error("Ollama stream %d error body: %s", resp.status_code, body.decode()[:800])
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})

                # Text content
                text = delta.get("content") or ""
                if text:
                    text_parts.append(text)
                    try:
                        on_text(text)
                    except Exception:
                        pass

                # Tool call chunks
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                    if tc.get("id"):
                        tool_calls_acc[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls_acc[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls_acc[idx]["args"] += fn["arguments"]

                if choice.get("finish_reason"):
                    stop_reason = choice["finish_reason"]

                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage = SimpleNamespace(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                    )

    tool_calls_raw = [
        SimpleNamespace(
            id=v["id"] or f"call_{i}",
            function=SimpleNamespace(name=v["name"], arguments=v["args"]),
        )
        for i, v in sorted(tool_calls_acc.items())
    ]
    # Strip <think>...</think> blocks that qwen3 leaks even with think=False
    import re as _re
    full_text = _re.sub(r'<think>.*?</think>', '', "".join(text_parts), flags=_re.DOTALL).strip()
    return full_text, tool_calls_raw, stop_reason, usage


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def complete_with_fallback(
    models: list[str],
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    extra_params: dict | None = None,
    think_level: str = "off",
    on_status: Any = None,
) -> Any:
    """Try models in order, falling back on rate-limit or server errors."""
    import litellm

    last_exc: Exception | None = None

    for i, model in enumerate(models):
        try:
            if _is_ollama(model):
                response = await ollama_complete(
                    _ollama_name(model), messages, tools, temperature, max_tokens,
                    think=(think_level != "off"),
                )
            else:
                normalized_messages = normalize_messages_for_model(model, messages)
                kwargs: dict[str, Any] = dict(
                    model=model,
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if extra_params:
                    kwargs.update(extra_params)
                kwargs.update(_think_params(model, think_level))
                response = await litellm.acompletion(**kwargs)

            if i > 0:
                logger.info("Fell back to model %s after %d failures", model, i)
            return response

        except Exception as e:
            last_exc = e
            code = _extract_status_code(e)
            is_context_exceeded = "ContextWindowExceeded" in type(e).__name__ or ("context" in str(e).lower() and "length" in str(e).lower())
            # Special case: max_tokens too large but input fits — retry same model with reduced max_tokens
            if is_context_exceeded and "max_tokens" in str(e).lower():
                import re as _re
                _m_input = _re.search(r"has (\d+) input tokens", str(e))
                _m_limit = _re.search(r"context length is (\d+)", str(e))
                if _m_input and _m_limit:
                    _avail = int(_m_limit.group(1)) - int(_m_input.group(1)) - 100
                    if _avail > 200:
                        max_tokens = _avail
                        continue  # retry same model index (i stays the same via while-style continue)
            if code not in RETRYABLE_CODES and code != 0 and not is_context_exceeded:
                raise
            if i < len(models) - 1:
                msg = f"[{model}] failed (code={code}): {str(e)[:120]} — trying next model"
                logger.warning(msg)
                if on_status:
                    on_status(f"\033[90m{msg}\033[0m\n")
                await asyncio.sleep(0.5)
            else:
                msg = f"All models failed. Last error: {e}"
                logger.error(msg)
                if on_status:
                    on_status(f"\033[91m{msg}\033[0m\n")

    raise last_exc or RuntimeError("All models failed")


def _extract_status_code(exc: Exception) -> int:
    # litellm raises various exception types with status_code attribute
    for attr in ("status_code", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg:
        return 429
    if "500" in msg:
        return 500
    if "503" in msg or "unavailable" in msg:
        return 503
    return 0
