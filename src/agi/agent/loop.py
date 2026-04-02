from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from typing import Any

from agi.agent.compaction import needs_compaction, compact
from agi.agent.memory_flush import should_run_memory_flush, run_memory_flush
from agi.agent.context import build_messages, build_user_message
from agi.agent.model_fallback import complete_with_fallback, _is_ollama, normalize_messages_for_model
from agi.agent.permissions import is_allowed_by_mode, needs_prompt, get_tool_level
from agi.config import AgentConfig
from agi.hooks.manager import make_event, trigger_hook
from agi.storage.db import usage_record
from agi.tools.registry import dispatch, get_all_schemas
from agi.types import InboundMessage, SessionRecord, ToolContext

logger = logging.getLogger(__name__)

DEAD_LOOP_THRESHOLD = 4
MAX_TOOL_OUTPUT_CHARS = 8000
# Max image dimension sent to vision model (resize if larger)
_IMAGE_MAX_PX = 1280


class _ImageResult:
    """Tool result that carries image data for vision models (e.g. browser screenshot)."""
    def __init__(self, text: str, base64_data: str, mime_type: str) -> None:
        self.text = text
        self.base64_data = base64_data
        self.mime_type = mime_type


def _tool_error_text(result: Any) -> str | None:
    """Return error text when tool output clearly indicates failure."""
    if isinstance(result, dict):
        err = result.get("error")
        return str(err) if err else None
    text = str(result or "").strip()
    if not text:
        return None
    low = text.lower()
    markers = (
        "工具调用失败",
        "error calling mcp tool",
        "tool loop detected",
        "not permitted",
    )
    if any(m in low for m in markers):
        return text
    if low.startswith("error:") or low.startswith("error "):
        return text
    return None


def _looks_like_fabricated_tool_output(text: str) -> bool:
    s = (text or "").lower()
    markers = (
        "<tool_response>",
        "</tool_response>",
        '"name": "',
        '"tool": "',
        '"arguments":',
    )
    return any(m in s for m in markers)


class AgentLoop:
    def __init__(self, app: Any) -> None:
        self._app = app

    async def run(
        self,
        msg: InboundMessage,
        session: SessionRecord,
        agent_cfg: AgentConfig,
    ) -> str:
        app = self._app
        _run_start_ms = int(time.time() * 1000)

        # Streaming callback — set by channel in msg.metadata
        on_text = msg.metadata.get("on_text")  # callable(chunk: str) | None

        # Build tool context
        ctx = ToolContext(
            agent_id=session.agent_id,
            session_key=session.session_key,
            channel=session.channel,
            peer_kind=session.peer_kind,
            peer_id=session.peer_id,
            sender=msg.sender,
            account_id=session.account_id,
            thread_id=session.thread_id,
            app=app,
            on_text=on_text,
        )

        # Resolve model list (primary + fallbacks)
        model_override = session.meta.get("model_override")
        model_cfg = agent_cfg.model
        primary = str(model_override or model_cfg.primary)
        models = [primary] + [m for m in model_cfg.fallbacks if m != primary]
        is_ollama_model = _is_ollama(primary)
        think_level = str(session.meta.get("think_level") or agent_cfg.think_level)

        # Filter tools by profile (built-in + MCP)
        # Merge per-message overrides (e.g. tools_deny from cron) on top of session.meta
        mcp_schemas = app.mcp_manager.get_all_schemas() if app.mcp_manager else []
        msg_meta = msg.metadata or {}
        effective_meta = {**session.meta}
        if "tools_deny" in msg_meta:
            effective_meta["tools_deny"] = list(msg_meta["tools_deny"])
        tools = _filter_tools(get_all_schemas() + mcp_schemas, agent_cfg, effective_meta)

        # Memory context
        memory_ctx = ""
        if agent_cfg.memory_enabled:
            try:
                memory_ctx = await app.memory_manager.build_context(
                    agent_cfg.id,
                    user_id=msg.sender or session.peer_id,
                    chat_id=session.peer_id,
                    thread_id=session.thread_id or "",
                    query=msg.content,
                    top_k=6,
                )
            except Exception as e:
                logger.debug("Memory recall failed: %s", e)

        # AGENT.md — agent persona file, injected directly into system prompt
        agent_md = ""
        if app.cfg.config_dir:
            from pathlib import Path as _Path
            _agent_md_path = (
                _Path(app.cfg.config_dir) / app.cfg.memory.memory_dir
                / "agents" / agent_cfg.id / "AGENT.md"
            )
            if _agent_md_path.exists():
                try:
                    agent_md = _agent_md_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        # Skill context — openclaw style: list skills with paths so model reads SKILL.md via read_file
        skill_ctx = ""
        skill_mgr = getattr(app, "skill_manager", None)
        if skill_mgr:
            try:
                agent_skills_dir = app.cfg.resolved_agent_skills_dir(agent_cfg.id) if app.cfg.config_dir else None
                skill_ctx = skill_mgr.build_prompt(agent_dir=agent_skills_dir)
            except Exception:
                pass

        # Skip LLM call for empty messages (GLM rejects content: "")
        raw_content = (msg.content or "").strip()
        if not raw_content and not msg.attachments:
            return ""

        # Fire message:start event
        asyncio.ensure_future(trigger_hook(make_event(
            "message", "start",
            session_key=session.session_key,
            agent_id=session.agent_id,
            channel=session.channel,
            content=msg.content,
        )))

        user_msg = build_user_message(msg.content, msg.attachments)
        history = list(session.history)

        # Compaction check — memory flush before compacting (openclaw style)
        if needs_compaction(history, model_cfg.max_tokens, agent_cfg.compaction_threshold):
            if agent_cfg.memory_enabled and app.memory_manager and app.cfg.memory.flush_enabled:
                if should_run_memory_flush(history, model_cfg.max_tokens):
                    config_dir = app.cfg.config_dir
                    _flush_user_id = msg.sender or session.peer_id
                    await run_memory_flush(
                        history, models, model_cfg.max_tokens,
                        config_dir, app.cfg.memory.memory_dir, agent_cfg.id,
                        user_id=_flush_user_id,
                        peer_kind=session.peer_kind,
                        chat_id=session.peer_id,
                        thread_id=session.thread_id or "",
                        custom_prompt=app.cfg.memory.flush_prompt,
                    )
                    # Re-index the user's flush dir
                    from pathlib import Path
                    from agi.memory.file_sync import sync_user_flush
                    state_root = Path(config_dir) / app.cfg.memory.memory_dir
                    await sync_user_flush(app.db, state_root, _flush_user_id, app.cfg.memory)
            history = await compact(history, primary, model_cfg.max_tokens)
            logger.info("Compacted history for session %s", session.session_key)

        working: list[dict[str, Any]] = []
        call_counter: Counter = Counter()
        tool_failures: list[str] = []
        tool_success_count = 0
        final_text = ""
        intermediate_nudge_count = 0

        for iteration in range(agent_cfg.max_iterations):
            # iteration 0: history + working + user_msg (appended last by build_messages)
            # iteration N: history + user_msg + working (user_msg before tool calls/results)
            if iteration == 0:
                messages = build_messages(agent_cfg, history + working, user_msg, memory_ctx, skill_ctx, agent_md)
            else:
                messages = build_messages(agent_cfg, history + [user_msg] + working, None, memory_ctx, skill_ctx, agent_md)

            # Use streaming when on_text callback is present (CLI/Telegram)
            if on_text is not None:
                on_text("\033[90m[思考中...]\033[0m\n")
                text, tool_calls_raw, stop_reason, usage = await _stream_complete(
                    models, messages, tools, model_cfg, on_text, think_level=think_level
                )
            else:
                response = await complete_with_fallback(
                    models=models,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=model_cfg.temperature,
                    max_tokens=model_cfg.max_tokens,
                    think_level=think_level,
                )
                choice = response.choices[0]
                rmsg = choice.message
                text = rmsg.content or ""
                tool_calls_raw = rmsg.tool_calls or []
                stop_reason = choice.finish_reason or "stop"
                usage = response.usage

            if text:
                final_text = text

            # Record usage
            if usage:
                asyncio.ensure_future(usage_record(
                    app.db, agent_cfg.id, session.session_key, primary,
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                ))

            # Build assistant message.
            # When tool_calls are present, content must be None — not "" and especially not
            # any "thinking" text the model leaked alongside the tool call. Ollama rejects
            # assistant messages that have both non-empty content and tool_calls.
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": None if (tool_calls_raw and is_ollama_model) else (text or "")}
            if tool_calls_raw:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls_raw
                ]
            working.append(assistant_msg)

            # Fallback: some models (e.g. qwen3-vl) write tool calls as plain text JSON
            # instead of using the function calling API. Detect and execute them.
            text_fallback_used = False
            if not tool_calls_raw and text:
                valid_names = {s.get("function", {}).get("name") for s in tools}
                parsed = _try_parse_text_tool_call(text, valid_names)
                if parsed:
                    name, args = parsed
                    fake_id = f"text_call_{iteration}"
                    tool_calls_raw = [_ToolCallProxy(fake_id, name, json.dumps(args))]
                    text_fallback_used = True
                    # Replace assistant message with proper tool_calls format
                    working[-1] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": fake_id, "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}],
                    }

            # Drop tool calls that are missing required parameters (e.g. browser with no action).
            # Keeping them would cause Ollama to return 400 on the *next* iteration because
            # it validates tool_calls in message history against the schema.
            _schema_map = {s["function"]["name"]: s["function"] for s in tools}
            filtered_tcs = []
            for _tc in tool_calls_raw:
                _raw_args = _tc.function.arguments
                _tc_args: dict[str, Any]
                if isinstance(_raw_args, dict):
                    _tc_args = _raw_args
                elif isinstance(_raw_args, str):
                    if _raw_args.strip():
                        try:
                            parsed = json.loads(_raw_args)
                            _tc_args = parsed if isinstance(parsed, dict) else {}
                        except Exception:
                            _tc_args = {}
                    else:
                        _tc_args = {}
                else:
                    _tc_args = {}
                _schema = _schema_map.get(_tc.function.name, {})
                _required = _schema.get("parameters", {}).get("required", [])
                if any(r not in _tc_args for r in _required):
                    logger.warning("Dropping tool call %r — missing required args %s (got %s)",
                                   _tc.function.name, _required, list(_tc_args.keys()))
                    continue
                # Normalize arguments so history always contains valid JSON object text.
                filtered_tcs.append(_ToolCallProxy(
                    _tc.id,
                    _tc.function.name,
                    json.dumps(_tc_args, ensure_ascii=False),
                ))
            tool_calls_raw = filtered_tcs

            if not tool_calls_raw:
                # All tool calls were invalid and dropped. Remove the assistant message
                # we already appended to working — it contains bad tool_calls that would
                # cause Ollama 400 ("invalid tool call arguments") in any subsequent call.
                if working and working[-1].get("role") == "assistant" and working[-1].get("tool_calls"):
                    working.pop()
                break
            else:
                # Keep history consistent with validated calls only; otherwise
                # dropped invalid calls may still remain in `working[-1]` and
                # cause Ollama to reject subsequent requests with 400.
                if working and working[-1].get("role") == "assistant":
                    working[-1]["content"] = None
                    working[-1]["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls_raw
                    ]

            # Execute tools concurrently
            permission_mode = getattr(agent_cfg, "permission_mode", "allow")
            tool_infos = []
            for tc in tool_calls_raw:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                loop_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                call_counter[loop_key] += 1
                dead_loop = call_counter[loop_key] >= DEAD_LOOP_THRESHOLD
                denied = not _is_allowed(name, agent_cfg, session.meta)

                # Prompt mode: ask user before dangerous tools (CLI only)
                if not denied and needs_prompt(name, permission_mode) and session.channel == "cli":
                    brief = {k: v for k, v in args.items()
                             if isinstance(v, (str, int, float, bool)) and len(str(v)) < 120}
                    args_str = ", ".join(f"{k}={v!r}" for k, v in brief.items())
                    if on_text:
                        on_text(f"\n\033[93m[权限确认] {name}({args_str})\n允许执行? [y/N] \033[0m")
                    try:
                        answer = await asyncio.get_event_loop().run_in_executor(None, input)
                        if answer.strip().lower() not in ("y", "yes", "是"):
                            denied = True
                    except Exception:
                        denied = True

                tool_infos.append((tc.id, name, args, dead_loop, denied))

            # Notify user of tool calls in progress
            if on_text:
                # Print on a new line if the model output text before the tool call
                prefix = "\n" if text else ""
                for _, name, args, dead, denied in tool_infos:
                    if name == "say":
                        continue  # say tool displays via ctx.on_text directly
                    # Show key args inline (skip large/binary values)
                    brief = {k: v for k, v in args.items()
                             if isinstance(v, (str, int, float, bool)) and len(str(v)) < 80}
                    args_str = ", ".join(f"{k}={v!r}" for k, v in brief.items())
                    status = " [denied]" if denied else (" [loop]" if dead else "")
                    on_text(f"{prefix}\033[90m[{name}({args_str}){status}]\033[0m\n")
                    prefix = ""

            results = await asyncio.gather(
                *[_exec_tool(ctx, tid, name, args, dead, denied)
                  for tid, name, args, dead, denied in tool_infos],
                return_exceptions=True,
            )

            pending_images: list[_ImageResult] = []
            for (tid, name, args, _, _), res in zip(tool_infos, results):
                if isinstance(res, Exception):
                    res = {"error": str(res)}
                if isinstance(res, _ImageResult):
                    tool_success_count += 1
                    if on_text:
                        on_text(f"\033[90m[{name} → 截图 {res.mime_type}]\033[0m\n")
                    # Most providers (Ollama, GLM, etc.) don't support multimodal content
                    # in role=tool messages. Always use text-only tool result + synthetic
                    # user message with image so any vision model can see the screenshot.
                    working.append(_tool_result(tid, name, res.text))
                    pending_images.append(res)
                else:
                    err = _tool_error_text(res)
                    if err:
                        tool_failures.append(f"{name}: {err}")
                        if on_text:
                            on_text(f"\033[91m[{name} 失败: {err[:120]}]\033[0m\n")
                    else:
                        tool_success_count += 1
                    working.append(_tool_result(tid, name, res))

            # Inject screenshots after tool results (role order: tool... → user).
            # If a separate vision_model is configured, call it to describe the image
            # as text (works with non-vision primary models like GLM-4.7-Flash).
            # Otherwise, send the image directly for vision-capable primary models.
            vision_model = model_cfg.vision_model.strip()
            for img in pending_images:
                if vision_model:
                    # Describe image via vision model, inject description as plain text
                    desc = await _vision_describe(img, vision_model, model_cfg)
                    if on_text:
                        on_text(f"\033[90m[vision描述: {desc[:80]}...]\033[0m\n" if len(desc) > 80 else f"\033[90m[vision描述: {desc}]\033[0m\n")
                    working.append({"role": "user", "content": f"[截图内容]\n{desc}"})
                else:
                    working.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "[截图]"},
                            {"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.base64_data}"}},
                        ],
                    })

            if not text_fallback_used and stop_reason not in ("tool_calls", "tool_use"):
                # Don't break if the model output looks like an intermediate planning
                # message (not a real final answer) and we still have tool results —
                # this happens when a fallback model outputs "I will now..." text
                # instead of a tool call after receiving a browser/navigate result.
                _intermediate_markers = (
                    "我将", "我会", "i will", "i'll", "let me", "让我",
                    "接下来", "正在", "now i", "next i",
                )
                _is_intermediate = text and any(
                    m in text.lower() for m in _intermediate_markers
                )
                if not _is_intermediate or intermediate_nudge_count >= 2:
                    break
                # Remove the intermediate assistant message so the next iteration
                # doesn't include it as a fake final answer in history.
                if working and working[-1].get("role") == "assistant" and working[-1].get("content"):
                    working.pop()
                final_text = ""
                intermediate_nudge_count += 1

        # If the loop ended with tool results but no final text, do one more pass
        # without tools to force the model to summarize the results.
        has_tool_results = any(m.get("role") == "tool" for m in working)
        if has_tool_results and tool_failures and tool_success_count == 0:
            fail_lines = ["工具调用失败，无法可靠完成本次请求。"]
            for e in tool_failures[-3:]:
                fail_lines.append(f"- {e[:280]}")
            fail_msg = "\n".join(fail_lines)
            working.append({"role": "assistant", "content": fail_msg})
            full_history = history + [user_msg] + working
            await app.session_store.replace_history(session.session_key, full_history)
            return fail_msg
        if not final_text and has_tool_results:
            # Exclude old history to avoid contamination from previous turns;
            # only the current question + tool exchange is needed for the summary.
            messages = build_messages(agent_cfg, [user_msg] + working, None, memory_ctx, skill_ctx, agent_md)
            if on_text is not None:
                on_text("\033[90m[整理结果...]\033[0m\n")
                text, _, _, usage = await _stream_complete(models, messages, None, model_cfg, on_text, think_level=think_level)
            else:
                response = await complete_with_fallback(
                    models=models, messages=messages, tools=None,
                    temperature=model_cfg.temperature, max_tokens=model_cfg.max_tokens,
                    think_level=think_level,
                )
                choice = response.choices[0]
                text = choice.message.content or ""
                usage = response.usage
            if text:
                final_text = text
                working.append({"role": "assistant", "content": text})
            if usage:
                asyncio.ensure_future(usage_record(
                    app.db, agent_cfg.id, session.session_key, primary,
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                ))

        # If model produced neither text nor tool calls, do a plain-text retry.
        if not final_text and not has_tool_results:
            try:
                messages = build_messages(agent_cfg, history + [user_msg] + working, None, memory_ctx, skill_ctx, agent_md)
                response = await complete_with_fallback(
                    models=models,
                    messages=messages,
                    tools=None,
                    temperature=model_cfg.temperature,
                    max_tokens=model_cfg.max_tokens,
                    think_level=think_level,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    final_text = text
                    working.append({"role": "assistant", "content": text})
            except Exception:
                pass
            if not final_text:
                final_text = "我刚才没有拿到有效输出，请重试一次。"
                working.append({"role": "assistant", "content": final_text})

        # Guardrail: model sometimes fakes tool call blocks in plain text.
        # If no real tool result exists, do not trust pseudo tool outputs.
        if final_text and (not has_tool_results) and _looks_like_fabricated_tool_output(final_text):
            final_text = (
                "我刚才没有真正调用到工具（只是模型文本模拟），请重试并明确要求调用对应 MCP 工具。"
            )
            if working and working[-1].get("role") == "assistant":
                working[-1]["content"] = final_text
            else:
                working.append({"role": "assistant", "content": final_text})

        # Persist history
        full_history = history + [user_msg] + working
        await app.session_store.replace_history(session.session_key, full_history)

        _reply = final_text or "(no response)"

        # Fire message:stop event
        asyncio.ensure_future(trigger_hook(make_event(
            "message", "stop",
            session_key=session.session_key,
            agent_id=session.agent_id,
            channel=session.channel,
            duration_ms=int(time.time() * 1000) - _run_start_ms,
            reply=_reply,
        )))

        return _reply


async def _stream_complete(
    models: list[str],
    messages: list[dict],
    tools: list[dict] | None,
    model_cfg: Any,
    on_text: Any,
    think_level: str = "off",
) -> tuple[str, list, str, Any]:
    """Stream LLM response, calling on_text for each text chunk.
    Returns (full_text, tool_calls, stop_reason, usage).
    """
    import litellm

    from agi.agent.model_fallback import _is_ollama, _ollama_name, ollama_stream, _think_params

    last_exc: Exception | None = None
    for model in models:
        try:
            # Ollama thinking models: bypass litellm to avoid content-stripping bug
            if _is_ollama(model):
                return await ollama_stream(
                    _ollama_name(model), messages, tools,
                    model_cfg.temperature, model_cfg.max_tokens, on_text,
                    think=(think_level != "off"),
                )

            kwargs: dict[str, Any] = dict(
                model=model,
                messages=normalize_messages_for_model(model, messages),
                temperature=model_cfg.temperature,
                max_tokens=model_cfg.max_tokens,
                stream=True,
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            kwargs.update(_think_params(model, think_level))

            text_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}  # index -> {id, name, args}
            stop_reason = "stop"
            usage = None

            response = await litellm.acompletion(**kwargs)
            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = choice.delta

                # Text chunk
                if delta.content:
                    text_parts.append(delta.content)
                    try:
                        on_text(delta.content)
                    except Exception:
                        pass

                # Tool call chunks (assembled across multiple chunks)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": getattr(tc.function, "name", "") or "",
                                "args": "",
                            }
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if hasattr(tc.function, "name") and tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if hasattr(tc.function, "arguments") and tc.function.arguments:
                            tool_calls_acc[idx]["args"] += tc.function.arguments

                if choice.finish_reason:
                    stop_reason = choice.finish_reason

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage

            # Convert accumulated tool calls to litellm-compatible objects (sorted by stream index)
            tool_calls_raw = [
                _ToolCallProxy(v["id"], v["name"], v["args"])
                for _, v in sorted(tool_calls_acc.items())
            ] if tool_calls_acc else []

            return "".join(text_parts), tool_calls_raw, stop_reason, usage

        except Exception as e:
            last_exc = e
            from agi.agent.model_fallback import _extract_status_code, RETRYABLE_CODES
            if _extract_status_code(e) not in RETRYABLE_CODES:
                raise
            if model != models[-1]:
                logger.warning("Streaming model %s failed, trying next: %s", model, e)
                continue

    raise last_exc or RuntimeError("All models failed")


class _ToolCallProxy:
    """Minimal proxy to match litellm ToolCall interface."""
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _FunctionProxy(name, arguments)


class _FunctionProxy:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


async def _exec_tool(
    ctx: ToolContext,
    call_id: str,
    name: str,
    args: dict,
    dead_loop: bool,
    denied: bool,
) -> Any:
    if dead_loop:
        return {"error": "Tool loop detected — stop calling this tool."}
    if denied:
        return {"error": f"Tool '{name}' not permitted."}

    args_preview = json.dumps(args, ensure_ascii=False)
    if len(args_preview) > 300:
        args_preview = args_preview[:300] + "...(truncated)"
    logger.info(
        "Tool call session=%s tool=%s call_id=%s args=%s",
        ctx.session_key,
        name,
        call_id,
        args_preview,
    )

    try:
        await trigger_hook(make_event(
            "tool",
            "call",
            session_key=ctx.session_key,
            tool=name,
            args=args,
            call_id=call_id,
            agent_id=ctx.agent_id,
            channel=ctx.channel,
        ))
    except Exception:
        pass

    try:
        mcp_mgr = getattr(ctx.app, "mcp_manager", None)
        if mcp_mgr and mcp_mgr.is_mcp_tool(name):
            result = await mcp_mgr.call(name, args)
        else:
            result = await dispatch(name, ctx, args)
    except Exception as e:
        result = {"error": str(e)}

    ok = not (isinstance(result, dict) and "error" in result)
    logger.info(
        "Tool result session=%s tool=%s call_id=%s ok=%s",
        ctx.session_key,
        name,
        call_id,
        ok,
    )

    try:
        await trigger_hook(make_event(
            "tool",
            "result",
            session_key=ctx.session_key,
            tool=name,
            result=result,
            call_id=call_id,
            agent_id=ctx.agent_id,
            channel=ctx.channel,
            success=ok,
        ))
    except Exception:
        pass

    # Detect image result (browser screenshot etc.): preserve as _ImageResult
    if isinstance(result, dict) and result.get("ok") and result.get("base64") and result.get("mime_type"):
        b64 = str(result["base64"])
        mime = str(result["mime_type"])
        b64 = _resize_image_b64(b64, mime)
        text_info = {k: v for k, v in result.items() if k != "base64"}
        return _ImageResult(json.dumps(text_info, ensure_ascii=False), b64, mime)

    # Truncate large outputs
    result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if len(result_str) > MAX_TOOL_OUTPUT_CHARS:
        result_str = result_str[:MAX_TOOL_OUTPUT_CHARS] + f"\n[...truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"
    return result_str


def _tool_result(call_id: str, name: str, result: Any) -> dict:
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def _tool_result_with_image(call_id: str, name: str, res: _ImageResult) -> dict:
    """Build a multimodal tool result message carrying both text metadata and an image."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": [
            {"type": "text", "text": res.text},
            {"type": "image_url", "image_url": {"url": f"data:{res.mime_type};base64,{res.base64_data}"}},
        ],
    }


def _resize_image_b64(b64: str, mime_type: str) -> str:
    """Resize image if larger than _IMAGE_MAX_PX on either side. Returns base64."""
    try:
        import base64 as _b64mod
        from io import BytesIO
        from PIL import Image

        data = _b64mod.b64decode(b64)
        img = Image.open(BytesIO(data))
        if max(img.width, img.height) <= _IMAGE_MAX_PX:
            return b64
        img.thumbnail((_IMAGE_MAX_PX, _IMAGE_MAX_PX), Image.LANCZOS)
        buf = BytesIO()
        fmt = "JPEG" if "jpeg" in mime_type else "PNG"
        img.save(buf, format=fmt, quality=85)
        return _b64mod.b64encode(buf.getvalue()).decode()
    except Exception:
        return b64


def _filter_tools(schemas: list[dict], agent_cfg: AgentConfig, meta: dict) -> list[dict]:
    profile = str(meta.get("tool_profile") or agent_cfg.tool_profile)
    allow = list(meta.get("tools_allow") or agent_cfg.tools_allow)
    deny = list(meta.get("tools_deny") or agent_cfg.tools_deny)

    SAFE_TOOLS = {"web_search", "web_fetch", "remember", "recall", "forget",
                  "spawn_agent", "list_subagents", "cron_add", "cron_list", "cron_delete"}
    MINIMAL_TOOLS = {"remember", "recall", "spawn_agent"}

    filtered = []
    for s in schemas:
        name = s.get("function", {}).get("name", "")
        if deny and name in deny:
            continue
        if allow and name not in allow:
            continue
        if profile == "safe" and name not in SAFE_TOOLS:
            continue
        if profile == "minimal" and name not in MINIMAL_TOOLS:
            continue
        filtered.append(s)
    return filtered


def _try_parse_text_tool_call(text: str, valid_names: set) -> tuple[str, dict] | None:
    """Detect when a model outputs a tool call as plain JSON text instead of a function call."""
    import re
    text = text.strip()
    # Try to find JSON object in the text
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    raw = match.group()

    def _extract(data: dict) -> tuple[str, dict] | None:
        fn = data.get("function")
        name = (
            data.get("name") or
            data.get("tool") or
            (fn if isinstance(fn, str) else (fn or {}).get("name"))
        )
        args = data.get("arguments") or data.get("parameters") or data.get("input") or {}
        if name and name in valid_names and isinstance(args, dict):
            return name, args
        return None

    try:
        return _extract(json.loads(raw))
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    # Some models emit shell commands containing jq-style \( or \. sequences which
    # are not valid JSON escapes.  Fix them by double-escaping lone backslashes.
    try:
        fixed = re.sub(r'\\(?!["\\/bfnrtu0-9])', r'\\\\', raw)
        result = _extract(json.loads(fixed))
        if result:
            return result
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    return None


async def _vision_describe(img: _ImageResult, vision_model: str, model_cfg: Any) -> str:
    """Call a vision-capable model to describe a screenshot, returning plain text."""
    from agi.agent.model_fallback import complete_with_fallback, _is_ollama, ollama_complete, _ollama_name
    try:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "请详细描述这张截图的内容，包括页面标题、主要文字、布局和重要元素。"},
            {"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.base64_data}"}},
        ]}]
        if _is_ollama(vision_model):
            resp = await ollama_complete(_ollama_name(vision_model), messages, None,
                                         model_cfg.temperature, model_cfg.max_tokens)
        else:
            resp = await complete_with_fallback(
                models=[vision_model], messages=messages, tools=None,
                temperature=model_cfg.temperature, max_tokens=1024,
            )
        return resp.choices[0].message.content or "(无描述)"
    except Exception as e:
        logger.warning("Vision describe failed (%s): %s", vision_model, e)
        return f"(截图描述失败: {e})"


def _is_allowed(name: str, agent_cfg: AgentConfig, meta: dict) -> bool:
    deny = list(meta.get("tools_deny") or agent_cfg.tools_deny)
    allow = list(meta.get("tools_allow") or agent_cfg.tools_allow)
    if deny and name in deny:
        return False
    if allow and name not in allow:
        return False
    permission_mode = getattr(agent_cfg, "permission_mode", "allow")
    if not is_allowed_by_mode(name, permission_mode):
        return False
    return True
