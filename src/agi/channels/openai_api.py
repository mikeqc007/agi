from __future__ import annotations

"""OpenAI-compatible API channel.

Exposes:
  GET  /v1/models                — list agents as models
  POST /v1/chat/completions      — chat with an agent (streaming SSE or JSON)

This lets any OpenAI-compatible frontend (Open WebUI, Chatbox, LobeChat,
Chatbot UI, etc.) connect to agi directly.

Usage in Open WebUI:
  Settings → Connections → OpenAI API
  Base URL: http://localhost:8080
  API Key:  (any string, or set api_key in config)
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from aiohttp import web

from agi.channels.dispatcher import GatewayDispatcher

logger = logging.getLogger(__name__)


class OpenAIApiChannel:
    def __init__(
        self,
        app_runtime: Any,
        dispatcher: GatewayDispatcher,
        host: str = "0.0.0.0",
        port: int = 8080,
        api_key: str = "",          # empty = no auth
    ) -> None:
        self._runtime = app_runtime
        self._dispatcher = dispatcher
        self._host = host
        self._port = port
        self._api_key = api_key
        self._web_app: web.Application | None = None
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        self._web_app = web.Application(middlewares=[self._auth_middleware])
        self._web_app.router.add_get("/v1/models", self._handle_models)
        self._web_app.router.add_post("/v1/chat/completions", self._handle_chat)
        self._web_app.router.add_get("/", self._handle_root)

        self._runner = web.AppRunner(self._web_app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info("OpenAI-compatible API running at http://%s:%d", self._host, self._port)
        logger.info("Connect Open WebUI → Base URL: http://localhost:%d  API Key: %s",
                    self._port, self._api_key or "(none)")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler: Any) -> web.Response:
        if self._api_key and request.path.startswith("/v1"):
            auth = request.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if token != self._api_key:
                return web.json_response(
                    {"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
                    status=401,
                )
        return await handler(request)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    async def _handle_root(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "agi OpenAI API"})

    async def _handle_models(self, request: web.Request) -> web.Response:
        """List all configured agents as OpenAI models."""
        models = [
            {
                "id": agent.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agi",
                "name": agent.name or agent.id,
            }
            for agent in self._runtime.cfg.agents
        ]
        return web.json_response({"object": "list", "data": models})

    async def _handle_chat(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        model = body.get("model", self._runtime.cfg.agents[0].id)
        messages: list[dict] = body.get("messages", [])
        stream: bool = body.get("stream", False)
        user_id = str(body.get("user", "openai-api-user"))

        if not messages:
            return web.json_response({"error": "No messages provided"}, status=400)

        # Extract the last user message
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user:
            return web.json_response({"error": "No user message found"}, status=400)

        content = last_user.get("content", "")
        if isinstance(content, list):
            # Multimodal: extract text parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"

        if stream:
            return await self._stream_response(
                request, model, user_id, content, messages, completion_id
            )
        else:
            return await self._sync_response(
                model, user_id, content, messages, completion_id
            )

    # ------------------------------------------------------------------
    # Response modes
    # ------------------------------------------------------------------

    async def _sync_response(
        self, model: str, user_id: str, content: str,
        messages: list[dict], completion_id: str,
    ) -> web.Response:
        reply = await self._run_agent(model, user_id, content, messages)
        return web.json_response({
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    async def _stream_response(
        self, request: web.Request, model: str, user_id: str, content: str,
        messages: list[dict], completion_id: str,
    ) -> web.StreamResponse:
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)

        # Streaming via queue
        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_chunk(chunk: str) -> None:
            chunk_queue.put_nowait(chunk)

        async def _producer() -> None:
            try:
                await self._run_agent(model, user_id, content, messages, on_text=on_chunk)
            finally:
                chunk_queue.put_nowait(None)  # sentinel

        producer_task = asyncio.create_task(_producer())

        # Send SSE chunks
        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    break
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }],
                }
                await response.write(f"data: {json.dumps(data)}\n\n".encode())

            # Final chunk
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            await response.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
        except Exception as e:
            logger.debug("Stream response error: %s", e)
        finally:
            producer_task.cancel()

        return response

    # ------------------------------------------------------------------
    # Agent invocation
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        agent_id: str,
        user_id: str,
        content: str,
        messages: list[dict],
        on_text: Any = None,
    ) -> str:
        extra_meta: dict[str, Any] = {}
        if on_text:
            extra_meta["on_text"] = on_text
        # Use provided messages as session history (stateless OpenAI-compatible semantics).
        # Remove the last user message; normalize the rest to safe internal format:
        # - only user/assistant roles (drop system/tool/function)
        last_user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        prior = [
            m for i, m in enumerate(messages)
            if i != last_user_idx and m.get("role") != "system"
        ]
        extra_meta["prepopulate_history"] = prior
        return await self._dispatcher.submit_message(
            channel="openai_api",
            peer_kind="direct",
            peer_id=user_id,
            sender=user_id,
            content=content,
            agent_id=agent_id,
            metadata=extra_meta,
        )
