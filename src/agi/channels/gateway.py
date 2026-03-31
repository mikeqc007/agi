from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from aiohttp import web

from agi.channels.dispatcher import GatewayDispatcher

logger = logging.getLogger(__name__)


class GatewayChannel:
    """Lightweight HTTP gateway.

    Exposes:
      GET  /health
      GET  /v1/models
      POST /v1/chat/completions
      POST /v1/messages
    """

    def __init__(
        self,
        app_runtime: Any,
        dispatcher: GatewayDispatcher,
        host: str = "0.0.0.0",
        port: int = 8090,
        api_key: str = "",
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
        self._web_app.router.add_get("/", self._handle_root)
        self._web_app.router.add_get("/health", self._handle_health)
        self._web_app.router.add_get("/v1/models", self._handle_models)
        self._web_app.router.add_post("/v1/chat/completions", self._handle_chat)
        self._web_app.router.add_post("/v1/messages", self._handle_messages)

        self._runner = web.AppRunner(self._web_app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info("Gateway running at http://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler: Any) -> web.StreamResponse:
        if self._api_key and request.path.startswith("/v1"):
            auth = request.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if token != self._api_key:
                return web.json_response(
                    {"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
                    status=401,
                )
        return await handler(request)

    async def _handle_root(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "agi gateway"})

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "ts": int(time.time())})

    async def _handle_models(self, request: web.Request) -> web.Response:
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

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        model = str(body.get("model", self._runtime.cfg.agents[0].id))
        messages: list[dict[str, Any]] = body.get("messages", [])
        stream = bool(body.get("stream", False))
        user_id = str(body.get("user", "openai-api-user"))
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"

        if not messages:
            return web.json_response({"error": "No messages provided"}, status=400)

        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not last_user:
            return web.json_response({"error": "No user message found"}, status=400)

        content = self._extract_openai_content(last_user.get("content", ""))

        # Build prior history (stateless semantics: remove last user message by index)
        last_user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        prior = [
            m for i, m in enumerate(messages)
            if i != last_user_idx and m.get("role") != "system"
        ]
        extra_meta: dict[str, Any] = {"prepopulate_history": prior}

        if stream:
            return await self._stream_openai_chat(
                request=request,
                completion_id=completion_id,
                model=model,
                channel="openai_api",
                peer_id=user_id,
                content=content,
                agent_id=model,
                extra_meta=extra_meta,
            )

        reply = await self._run_agent(
            channel="openai_api",
            peer_kind="direct",
            peer_id=user_id,
            sender=user_id,
            content=content,
            agent_id=model,
            extra_meta=extra_meta,
        )
        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        content = str(body.get("content", "")).strip()
        if not content:
            return web.json_response({"error": "content is required"}, status=400)

        channel = str(body.get("channel", "gateway"))
        peer_kind = str(body.get("peer_kind", "direct"))
        peer_id = str(body.get("peer_id", "gateway-user"))
        sender = str(body.get("sender", peer_id))
        account_id = body.get("account_id")
        thread_id = body.get("thread_id")
        agent_id = str(body.get("agent_id", self._runtime.cfg.agents[0].id))
        session_key = body.get("session_key")
        stream = bool(body.get("stream", False))

        if stream:
            return await self._stream_gateway_messages(
                request=request,
                channel=channel,
                peer_kind=peer_kind,
                peer_id=peer_id,
                sender=sender,
                account_id=account_id,
                thread_id=thread_id,
                content=content,
                agent_id=agent_id,
                session_key=str(session_key) if session_key else None,
            )

        reply = await self._run_agent(
            channel=channel,
            peer_kind=peer_kind,
            peer_id=peer_id,
            sender=sender,
            content=content,
            account_id=account_id,
            thread_id=thread_id,
            agent_id=agent_id,
            session_key=str(session_key) if session_key else None,
        )
        return web.json_response({"reply": reply, "agent_id": agent_id})

    async def _stream_openai_chat(
        self,
        request: web.Request,
        completion_id: str,
        model: str,
        channel: str,
        peer_id: str,
        content: str,
        agent_id: str,
        extra_meta: dict | None = None,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_chunk(chunk: str) -> None:
            queue.put_nowait(chunk)

        async def _producer() -> None:
            try:
                await self._run_agent(
                    channel=channel,
                    peer_kind="direct",
                    peer_id=peer_id,
                    sender=peer_id,
                    content=content,
                    agent_id=agent_id,
                    on_text=on_chunk,
                    extra_meta=extra_meta,
                )
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_producer())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            await response.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
        finally:
            task.cancel()
        return response

    async def _stream_gateway_messages(
        self,
        request: web.Request,
        channel: str,
        peer_kind: str,
        peer_id: str,
        sender: str,
        account_id: Any,
        thread_id: Any,
        content: str,
        agent_id: str,
        session_key: str | None,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_chunk(chunk: str) -> None:
            queue.put_nowait(chunk)

        async def _producer() -> None:
            try:
                await self._run_agent(
                    channel=channel,
                    peer_kind=peer_kind,
                    peer_id=peer_id,
                    sender=sender,
                    content=content,
                    account_id=str(account_id) if account_id is not None else None,
                    thread_id=str(thread_id) if thread_id is not None else None,
                    agent_id=agent_id,
                    session_key=session_key,
                    on_text=on_chunk,
                )
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_producer())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                await response.write(f"data: {json.dumps({'delta': chunk})}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
        finally:
            task.cancel()
        return response

    async def _run_agent(
        self,
        channel: str,
        peer_kind: str,
        peer_id: str,
        sender: str,
        content: str,
        agent_id: str,
        account_id: str | None = None,
        thread_id: str | None = None,
        session_key: str | None = None,
        on_text: Any = None,
        extra_meta: dict | None = None,
    ) -> str:
        meta: dict[str, Any] = {}
        if on_text:
            meta["on_text"] = on_text
        if extra_meta:
            meta.update(extra_meta)
        return await self._dispatcher.submit_message(
            channel=channel,
            peer_kind=peer_kind,
            peer_id=peer_id,
            sender=sender,
            content=content,
            agent_id=agent_id,
            account_id=account_id,
            thread_id=thread_id,
            session_key=session_key,
            metadata=meta,
        )

    @staticmethod
    def _extract_openai_content(content: Any) -> str:
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
            return " ".join(chunks).strip()
        return str(content or "")
