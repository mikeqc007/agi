from __future__ import annotations

import time
import uuid
from typing import Any

from agi.types import InboundMessage


class GatewayDispatcher:
    """Single ingress dispatcher for all channel messages."""

    def __init__(self, submit_fn: Any) -> None:
        self._submit = submit_fn

    async def submit_inbound(self, inbound: InboundMessage) -> str:
        return await self._submit(inbound)

    async def submit_message(
        self,
        *,
        channel: str,
        peer_kind: str,
        peer_id: str,
        sender: str,
        content: str,
        agent_id: str,
        account_id: str | None = None,
        thread_id: str | None = None,
        session_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        meta: dict[str, Any] = {"force_agent_id": agent_id}
        if session_key:
            meta["force_session_key"] = session_key
        if metadata:
            meta.update(metadata)
        inbound = InboundMessage(
            id=uuid.uuid4().int & 0x7FFFFFFF,
            channel=channel,
            peer_kind=peer_kind,
            peer_id=peer_id,
            sender=sender,
            content=content,
            created_at_ms=int(time.time() * 1000),
            account_id=account_id,
            thread_id=thread_id,
            metadata=meta,
        )
        return await self.submit_inbound(inbound)
