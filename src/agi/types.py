from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Attachment:
    kind: str           # image / audio / video / file
    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None
    filename: str | None = None


@dataclass
class InboundMessage:
    id: int
    channel: str
    peer_kind: str      # direct / group
    peer_id: str
    sender: str
    content: str
    created_at_ms: int
    account_id: str | None = None
    thread_id: str | None = None
    is_group: bool = False
    was_mentioned: bool = False
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionRecord:
    session_key: str
    agent_id: str
    channel: str
    peer_kind: str
    peer_id: str
    history: list[dict[str, Any]]
    meta: dict[str, Any]
    account_id: str | None = None
    thread_id: str | None = None
    created_at_ms: int = 0
    updated_at_ms: int = 0


@dataclass
class MemoryEntry:
    id: int
    agent_id: str
    content: str
    embedding: list[float] | None
    score: float = 0.0
    peer_id: str = ""
    scope: str = "agent"
    source_file: str | None = None
    created_at_ms: int = 0
    updated_at_ms: int = 0


@dataclass
class ToolContext:
    agent_id: str
    session_key: str
    channel: str
    peer_kind: str
    peer_id: str
    app: Any                        # AppRuntime
    sender: str = ""                # sender/user identity (differs from peer_id in group chats)
    account_id: str | None = None
    thread_id: str | None = None
    on_text: Any = None             # streaming callback (callable(str)) if in CLI/streaming mode
