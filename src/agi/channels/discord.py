from __future__ import annotations

"""Discord channel.

Setup (5 minutes):
  1. Go to https://discord.com/developers/applications
  2. New Application → Bot → Reset Token → copy token
  3. OAuth2 → URL Generator → bot + Send Messages + Read Message History
  4. Open generated URL → invite bot to your server
  5. Set DISCORD_TOKEN env var or discord.token in config
"""

import asyncio
import logging
import time
from typing import Any

from agi.channels.dispatcher import GatewayDispatcher
from agi.config import DiscordConfig
from agi.types import Attachment, InboundMessage

logger = logging.getLogger(__name__)

MAX_MESSAGE_LEN = 2000       # Discord's limit
STREAM_EDIT_INTERVAL = 2.0   # seconds between streaming edits


class DiscordChannel:
    def __init__(self, cfg: DiscordConfig, dispatcher: GatewayDispatcher) -> None:
        self._cfg = cfg
        self._dispatcher = dispatcher
        self._client: Any = None
        self._default_agent = cfg.default_agent_id
        self._channel_agents: dict[str, str] = {}  # per-channel agent override

    async def start(self) -> None:
        if not self._cfg.token:
            logger.warning("Discord token not set — channel disabled")
            return
        try:
            import discord
        except ImportError:
            logger.error("discord.py not installed. Run: pip install discord.py")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            logger.info("Discord bot logged in as %s", self._client.user)

        @self._client.event
        async def on_message(message: Any) -> None:
            if message.author == self._client.user:
                return
            await self._handle_message(message)

        asyncio.create_task(self._client.start(self._cfg.token))

    async def stop(self) -> None:
        if self._client:
            await self._client.close()

    async def send_text(self, channel_id: str, text: str) -> None:
        if not self._client:
            return
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))
            for chunk in _split_message(text):
                await channel.send(chunk)
        except Exception as e:
            logger.warning("Discord send_text error: %s", e)

    async def _handle_message(self, message: Any) -> None:
        import discord

        # Auth check
        if self._cfg.allowed_users and message.author.id not in self._cfg.allowed_users:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_thread = isinstance(message.channel, discord.Thread)
        is_group = not is_dm
        # For threads: peer_id = parent channel, thread_id = thread channel
        peer_id = str(message.channel.parent_id if is_thread else message.channel.id)
        thread_id = str(message.channel.id) if is_thread else None
        agent_id = self._channel_agents.get(str(message.channel.id), self._default_agent)

        # Mention detection — required in servers unless it's a DM
        was_mentioned = self._client.user in message.mentions
        if is_group and self._cfg.require_mention and not was_mentioned:
            return

        # Clean content
        content = message.content or ""
        if self._client.user:
            content = content.replace(f"<@{self._client.user.id}>", "").strip()

        # Handle slash-style commands
        if content.startswith("/"):
            await self._handle_command(content, message)
            return

        # Collect attachments
        attachments: list[Attachment] = []
        for att in message.attachments:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(att.url) as resp:
                        data = await resp.read()
                kind = "image" if att.content_type and "image" in att.content_type else "file"
                attachments.append(Attachment(
                    kind=kind,
                    data=data,
                    url=att.url,
                    mime_type=att.content_type,
                    filename=att.filename,
                ))
            except Exception as e:
                logger.debug("Failed to download attachment: %s", e)

        # Whisper transcription for audio
        whisper = getattr(self, "_app_runtime", None)
        whisper = getattr(whisper, "whisper", None) if whisper else None

        # Streaming: send placeholder, edit as chunks come
        streamer = _DiscordStreamer(message.channel)

        inbound = InboundMessage(
            id=message.id,
            channel="discord",
            peer_kind="group" if is_group else "direct",
            peer_id=peer_id,
            sender=str(message.author.id),
            content=content,
            created_at_ms=int(time.time() * 1000),
            account_id=str(message.author.id),
            thread_id=thread_id,
            is_group=is_group,
            was_mentioned=was_mentioned,
            attachments=attachments,
            metadata={
                "force_agent_id": agent_id,
                "on_text": streamer.on_chunk,
            },
        )

        # Show typing indicator
        async with message.channel.typing():
            try:
                reply = await self._dispatcher.submit_inbound(inbound)
                await streamer.finalize(reply)
            except Exception as e:
                logger.error("Discord message error: %s", e)
                await message.channel.send(f"Error: {e}")

    async def _handle_command(self, cmd: str, message: Any) -> None:
        parts = cmd.split(maxsplit=1)
        name = parts[0].lower()

        if name == "/clear":
            import discord as _discord
            is_dm = isinstance(message.channel, _discord.DMChannel)
            is_thread = isinstance(message.channel, _discord.Thread)
            peer_kind = "direct" if is_dm else "group"
            channel_id = str(message.channel.id)
            peer_id = str(message.channel.parent_id if is_thread else message.channel.id)
            thread_id = channel_id if is_thread else None
            agent_id = self._channel_agents.get(channel_id, self._default_agent)
            session_key = f"{agent_id}:discord:{peer_kind}:{peer_id}"
            if thread_id:
                session_key = f"{session_key}:thread:{thread_id}"
            await self._dispatcher.submit_inbound(InboundMessage(
                id=-1, channel="discord",
                peer_kind=peer_kind, peer_id=peer_id,
                sender="system", content="__clear_history__",
                created_at_ms=int(time.time() * 1000),
                thread_id=thread_id,
                metadata={"clear_history": True, "force_session_key": session_key, "force_agent_id": agent_id},
            ))
            await message.channel.send("History cleared.")

        elif name == "/agent" and len(parts) > 1:
            new_agent = parts[1].strip()
            self._channel_agents[str(message.channel.id)] = new_agent
            await message.channel.send(f"Switched to agent: {new_agent}")

        elif name == "/help":
            await message.channel.send(
                "**agi commands**\n"
                "`/clear` — clear conversation history\n"
                "`/agent <id>` — switch agent\n"
                "`/help` — this message"
            )


class _DiscordStreamer:
    """Progressive message editing for Discord streaming."""

    def __init__(self, channel: Any) -> None:
        self._channel = channel
        self._sent_msg: Any = None
        self._buffer = ""
        self._last_edit = 0.0
        self._lock = asyncio.Lock()

    def on_chunk(self, chunk: str) -> None:
        asyncio.create_task(self._handle_chunk(chunk))

    async def _handle_chunk(self, chunk: str) -> None:
        async with self._lock:
            self._buffer += chunk
            now = time.time()

            if self._sent_msg is None and len(self._buffer) >= 20:
                try:
                    self._sent_msg = await self._channel.send(self._buffer + " ▌")
                    self._last_edit = now
                except Exception:
                    pass
            elif self._sent_msg and (now - self._last_edit) >= STREAM_EDIT_INTERVAL:
                await self._try_edit(self._buffer + " ▌")
                self._last_edit = now

    async def finalize(self, full_text: str) -> None:
        async with self._lock:
            text = full_text if full_text and full_text not in (
                "(duplicate)", "(dropped)", "(no response)"
            ) else self._buffer

            if not text:
                return

            if self._sent_msg:
                await self._try_edit(text)
            else:
                for chunk in _split_message(text):
                    try:
                        await self._channel.send(chunk)
                    except Exception as e:
                        logger.warning("Discord send error: %s", e)

    async def _try_edit(self, text: str) -> None:
        if not self._sent_msg:
            return
        try:
            await self._sent_msg.edit(content=text[:MAX_MESSAGE_LEN])
        except Exception:
            pass


def _split_message(text: str) -> list[str]:
    if len(text) <= MAX_MESSAGE_LEN:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > MAX_MESSAGE_LEN:
        pos = remaining.rfind("\n\n", 0, MAX_MESSAGE_LEN)
        if pos < MAX_MESSAGE_LEN // 2:
            pos = remaining.rfind("\n", 0, MAX_MESSAGE_LEN)
        if pos < MAX_MESSAGE_LEN // 2:
            pos = MAX_MESSAGE_LEN
        chunks.append(remaining[:pos + 1].strip())
        remaining = remaining[pos + 1:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
