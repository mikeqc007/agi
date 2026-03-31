from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode
from telegram.error import TelegramError

from agi.channels.dispatcher import GatewayDispatcher
from agi.config import TelegramConfig
from agi.types import Attachment, InboundMessage

logger = logging.getLogger(__name__)

MAX_MESSAGE_LEN = 4096
CHUNK_OVERLAP_LINES = 1


class TelegramChannel:
    def __init__(self, cfg: TelegramConfig, dispatcher: GatewayDispatcher) -> None:
        self._cfg = cfg
        self._dispatcher = dispatcher
        self._app: Application | None = None
        self._default_agent = cfg.default_agent_id

    async def start(self) -> None:
        if not self._cfg.token:
            logger.warning("Telegram token not set — channel disabled")
            return

        self._app = (
            Application.builder()
            .token(self._cfg.token)
            .build()
        )
        app = self._app

        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("agent", self._cmd_agent))
        app.add_handler(CommandHandler("clear", self._cmd_clear))
        app.add_handler(MessageHandler(
            filters.TEXT | filters.PHOTO | filters.VOICE | filters.Document.ALL,
            self._handle_message,
        ))

        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send_text(
        self,
        peer_id: str,
        text: str,
        thread_id: str | None = None,
        parse_mode: str | None = None,
    ) -> None:
        if not self._app:
            return
        chunks = _split_message(text)
        for chunk in chunks:
            try:
                kwargs: dict[str, Any] = {"chat_id": peer_id, "text": chunk}
                if thread_id:
                    kwargs["message_thread_id"] = int(thread_id)
                if parse_mode:
                    kwargs["parse_mode"] = parse_mode
                await self._app.bot.send_message(**kwargs)
            except TelegramError as e:
                # Retry without parse_mode if formatting failed
                if parse_mode and "can't parse" in str(e).lower():
                    try:
                        kwargs.pop("parse_mode", None)
                        await self._app.bot.send_message(**kwargs)
                    except TelegramError:
                        pass
                else:
                    logger.warning("Telegram send error: %s", e)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "👋 agi ready.\n"
            "Just send a message to start.\n"
            "/help for commands, /agent <id> to switch agents."
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Commands:\n"
            "/start — show welcome\n"
            "/agent <id> — switch agent\n"
            "/clear — clear conversation history\n"
            "\nJust send text, images, or voice messages."
        )

    async def _cmd_agent(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        parts = (update.message.text or "").split(maxsplit=1)
        if len(parts) < 2:
            await update.message.reply_text("Usage: /agent <agent_id>")
            return
        agent_id = parts[1].strip()
        # Store per-chat so different chats/groups can have different agents
        ctx.chat_data["agent_id"] = agent_id
        await update.message.reply_text(f"Switched to agent: {agent_id}")

    async def _cmd_clear(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if not user:
            return
        chat = update.effective_chat
        if not chat:
            return
        peer_id = str(chat.id)
        is_group = chat.type in ("group", "supergroup")
        peer_kind = "group" if is_group else "direct"
        agent_id = ctx.chat_data.get("agent_id", self._default_agent)
        _msg = getattr(update, "effective_message", None)
        thread_id = str(_msg.message_thread_id) if _msg and getattr(_msg, "message_thread_id", None) else None
        session_key = f"{agent_id}:telegram:{peer_kind}:{peer_id}"
        if thread_id:
            session_key = f"{session_key}:thread:{thread_id}"
        try:
            await self._dispatcher.submit_inbound(InboundMessage(
                id=-1, channel="telegram",
                peer_kind=peer_kind, peer_id=peer_id,
                sender="system",
                content="__clear_history__",
                created_at_ms=int(time.time() * 1000),
                metadata={"force_session_key": session_key, "clear_history": True},
            ))
        except Exception:
            pass
        await update.message.reply_text("History cleared.")

    async def _handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if not msg:
            return

        user = update.effective_user
        if not user:
            return

        # Auth check
        if self._cfg.allowed_users and user.id not in self._cfg.allowed_users:
            await msg.reply_text("Not authorized.")
            return

        peer_id = str(update.effective_chat.id)
        is_group = update.effective_chat.type in ("group", "supergroup")
        agent_id = ctx.chat_data.get("agent_id", self._default_agent)

        # Thread support
        thread_id = str(msg.message_thread_id) if msg.message_thread_id else None

        # Mention detection
        was_mentioned = False
        text_content = msg.text or msg.caption or ""
        if is_group:
            bot_username = (await self._app.bot.get_me()).username
            was_mentioned = f"@{bot_username}" in text_content
            text_content = text_content.replace(f"@{bot_username}", "").strip()

        # Collect attachments
        attachments: list[Attachment] = []

        if msg.photo:
            largest = max(msg.photo, key=lambda p: p.file_size or 0)
            f = await largest.get_file()
            data = await f.download_as_bytearray()
            attachments.append(Attachment(kind="image", data=bytes(data), mime_type="image/jpeg"))

        if msg.voice:
            f = await msg.voice.get_file()
            data = await f.download_as_bytearray()
            audio_bytes = bytes(data)
            mime = msg.voice.mime_type or "audio/ogg"
            # Transcribe with Whisper if available
            whisper = getattr(self, "_app_runtime", None)
            whisper = getattr(whisper, "whisper", None) if whisper else None
            if whisper:
                try:
                    transcript = await whisper.transcribe(audio_bytes, mime)
                    if transcript:
                        text_content = (text_content + " " + transcript).strip()
                except Exception as e:
                    logger.debug("Whisper transcription failed: %s", e)
            else:
                attachments.append(Attachment(kind="audio", data=audio_bytes, mime_type=mime))

        if msg.document:
            f = await msg.document.get_file()
            data = await f.download_as_bytearray()
            attachments.append(Attachment(
                kind="file", data=bytes(data),
                mime_type=msg.document.mime_type,
                filename=msg.document.file_name,
            ))

        # Streaming: send placeholder message, edit it as chunks arrive
        streamer = _TelegramStreamer(self._app.bot, peer_id, thread_id)

        inbound = InboundMessage(
            id=msg.message_id,
            channel="telegram",
            peer_kind="group" if is_group else "direct",
            peer_id=peer_id,
            sender=str(user.id),
            content=text_content,
            created_at_ms=int((msg.date.timestamp() if msg.date else time.time()) * 1000),
            account_id=str(user.id),
            thread_id=thread_id,
            is_group=is_group,
            was_mentioned=was_mentioned,
            attachments=attachments,
            metadata={"force_agent_id": agent_id, "on_text": streamer.on_chunk},
        )

        try:
            await self._app.bot.send_chat_action(chat_id=peer_id, action="typing")
        except Exception:
            pass

        try:
            reply = await self._dispatcher.submit_inbound(inbound)
            # Finalize streamed message (or send fresh if streaming wasn't used)
            await streamer.finalize(reply)
            # TTS: send voice message if configured
            await self._maybe_send_voice(reply, peer_id, thread_id)
        except Exception as e:
            logger.error("Error handling message: %s", e)
            await msg.reply_text(f"Error: {e}")

    async def _maybe_send_voice(
        self, text: str, peer_id: str, thread_id: str | None
    ) -> None:
        """Generate TTS audio and send as voice message if TTS is enabled."""
        runtime = getattr(self, "_app_runtime", None)
        tts = getattr(runtime, "tts", None) if runtime else None
        if not tts or not text:
            return
        try:
            from agi.providers.tts import parse_tts_directives
            cleaned_text, overrides = parse_tts_directives(text)
            speak_text = overrides.get("tts_text") or cleaned_text
            if not speak_text:
                return
            audio = await tts.speak(
                speak_text,
                override_provider=overrides.get("provider", ""),
                override_voice=overrides.get("voice", ""),
                override_model=overrides.get("model", ""),
            )
            if audio:
                import io
                kwargs: dict[str, Any] = {
                    "chat_id": peer_id,
                    "voice": io.BytesIO(audio),
                }
                if thread_id:
                    kwargs["message_thread_id"] = int(thread_id)
                await self._app.bot.send_voice(**kwargs)
        except Exception as e:
            logger.warning("TTS voice send failed: %s", e)


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

class _TelegramStreamer:
    """Accumulates text chunks and edits a Telegram message progressively.

    Rate-limits edits to ~1 per 2 seconds to avoid Telegram flood limits.
    """
    EDIT_INTERVAL = 2.0     # seconds between edits
    MIN_CHUNK_LEN = 20      # don't edit for tiny chunks

    def __init__(self, bot: Any, chat_id: str, thread_id: str | None) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._thread_id = thread_id
        self._sent_msg: Any = None
        self._buffer = ""
        self._last_edit = 0.0
        self._lock = asyncio.Lock()

    @staticmethod
    def _strip_ansi(text: str) -> str:
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def on_chunk(self, chunk: str) -> None:
        """Sync callback called from agent loop for each text chunk."""
        asyncio.create_task(self._handle_chunk(chunk))

    async def _handle_chunk(self, chunk: str) -> None:
        async with self._lock:
            self._buffer += self._strip_ansi(chunk)
            now = time.time()

            if self._sent_msg is None and len(self._buffer) >= self.MIN_CHUNK_LEN:
                # Send the first message
                try:
                    kwargs: dict[str, Any] = {
                        "chat_id": self._chat_id,
                        "text": self._buffer + " ▌",
                    }
                    if self._thread_id:
                        kwargs["message_thread_id"] = int(self._thread_id)
                    self._sent_msg = await self._bot.send_message(**kwargs)
                    self._last_edit = now
                except Exception:
                    pass

            elif self._sent_msg and (now - self._last_edit) >= self.EDIT_INTERVAL:
                # Edit existing message
                await self._try_edit(self._buffer + " ▌")
                self._last_edit = now

    async def finalize(self, full_text: str) -> None:
        """Called after agent loop completes. Send or update final message."""
        async with self._lock:
            text = full_text if full_text and full_text not in (
                "(duplicate)", "(dropped)", "(no response)"
            ) else self._buffer

            if not text:
                return

            if self._sent_msg:
                # Remove cursor and set final text
                await self._try_edit(text)
            else:
                # No streaming happened — send normally (possibly chunked)
                for chunk in _split_message(text):
                    try:
                        kwargs: dict[str, Any] = {"chat_id": self._chat_id, "text": chunk}
                        if self._thread_id:
                            kwargs["message_thread_id"] = int(self._thread_id)
                        await self._bot.send_message(**kwargs)
                    except Exception as e:
                        logger.warning("Telegram send error: %s", e)

    async def _try_edit(self, text: str) -> None:
        if not self._sent_msg:
            return
        # Truncate to Telegram limit
        display = text[:MAX_MESSAGE_LEN]
        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._sent_msg.message_id,
                text=display,
            )
        except Exception:
            pass  # message unchanged or deleted — ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_message(text: str) -> list[str]:
    """Split text into chunks ≤ MAX_MESSAGE_LEN chars, breaking at paragraphs."""
    if len(text) <= MAX_MESSAGE_LEN:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > MAX_MESSAGE_LEN:
        # Find best break point
        limit = MAX_MESSAGE_LEN
        break_pos = remaining.rfind("\n\n", 0, limit)
        if break_pos < limit // 2:
            break_pos = remaining.rfind("\n", 0, limit)
        if break_pos < limit // 2:
            break_pos = remaining.rfind(". ", 0, limit)
        if break_pos < limit // 2:
            break_pos = limit

        chunk = remaining[:break_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[break_pos + 1:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks
