from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from agi.agent.loop import AgentLoop
from agi.channels.dispatcher import GatewayDispatcher
from agi.channels.gateway import GatewayChannel
from agi.agent.sessions import SessionStore
from agi.agent.subagents import SubagentManager
from agi.channels.discord import DiscordChannel
from agi.channels.openai_api import OpenAIApiChannel
from agi.channels.telegram import TelegramChannel
from agi.config import AppConfig
from agi.cron.scheduler import CronService
from agi.hooks.manager import HookManager, make_event, trigger_hook
from agi.memory.store import MemoryManager
from agi.providers.tts import TtsProvider
from agi.providers.whisper import WhisperProvider
from agi.queue.queue import MessageQueue
from agi.storage.db import open_db
from agi.tools.mcp import MCPManager
from agi.tools.skills import SkillManager
from agi.types import InboundMessage

# Import all tools to register them
import agi.tools.shell          # noqa: F401
import agi.tools.fs             # noqa: F401
import agi.tools.web            # noqa: F401
import agi.tools.memory_tool    # noqa: F401
import agi.tools.subagents_tool # noqa: F401
import agi.tools.computer       # noqa: F401
import agi.tools.browser_tool   # noqa: F401
import agi.cron.cron_tool       # noqa: F401
import agi.tools.skills         # noqa: F401
import agi.tools.say            # noqa: F401
import agi.tools.todo           # noqa: F401

logger = logging.getLogger(__name__)


class AppRuntime:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.db: Any = None
        self.session_store: SessionStore | None = None
        self.memory_manager: MemoryManager | None = None
        self.mcp_manager: MCPManager | None = None
        self.skill_manager: SkillManager | None = None
        self.subagent_manager: SubagentManager | None = None
        self.cron_service: CronService | None = None
        self.whisper: WhisperProvider | None = None
        self.tts: TtsProvider | None = None
        self.hook_manager: HookManager | None = None
        self._queue: MessageQueue | None = None
        self._loop: AgentLoop | None = None
        self._telegram: TelegramChannel | None = None
        self._discord: DiscordChannel | None = None
        self._openai_api: OpenAIApiChannel | None = None
        self._gateway: GatewayChannel | None = None
        self._cli: Any = None
        self._memory_sync_task: asyncio.Task | None = None
        self.dispatcher = GatewayDispatcher(self.submit_internal)

    async def start(self) -> None:
        # Database
        self.db = await open_db(self.cfg.resolved_db_path())
        logger.info("Database opened: %s", self.cfg.resolved_db_path())

        # Core services
        self.session_store = SessionStore(self.db)
        self.session_store.start_reaper(
            interval_hours=self.cfg.session_reap_interval_hours,
            max_age_days=self.cfg.session_max_age_days,
        )

        self.memory_manager = MemoryManager(self.db, self.cfg.memory)

        self.mcp_manager = MCPManager()
        for srv in self.cfg.mcp.servers:
            self.mcp_manager.add_server(srv.name, srv.command, srv.args, srv.env or None)
        await self.mcp_manager.start_all()

        self.skill_manager = SkillManager(self.cfg.resolved_skills_dir())

        self.subagent_manager = SubagentManager(
            submit_fn=self.submit_internal,
            max_concurrent=self.cfg.max_subagent_concurrent,
            default_timeout=self.cfg.subagent_timeout_seconds,
        )

        self._queue = MessageQueue(
            process_fn=self._process_message,
            mode=self.cfg.queue.mode,
            cap=self.cfg.queue.cap,
        )

        self._loop = AgentLoop(self)

        self.cron_service = CronService(self.db, self.submit_internal,
                                        notify_fn=self._cron_notify,
                                        on_text_fn=self._cron_on_text)
        await self.cron_service.start()

        # Whisper
        whisper_cfg = getattr(self.cfg, "whisper", None)
        if whisper_cfg is not False:
            self.whisper = WhisperProvider(
                backend=getattr(whisper_cfg, "backend", "auto") if whisper_cfg else "auto",
                model=getattr(whisper_cfg, "model", "base") if whisper_cfg else "base",
                api_key=self.cfg.keys.get("GROQ_API_KEY", ""),
            )

        # TTS
        tts_cfg = self.cfg.tts
        if tts_cfg.provider not in ("none", ""):
            self.tts = TtsProvider(
                provider=tts_cfg.provider,
                voice=tts_cfg.voice,
                api_key=tts_cfg.api_key,
                model=tts_cfg.model,
            )

        # Hooks — load user scripts, fire startup event
        self.hook_manager = HookManager()
        await self.hook_manager.startup(self)

        # Telegram
        self._telegram = TelegramChannel(self.cfg.telegram, self.dispatcher)
        self._telegram._app_runtime = self   # give TelegramChannel access to whisper
        await self._telegram.start()

        # Discord
        self._discord = DiscordChannel(self.cfg.discord, self.dispatcher)
        await self._discord.start()

        # OpenAI-compatible API (opt-in via config)
        if self.cfg.openai_api.enabled:
            self._openai_api = OpenAIApiChannel(
                app_runtime=self,
                dispatcher=self.dispatcher,
                host=self.cfg.openai_api.host,
                port=self.cfg.openai_api.port,
                api_key=self.cfg.openai_api.api_key,
            )
            await self._openai_api.start()

        # Unified HTTP gateway (opt-in via config)
        if self.cfg.gateway.enabled:
            self._gateway = GatewayChannel(
                app_runtime=self,
                dispatcher=self.dispatcher,
                host=self.cfg.gateway.host,
                port=self.cfg.gateway.port,
                api_key=self.cfg.gateway.api_key,
            )
            await self._gateway.start()

        # CLI channel (started separately via --cli flag; not auto-started here)

        # Memory file sync — index existing .md files on startup, then watch periodically
        if self.cfg.memory.flush_enabled and self.cfg.config_dir:
            self._ensure_memory_dirs()
            await self._sync_memory_files()
            self._memory_sync_task = asyncio.ensure_future(self._memory_sync_loop())

        logger.info("agi started with %d agent(s)", len(self.cfg.agents))

    async def start_cli(self, agent_id: str = "default") -> None:
        from agi.channels.cli import CLIChannel
        self._cli = CLIChannel(self.dispatcher, agent_id, db=self.db)
        await self._cli.start()

    def _ensure_memory_dirs(self) -> None:
        """Create the persistent runtime state directory tree under config_dir/state/."""
        from pathlib import Path
        state_root = Path(self.cfg.config_dir) / self.cfg.memory.memory_dir
        # Top-level scopes
        for d in ("global", "users", "chats", "threads"):
            (state_root / d).mkdir(parents=True, exist_ok=True)
        # Per-agent directories
        for agent in self.cfg.agents:
            agent_dir = state_root / "agents" / agent.id
            for sub in ("kb", "memory", "skills"):
                (agent_dir / sub).mkdir(parents=True, exist_ok=True)
            # Create AGENT.md template if missing
            agent_md = agent_dir / "AGENT.md"
            if not agent_md.exists():
                agent_md.write_text(
                    f"# {agent.name or agent.id}\n\n"
                    "<!-- Describe this agent's persona, goals, and constraints here.\n"
                    "     This file is injected directly into the system prompt. -->\n",
                    encoding="utf-8",
                )
        logger.info("Runtime state directory tree ready at %s", state_root)

    async def _sync_memory_files(self) -> None:
        """Index all memory .md files across all three tiers."""
        from pathlib import Path
        from agi.memory.file_sync import sync_memory_root
        state_root = Path(self.cfg.config_dir) / self.cfg.memory.memory_dir
        try:
            await sync_memory_root(self.db, state_root, self.cfg.agents, self.cfg.memory)
        except Exception as e:
            logger.warning("Memory file sync failed: %s", e)

    async def _memory_sync_loop(self) -> None:
        """Background task: re-index memory files every 5 minutes (handles user edits)."""
        while True:
            try:
                await asyncio.sleep(300)
                await self._sync_memory_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Memory sync loop error: %s", e)

    async def stop(self) -> None:
        if self._memory_sync_task:
            self._memory_sync_task.cancel()
        if self.hook_manager:
            await self.hook_manager.shutdown()
        if self._cli:
            await self._cli.stop()
        if self._telegram:
            await self._telegram.stop()
        if self._discord:
            await self._discord.stop()
        if self._openai_api:
            await self._openai_api.stop()
        if self._gateway:
            await self._gateway.stop()
        if self.cron_service:
            await self.cron_service.stop()
        if self.mcp_manager:
            await self.mcp_manager.stop_all()
        if self.db:
            await self.db.close()
        logger.info("agi stopped")

    def _cron_on_text(self, notify: str):
        """Return the streaming on_text callable for the given notify spec."""
        import sys
        if notify == "cli":
            if self._cli:
                return self._cli.print_chunk
            return sys.stdout.write
        return None

    async def _cron_notify(self, target: str, reply: str) -> None:
        """Dispatch cron result to a single target (called for both output= and notify=).

        Target formats:
          "log:path/to/file.log"  → append timestamped reply to file  (output=)
          "cli"                   → print to CLI terminal              (notify=)
          "telegram:CHAT_ID"      → send via Telegram bot              (notify=)
          "discord:CHANNEL_ID"    → send via Discord channel           (notify=)
        """
        notify = target
        import sys
        from datetime import datetime

        if not reply:
            return

        if notify == "cli":
            # Streaming already handled via on_text; print a newline to close the line
            output = "\n"
            if self._cli:
                self._cli.print_chunk(output)
            else:
                sys.stdout.write(output)
                sys.stdout.flush()

        elif notify.startswith("log:") or ("/" in notify or notify.endswith(".log")):
            raw_path = notify[4:].strip() if notify.startswith("log:") else notify.strip()
            path = Path(raw_path).expanduser()
            if not path.is_absolute() and self.cfg.config_dir:
                path = Path(self.cfg.config_dir) / path
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}]\n{reply}\n\n")
            logger.info("Cron notify wrote to %s", path)

        elif notify.startswith("telegram:"):
            chat_id = notify[9:].strip()
            if self._telegram:
                await self._telegram.send_text(chat_id, reply)
            else:
                logger.warning("Cron notify telegram: Telegram channel not running")

        elif notify.startswith("discord:"):
            channel_id = notify[8:].strip()
            if self._discord:
                await self._discord.send_text(channel_id, reply)
            else:
                logger.warning("Cron notify discord: Discord channel not running")

    def _default_agent_for(self, channel: str) -> str:
        """Return the channel-specific default agent id, or fallback to first agent."""
        if channel == "telegram":
            return self.cfg.telegram.default_agent_id or self.cfg.agents[0].id
        if channel == "discord":
            return self.cfg.discord.default_agent_id or self.cfg.agents[0].id
        return self.cfg.agents[0].id

    async def submit_internal(self, msg: InboundMessage) -> str:
        """Entry point for all messages (from channels, cron, subagents)."""
        meta = msg.metadata or {}

        # Handle special commands
        if meta.get("clear_history"):
            if meta.get("force_session_key"):
                key = meta["force_session_key"]
            else:
                agent_id = str(meta.get("force_agent_id") or self._default_agent_for(msg.channel))
                key = f"{agent_id}:{msg.channel}:{msg.peer_kind}:{msg.peer_id}"
                if msg.thread_id:
                    key = f"{key}:thread:{msg.thread_id}"
            if self.session_store:
                await self.session_store.replace_history(key, [])
            return ""

        # Resolve agent
        agent_id = str(meta.get("force_agent_id") or
                       self._default_agent_for(msg.channel) or
                       self.cfg.agents[0].id)

        # Build session key
        if force_key := meta.get("force_session_key"):
            session_key = str(force_key)
        else:
            session_key = f"{agent_id}:{msg.channel}:{msg.peer_kind}:{msg.peer_id}"
            if msg.thread_id:
                session_key = f"{session_key}:thread:{msg.thread_id}"

        # Hook: message:received (after session_key is resolved)
        await trigger_hook(make_event(
            "message", "received",
            session_key=session_key,
            content=msg.content,
            channel=msg.channel,
            sender=msg.sender,
        ))

        # Get/create session
        session = await self.session_store.get_or_create(
            key=session_key,
            agent_id=agent_id,
            channel=msg.channel,
            peer_kind=msg.peer_kind,
            peer_id=msg.peer_id,
            account_id=msg.account_id,
            thread_id=msg.thread_id,
        )

        # Apply meta patches (from subagent spawner)
        if patch := meta.get("session_meta_patch"):
            await self.session_store.patch_meta(session_key, patch)
            session = await self.session_store.get(session_key)

        # Reset stale session history if max_age exceeded (used by cron)
        if max_age_hours := meta.get("session_max_age_hours"):
            age_s = (time.time() * 1000 - (session.updated_at_ms or 0)) / 1000
            if session.history and age_s > max_age_hours * 3600:
                await self.session_store.replace_history(session_key, [])
                session = await self.session_store.get(session_key)
                logger.info("Cron session %s history reset (age %.1fh)", session_key, age_s / 3600)

        # Enqueue (handles concurrency + drop/collect)
        return await self._queue.enqueue(session_key, msg, session.meta)

    async def _process_message(self, msg: InboundMessage) -> str:
        """Called by queue — runs the agent loop under session lock."""
        meta = msg.metadata or {}
        agent_id = str(meta.get("force_agent_id") or
                       self._default_agent_for(msg.channel) or
                       self.cfg.agents[0].id)

        if force_key := meta.get("force_session_key"):
            session_key = str(force_key)
        else:
            session_key = f"{agent_id}:{msg.channel}:{msg.peer_kind}:{msg.peer_id}"
            if msg.thread_id:
                session_key = f"{session_key}:thread:{msg.thread_id}"

        agent_cfg = self.cfg.agent(agent_id)

        # Session lock ensures only one coroutine processes this session at a time
        async with self.session_store.lock(session_key):
            session = await self.session_store.get(session_key)
            if session is None:
                session = await self.session_store.get_or_create(
                    session_key, agent_id, msg.channel,
                    msg.peer_kind, msg.peer_id, msg.account_id, msg.thread_id,
                )

            # Stateless API: caller provides full history; overwrite session history
            if self.session_store and "prepopulate_history" in meta:
                prepop = meta["prepopulate_history"]
                await self.session_store.replace_history(session_key, prepop)
                session = await self.session_store.get(session_key) or session

            await trigger_hook(make_event(
                "agent", "start",
                session_key=session_key,
                agent_id=agent_id,
                channel=msg.channel,
                peer_id=msg.peer_id,
            ))
            try:
                reply = await self._loop.run(msg, session, agent_cfg)
            except Exception as e:
                logger.error("Agent loop error for session %s: %s", session_key, e)
                await trigger_hook(make_event(
                    "agent", "end",
                    session_key=session_key,
                    agent_id=agent_id,
                    channel=msg.channel,
                    peer_id=msg.peer_id,
                    success=False,
                    error=str(e),
                ))
                return f"Error: {e}"
            await trigger_hook(make_event(
                "agent", "end",
                session_key=session_key,
                agent_id=agent_id,
                channel=msg.channel,
                peer_id=msg.peer_id,
                success=True,
            ))

            # Hook: message:sent
            await trigger_hook(make_event(
                "message", "sent",
                session_key=session_key,
                content=reply,
                channel=msg.channel,
                peer_id=msg.peer_id,
                success=True,
            ))
            return reply
