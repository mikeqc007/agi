from __future__ import annotations

"""CLI channel — test the agent directly in your terminal.

Usage: agi run --cli
       agi run --cli --agent myagent
"""

import asyncio
import logging
import sys
import time
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from agi.channels.dispatcher import GatewayDispatcher
from agi.types import InboundMessage

logger = logging.getLogger(__name__)

PROMPT = ANSI("\033[96mYou\033[0m > ")
AGENT_PREFIX = "\033[92mAgent\033[0m > "


class CLIChannel:
    def __init__(self, dispatcher: GatewayDispatcher, agent_id: str = "default", db: Any = None) -> None:
        self._dispatcher = dispatcher
        self._agent_id = agent_id
        self._peer_id = "cli-user"
        self._msg_id = 0
        self._task: asyncio.Task | None = None
        _history_file = __import__("pathlib").Path.home() / ".agi_history"
        self._session = PromptSession(history=FileHistory(str(_history_file)))
        self._db = db

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info("CLI channel started (agent: %s)", self._agent_id)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def print_chunk(self, chunk: str) -> None:
        """Called by agent loop for streaming output.

        patch_stdout() ensures this is safe to call from any asyncio task —
        it temporarily clears the prompt, writes the chunk, then redraws.
        """
        sys.stdout.write(chunk)
        sys.stdout.flush()

    async def _loop(self) -> None:
        print(f"\n\033[1magi CLI\033[0m  (agent: {self._agent_id})")
        print("Type your message and press Enter. Ctrl+C or 'exit' to quit.\n")

        # patch_stdout() intercepts all sys.stdout writes from any coroutine or
        # thread while waiting for input, clears the prompt line, prints the
        # external output, then redraws the prompt.  This eliminates the need
        # for the user to press Enter after subagent announce output.
        with patch_stdout(raw=True):
            while True:
                try:
                    text = await self._session.prompt_async(PROMPT)
                except KeyboardInterrupt:
                    print("\nBye.")
                    break
                except EOFError:
                    print("\nBye.")
                    break

                text = text.strip()
                if not text:
                    continue
                if text.lower() in ("exit", "quit", "bye"):
                    print("Bye.")
                    break
                if text.startswith("/"):
                    await self._handle_command(text)
                    continue

                self._msg_id += 1
                inbound = InboundMessage(
                    id=self._msg_id,
                    channel="cli",
                    peer_kind="direct",
                    peer_id=self._peer_id,
                    sender=self._peer_id,
                    content=text,
                    created_at_ms=int(time.time() * 1000),
                    metadata={
                        "force_agent_id": self._agent_id,
                        "on_text": self.print_chunk,
                    },
                )

                sys.stdout.write(AGENT_PREFIX)
                sys.stdout.flush()

                streamed: list[str] = []
                original_on_text = inbound.metadata["on_text"]

                def tracking_on_text(chunk: str) -> None:
                    streamed.append(chunk)
                    original_on_text(chunk)

                inbound.metadata["on_text"] = tracking_on_text

                _t0 = time.time()
                try:
                    reply = await self._dispatcher.submit_inbound(inbound)
                    if streamed:
                        print()  # newline after streamed output
                    elif reply:
                        print(reply)
                    # Show token usage if DB available
                    if self._db is not None:
                        try:
                            from agi.storage.db import usage_query
                            _elapsed = time.time() - _t0
                            stats = await usage_query(self._db, session_key=f"agent:{self._agent_id}:cli:direct:{self._peer_id}")
                            if stats["calls"] > 0:
                                print(f"\033[90m[tokens: {stats['prompt_tokens']}↑ {stats['completion_tokens']}↓  {_elapsed:.1f}s]\033[0m")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"\n\033[91mError: {e}\033[0m")

    async def _handle_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        name = parts[0].lower()

        if name == "/clear":
            self._msg_id += 1
            await self._dispatcher.submit_inbound(InboundMessage(
                id=self._msg_id, channel="cli",
                peer_kind="direct", peer_id=self._peer_id,
                sender="system", content="__clear_history__",
                created_at_ms=int(time.time() * 1000),
                metadata={
                    "force_agent_id": self._agent_id,
                    "clear_history": True,
                },
            ))
            print("History cleared.")

        elif name == "/agent" and len(parts) > 1:
            self._agent_id = parts[1].strip()
            print(f"Switched to agent: {self._agent_id}")

        elif name == "/clearcron":
            if self._db is None:
                print("Database not available.")
            else:
                async with self._db.execute(
                    "DELETE FROM sessions WHERE channel='cron'"
                ) as cur:
                    deleted = cur.rowcount
                await self._db.commit()
                print(f"Cleared {deleted} cron session(s).")

        elif name == "/usage":
            if self._db is None:
                print("Database not available.")
            else:
                try:
                    from agi.storage.db import usage_query
                    total = await usage_query(self._db)
                    agent = await usage_query(self._db, agent_id=self._agent_id)
                    print(
                        f"Agent [{self._agent_id}]: "
                        f"{agent['prompt_tokens']}↑ {agent['completion_tokens']}↓ "
                        f"= {agent['total_tokens']} tokens ({agent['calls']} calls)\n"
                        f"Total all agents: {total['total_tokens']} tokens"
                    )
                except Exception as e:
                    print(f"Usage query failed: {e}")

        elif name == "/help":
            print(
                "/clear        — clear conversation history\n"
                "/clearcron    — clear cron session history\n"
                "/agent <id>   — switch agent\n"
                "/usage        — show token usage stats\n"
                "/help         — show this help\n"
                "exit / quit   — quit"
            )
        else:
            print(f"Unknown command: {name}. Try /help")
