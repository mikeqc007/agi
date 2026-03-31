from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="agi", help="agi — minimal AI agent")


def _config_dir(config: Optional[Path]) -> Path:
    """Return the directory that contains (or would contain) the config file."""
    import os
    if config:
        return Path(config).parent
    env_path = os.environ.get("MINICLAW_CONFIG")
    if env_path:
        return Path(env_path).parent
    for candidate in (Path("agi.yaml"), Path("config.yaml")):
        if candidate.exists():
            return candidate.parent
    return Path.home() / ".agi"


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    cli: bool = typer.Option(False, "--cli", help="Start interactive CLI instead of Telegram"),
    agent: str = typer.Option("default", "--agent", "-a", help="Agent ID to use with --cli"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Write logs to this file"),
) -> None:
    """Start agi."""
    file_level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers filter by level

    # Console: ERROR only — keeps the terminal clean (warnings go to log file).
    # In --debug mode, lower to DEBUG so developers see everything.
    console_level = logging.DEBUG if debug else logging.ERROR
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler: auto-defaults to logs/agi.log next to the config file.
    if log_file:
        resolved_log = Path(log_file)
    else:
        resolved_log = _config_dir(config) / "logs" / "agi.log"
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(resolved_log, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Telegram channel: separate log file
    tg_log = _config_dir(config) / "logs" / "telegram.log"
    tg_log.parent.mkdir(parents=True, exist_ok=True)
    tg_fh = logging.FileHandler(tg_log, encoding="utf-8")
    tg_fh.setLevel(logging.DEBUG)
    tg_fh.setFormatter(formatter)
    logging.getLogger("agi.channels.telegram").addHandler(tg_fh)

    for lib in ("httpx", "httpcore", "telegram", "apscheduler",
                "litellm", "LiteLLM", "LiteLLM Proxy"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    # Some MCP servers print non-JSON logs to stdout; this avoids noisy JSON-RPC parse tracebacks.
    logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)
    # Suppress litellm's direct print() spam ("Provider List:", "Give Feedback", etc.)
    import litellm as _litellm
    _litellm.suppress_debug_info = True

    asyncio.run(_run(config, cli_mode=cli, agent_id=agent))


async def _run(config_path: Path | None, cli_mode: bool = False, agent_id: str = "default") -> None:
    from agi.config import load_config
    from agi.app import AppRuntime

    cfg = load_config(config_path)
    runtime = AppRuntime(cfg)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Suppress anyio/mcp shutdown noise: when the MCP stdio_client's anyio TaskGroup
    # is torn down, background I/O tasks raise RuntimeError("cancel scope in a different
    # task").  This is a known mcp library bug — nothing we can fix from our side.
    def _exc_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_exc_handler)

    await runtime.start()

    if cli_mode:
        await runtime.start_cli(agent_id)
        # In CLI mode, wait for the CLI task to finish (user types 'exit')
        if runtime._cli and runtime._cli._task:
            try:
                await runtime._cli._task
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
    else:
        print("agi running. Press Ctrl+C to stop.")
        await stop_event.wait()

    print("\nShutting down...")
    await runtime.stop()


@app.command()
def init(
    path: Path = typer.Argument(
        Path.home() / ".agi" / "config.yaml",
        help="Where to write the config file",
    )
) -> None:
    """Create a default config file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        typer.echo(f"Config already exists: {path}")
        raise typer.Exit(1)

    default_config = """\
# agi configuration

telegram:
  token: ""          # set TELEGRAM_TOKEN env var or fill here
  allowed_users: []  # list of Telegram user IDs, empty = allow all
  default_agent_id: default

agents:
  - id: default
    name: "Assistant"
    system_prompt: |
      You are a helpful AI assistant. You can use tools to help the user.
      You have access to shell, file system, web search, memory, and computer automation.
    model:
      primary: "ollama/qwen3:8b"
      fallbacks: []
      temperature: 0.7
      max_tokens: 8192
    memory_enabled: true
    max_iterations: 30
    tool_profile: default

memory:
  embedding_model: "ollama/nomic-embed-text"
  top_k: 6
  half_life_days: 30
  mmr_lambda: 0.7
  reranker: mmr

gateway:
  enabled: false
  host: "0.0.0.0"
  port: 8090
  api_key: ""   # optional; if set, send Authorization: Bearer <api_key>

queue:
  mode: drop    # drop: only latest message; collect: batch messages
  cap: 5

max_subagent_concurrent: 4
max_subagent_depth: 3
"""
    path.write_text(default_config)
    typer.echo(f"Config created: {path}")
    typer.echo("Edit it and set your TELEGRAM_TOKEN, then run: agi run")


if __name__ == "__main__":
    app()
