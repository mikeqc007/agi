# AGI — Autonomous Multi-Agent Runtime

A Python-based autonomous agent runtime supporting multi-channel interaction, dynamic subagent spawning, scheduled task automation, persistent memory, and extensible tool/skill systems.

## Architecture

```
Telegram / Discord / CLI / HTTP / OpenAI-compatible API
                        │
                 GatewayDispatcher
                        │
                   MessageQueue          ← per-session, drop/collect modes
                        │
                    AgentLoop            ← ReAct-style reasoning & tool execution
                   /    |    \
          SubagentManager  CronService  HookManager
                   │
         ┌─────────┼─────────┐
       Tools     Memory     Skills/MCP
  (shell/fs/web  (hybrid     (YAML/MD
   /browser/tts)  search)     + stdio)
```

### Channels

All inbound messages are normalized into `InboundMessage` and routed through a single `GatewayDispatcher`, regardless of origin.

| Channel | Description |
|---|---|
| Telegram | Bot API, polling or webhook |
| Discord | discord.py, mention-gated |
| CLI | Interactive terminal session |
| Gateway HTTP | Unified REST endpoint (`POST /v1/messages`) |
| OpenAI API | Compatible endpoint (`POST /v1/chat/completions`) |

### Agent Loop

Each session runs an isolated ReAct loop:

1. Build context — system prompt + memory injection + history compaction
2. Stream LLM response
3. Execute tool calls concurrently, collect results
4. Detect dead-loops (repeated identical calls)
5. Iterate until `end_turn` or max iterations

Sessions are guarded by per-session `asyncio.Lock` for concurrency safety.

### Subagent Spawning

The `SubagentManager` supports dynamic child agent delegation:

- Agents can spawn subagents via the `spawn_subagent` tool
- Bounded recursive depth (configurable `max_depth`)
- Each subagent runs in an isolated session context
- Results are propagated asynchronously back to the parent session
- Max concurrent subagents configurable per runtime

### Cron Scheduler

`CronService` wraps APScheduler with agent-aware scheduling:

- Standard cron syntax: `0 9 * * 1-5`
- Interval shorthand: `interval:30m`, `interval:2h`
- One-shot jobs: `once`
- Jobs inject `InboundMessage` directly into the agent session
- Persisted to SQLite — survives restarts
- Manageable via the `cron` tool at runtime

### Memory

Hybrid retrieval pipeline per agent:

- **Dense**: sqlite-vec cosine similarity
- **Sparse**: FTS5 BM25
- **Fusion**: Reciprocal Rank Fusion
- **Temporal decay**: exponential scoring by recency
- **Reranking**: MMR or cross-encoder
- Memory is written as Markdown files and indexed to SQLite

### Tools

Built-in tools registered at startup:

| Tool | Description |
|---|---|
| `shell` | Execute shell commands (with approval gating) |
| `fs` | Read/write/list files |
| `web` | HTTP fetch, search |
| `memory` | Read/write persistent memory |
| `browser` | Playwright-based browser automation |
| `computer` | Screenshot, mouse, keyboard (pyautogui) |
| `tts` | Text-to-speech via edge-tts |
| `say` | Announce action before execution |
| `cron` | Manage scheduled tasks |
| `spawn_subagent` | Delegate tasks to child agents |
| `skills` | Invoke skills by name |
| `mcp` | Call MCP server tools |

### Skills

Skills are Markdown or YAML files with a prompt template. Loaded from:

1. `./skills/` (workspace)
2. Managed skills dir (configurable)

Skills are invokable by name through the `skills` tool.

### MCP

External MCP servers are configured in `agi.yaml` under `mcp.servers`. Each server is launched as a subprocess and communicates over stdio. Tools are auto-discovered and unified under the same OpenAI-compatible schema.

### Hooks

`HookManager` supports lifecycle hooks:

- `on_startup` / `on_shutdown`
- `on_message` — fires on every inbound message
- `on_reply` — fires after agent response
- `on_tool_call` — fires on every tool invocation

## Installation

```bash
pip install -e .
```

## Configuration

Copy the example config and fill in your values:

```bash
cp agi.yaml.example agi.yaml
```

Key sections in `agi.yaml`:

```yaml
agents:
  - id: default
    system_prompt: "..."
    model:
      primary: "openrouter/..."
      fallbacks: [...]

telegram:
  token: "YOUR_BOT_TOKEN"
  allowed_users: [123456789]

mcp:
  servers:
    - name: weather
      command: /path/to/mcp_server

keys:
  OPENROUTER_API_KEY: "..."
```

## Usage

```bash
# Start runtime (Telegram + HTTP gateway)
agi run

# Interactive CLI session
agi chat

# One-shot message
agi message "summarize today's news"
```

## HTTP API

### OpenAI-compatible

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "hello"}]}'
```

### Unified message endpoint

```bash
curl http://localhost:8090/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "api",
    "agent_id": "default",
    "peer_id": "user-1",
    "content": "hello"
  }'
```

## Requirements

- Python 3.11+
- SQLite with sqlite-vec extension
- Playwright (for browser tools): `playwright install chromium`
