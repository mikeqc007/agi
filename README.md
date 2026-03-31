# AGI — Autonomous Multi-Agent Runtime

A Python-based autonomous agent runtime supporting multi-channel interaction, dynamic subagent spawning, scheduled task automation, persistent memory, and extensible tool/skill systems.

## Architecture

```
Telegram / Discord / CLI / HTTP / OpenAI-compatible API
                         │
                  GatewayDispatcher
                         │
                    MessageQueue        ← per-session, drop/collect modes
                         │
                     AgentLoop          ← ReAct-style reasoning & tool execution
                    /    │    \
        SubagentManager  │  HookManager
                         │
                    CronService
                         │
            ┌────────────┼────────────┐
          Tools        Memory      Skills/MCP
     (shell/fs/web   (hybrid       (SKILL.md
      browser/tts)    search)       + stdio)
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
- **Reranking**: MMR or LLM pointwise
- Memory is written as Markdown files and indexed to SQLite

### Tools

Built-in tools registered at startup:

| Tool | Description |
|---|---|
| `shell` | Execute shell commands (with approval gating) |
| `read_file` / `write_file` / `list_dir` | File system access |
| `web_fetch` / `web_search` | HTTP fetch and DuckDuckGo search |
| `remember` / `recall` / `forget` | Persistent memory read/write |
| `browser` | Playwright-based browser automation |
| `screenshot` / `computer_use` / `mouse_click` / `keyboard_type` | Desktop automation via pyautogui |
| `say` | Announce action before execution |
| `cron_add` / `cron_list` / `cron_delete` | Manage scheduled tasks |
| `spawn_agent` / `list_subagents` / `kill_subagent` | Delegate tasks to child agents |
| `read_skill` | Load a skill's instructions at runtime |

### Skills

Each skill is a subdirectory under `skills/` containing a `SKILL.md` and an optional `scripts/` directory:

```
skills/
  summarize/
    SKILL.md              ← instructions, when to use, how to invoke scripts
    scripts/
      summarize.py        ← executable helper, auto chmod +x on load
```

Flow:
1. All skills are listed in the agent system prompt with their description
2. When the user's request matches a skill, the agent calls `read_skill("name")`
3. `SKILL.md` is returned with `{baseDir}` replaced by the skill's absolute path
4. The agent follows the instructions — calling `shell`, `fs`, `web`, or running scripts via `shell("python3 {baseDir}/scripts/...")`

Skills are loaded from:
1. `./skills/` (project-level, shared across agents)
2. `memory/agents/<id>/skills/` (per-agent overrides)

### MCP

External MCP servers are configured in `agi.yaml` under `mcp.servers`. Each server is launched as a subprocess and communicates over stdio. Tools are auto-discovered and unified under the same OpenAI-compatible schema.

### Hooks

`HookManager` supports lifecycle hooks:

- `on_startup` / `on_shutdown`
- `on_message` — fires on every inbound message
- `on_reply` — fires after agent response
- `on_tool_call` — fires on every tool invocation

## Quickstart

**1. Install**

```bash
git clone https://github.com/mikeqc007/agi.git
cd agi
pip install -e .
```

**2. Create config**

```bash
agi init          # writes ~/.agi/config.yaml with defaults
```

Or copy the example into the project directory:

```bash
cp agi.yaml.example agi.yaml
```

**3. Set your LLM**

Edit `agi.yaml` (or `~/.agi/config.yaml`) and set a model. The default uses Ollama:

```yaml
agents:
  - id: default
    model:
      primary: "ollama/qwen3:8b"   # or "openai/gpt-4o-mini", "gemini/gemini-2.5-flash", etc.
```

For OpenRouter / OpenAI / Gemini, add the key under `keys:`:

```yaml
keys:
  OPENROUTER_API_KEY: "sk-or-..."
  OPENAI_API_KEY: "sk-..."
```

**4. Run**

```bash
# Interactive CLI session (no Telegram needed)
agi run --cli

# Or start full runtime (Telegram + HTTP gateway)
agi run
```

**5. Verify**

With `agi run --cli` you should see a prompt. Type a message and the agent will respond.

If you enabled the gateway (`gateway.enabled: true` in config), check:

```bash
curl http://localhost:8090/health
```

## Configuration Reference

Key sections in `agi.yaml`:

```yaml
agents:
  - id: default
    system_prompt: "You are a helpful assistant."
    model:
      primary: "ollama/qwen3:8b"
      fallbacks: ["openai/gpt-4o-mini"]
      temperature: 0.7
      max_tokens: 8192

telegram:
  token: "YOUR_BOT_TOKEN"
  allowed_users: [123456789]    # empty = allow all

gateway:
  enabled: false
  host: "0.0.0.0"
  port: 8090

mcp:
  servers:
    - name: weather
      command: /path/to/mcp_server

keys:
  OPENROUTER_API_KEY: "..."
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

## Contributing

Contributions welcome.
