# AGI — Autonomous Multi-Agent Runtime

**Project Status:** Under development. Contributions welcome.

This project is a multi-agent runtime for long-horizon execution across multiple channels, including Telegram, Discord, CLI, HTTP, and OpenAI-compatible APIs.

It combines session-isolated agent loops, recursive subagent delegation, persistent memory, scheduled execution, and a unified capability layer spanning tools, skills, and MCP servers.

## What It Can Do

- Run agents locally through an interactive CLI or remotely through Telegram, Discord, HTTP, and OpenAI-compatible APIs
- Perform coding tasks like a coding agent, similar to Claude Code: read files, search codebases with grep and glob, make precise edits, run and verify code through shell — all driven by the agent loop
- Execute broader automation tasks through browser automation and desktop-control tools
- Accept text and image inputs, including channel attachments and screenshots produced during tool execution
- Route screenshot outputs back through the agent loop for vision-capable models or a dedicated `vision_model`
- Transcribe Telegram voice messages into text when Whisper is configured
- Return streamed text responses, with optional Telegram voice output via TTS
- Delegate long-horizon tasks to subagents running in parallel, isolated sessions
- Schedule recurring or one-shot automation jobs that survive restarts

## Quickstart

**1. Install**

```bash
git clone https://github.com/mikeqc007/agi.git
cd agi
pip install -e .
```

**2. Create config**

```bash
agi init
```

Or copy the example into the project directory:

```bash
cp agi.yaml.example agi.yaml
```

**3. Set your LLM**

Edit `agi.yaml` (or `~/.agi/config.yaml`) and set a model:

```yaml
agents:
  - id: default
    model:
      primary: "anthropic/claude-sonnet-4-5"   # or "openai/gpt-4o", "gemini/gemini-2.5-flash", etc.
      fallbacks: ["openai/gpt-4o-mini", "gemini/gemini-2.5-flash"]
```

Add the relevant API keys under `keys:`:

```yaml
keys:
  ANTHROPIC_API_KEY: "sk-ant-..."
  OPENAI_API_KEY: "sk-..."
  GEMINI_API_KEY: "..."
  OPENROUTER_API_KEY: "sk-or-..."   # for free/open-source models via OpenRouter
```

**4. Run**

```bash
# Interactive CLI session
agi run --cli

# Full runtime
agi run
```

**5. Verify**

With `agi run --cli` you should see a prompt. If the HTTP gateway is enabled, check:

```bash
curl http://localhost:8090/health
```

## Key Capabilities

- **Coding agent capability**  
  Perform coding tasks like a coding agent, similar to Claude Code: read a codebase, locate relevant code, make precise edits, run tests, and iterate — all in one agent loop, powered by file read/write/edit, grep, glob, and shell tools.

- **Session-isolated agent execution**  
  Per-session isolation via `asyncio.Lock` keeps agent turns scoped to a single session at a time.

- **Recursive multi-agent delegation**  
  Agents can spawn subagents under bounded depth and concurrency limits, enabling structured task decomposition without runaway agent trees.

- **Hybrid memory retrieval**  
  Dense retrieval, sparse retrieval, reciprocal rank fusion, temporal decay, and reranking are combined into a single memory pipeline.

- **Persistent scheduling**  
  Cron and interval jobs are stored in SQLite and injected back into agent sessions, allowing automation to survive restarts.

- **Unified capability layer**  
  Built-in tools, skill modules, and external MCP servers are exposed through one runtime model and one OpenAI-compatible tool schema.

- **OpenAI-compatible serving layer**  
  The runtime exposes both `/v1/chat/completions` and a unified `/v1/messages` endpoint for external integration.

## Architecture

```text
Telegram / Discord / CLI / HTTP / OpenAI API
                    │
             GatewayDispatcher
                    │
               MessageQueue
                    │
                AgentLoop
                    │
          ┌─────────┼─────────┐
        Tools    Memory    Skills/MCP
```

The system separates ingress, orchestration, execution, and capability layers:

- **GatewayDispatcher** normalizes inbound traffic into a single `InboundMessage` format
- **MessageQueue** decouples ingress from agent execution
- **AgentLoop** owns reasoning, tool execution, turn control, and loop termination
- **Tools / Memory / Skills / MCP** form the runtime capability substrate

All inbound messages are normalized into `InboundMessage` and routed through a single `GatewayDispatcher`, regardless of origin.

| Channel | Description |
|---|---|
| Telegram | Bot API, polling or webhook |
| Discord | discord.py, mention-gated |
| CLI | Interactive terminal session |
| Gateway HTTP | Unified REST endpoint (`POST /v1/messages`) |
| OpenAI API | Compatible endpoint (`POST /v1/chat/completions`) |

## Execution Model

Each session runs an isolated ReAct loop:

1. Build context with system prompt, memory injection, and history compaction
2. Stream LLM output
3. Execute tool calls concurrently and collect results
4. Detect dead-loops from repeated identical tool patterns
5. Iterate until `end_turn` or the configured iteration limit

Sessions are guarded by per-session `asyncio.Lock` to keep execution scoped per session.

## Multi-Agent Delegation

The `SubagentManager` supports dynamic child-agent orchestration:

- Agents can spawn subagents via the `spawn_subagent` tool
- Recursive depth is bounded by configurable `max_depth`
- Each subagent runs in an isolated session context
- Results are propagated asynchronously back to the parent
- Runtime-level limits cap concurrent subagent execution

This allows long-horizon tasks to be decomposed into parallel, isolated workflows.

## Scheduling

`CronService` wraps APScheduler with agent-aware scheduling:

- Standard cron syntax: `0 9 * * 1-5`
- Interval shorthand: `interval:30m`, `interval:2h`
- One-shot jobs: `once`
- Jobs inject `InboundMessage` directly into the target session
- Job state is persisted to SQLite across restarts
- Schedules are manageable at runtime via the `cron` tool

## Memory System

Each agent uses a hybrid retrieval pipeline:

- **Dense**: sqlite-vec cosine similarity
- **Sparse**: FTS5 BM25
- **Fusion**: Reciprocal Rank Fusion
- **Temporal decay**: exponential recency scoring
- **Reranking**: MMR or LLM pointwise scoring

Memory is written as Markdown files and indexed into SQLite.

## Capability System

AGI unifies built-in tools, runtime-loaded skills, and external MCP servers under the same execution model.

### Tools

Built-in tools registered at startup:

| Tool | Description |
|---|---|
| `shell` | Execute shell commands (with approval gating) |
| `read_file` / `write_file` / `edit_file` / `list_dir` | File system access with precise editing |
| `grep` / `glob` | Search code by regex pattern or file glob |
| `web_fetch` / `web_search` | HTTP fetch and DuckDuckGo search |
| `remember` / `recall` / `forget` | Persistent memory read/write |
| `browser` | Playwright-based browser automation |
| `screenshot` / `computer_use` / `mouse_click` / `keyboard_type` | Desktop automation via pyautogui |
| `say` | Announce action before execution |
| `cron_add` / `cron_list` / `cron_delete` | Manage scheduled tasks |
| `spawn_agent` / `list_subagents` / `kill_subagent` | Delegate tasks to child agents |
| `read_skill` | Load a skill's instructions at runtime |
| `todo` | Track task progress across agent steps |

### Skills

Each skill is a subdirectory under `skills/` containing a `SKILL.md` and an optional `scripts/` directory:

```text
skills/
  summarize/
    SKILL.md
    scripts/
      summarize.py
```

Flow:

1. All skills are listed in the agent system prompt with their description
2. When the user's request matches a skill, the agent calls `read_skill("name")`
3. `SKILL.md` is returned with `{baseDir}` replaced by the skill's absolute path
4. The agent follows the instructions by calling tools or executing helper scripts

Skills are loaded from:

1. `./skills/` for project-level shared skills
2. `state/agents/<id>/skills/` for per-agent overrides

### MCP

External MCP servers are configured in `agi.yaml` under `mcp.servers`. Each server is launched as a subprocess over stdio, its tools are auto-discovered, and the resulting capabilities are exposed through the same OpenAI-compatible schema used by native tools.

## Hooks

`HookManager` supports lifecycle hooks:

- `on_startup`
- `on_shutdown`
- `on_message`
- `on_reply`
- `on_tool_call`

## Example Flow

User: "Summarize this repository"

1. `GatewayDispatcher` normalizes the inbound message
2. `AgentLoop` builds context and injects relevant memory
3. The model selects a capability such as `read_skill("summarize")`
4. The runtime executes the skill instructions via tools or scripts
5. Results can be written back to memory
6. The final response is returned through the originating channel

## Configuration Reference

Key sections in `agi.yaml`:

```yaml
agents:
  - id: default
    system_prompt: "You are a helpful assistant."
    model:
      primary: "anthropic/claude-sonnet-4-5"
      fallbacks: ["openai/gpt-4o-mini", "gemini/gemini-2.5-flash"]
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
  ANTHROPIC_API_KEY: "sk-ant-..."
  OPENAI_API_KEY: "sk-..."
  GEMINI_API_KEY: "..."
  OPENROUTER_API_KEY: "sk-or-..."
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
