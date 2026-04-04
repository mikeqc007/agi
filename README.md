# AGI — Autonomous Multi-Agent Runtime

**Project Status:** Under active development.

AGI is a Python multi-agent runtime for long-horizon execution across every channel you already use — Telegram, Discord, Slack, WhatsApp, CLI, HTTP, and OpenAI-compatible APIs.

It's not a research prototype. It's a runtime that coordinates agent loops, persistent memory, scheduled execution, and a unified capability layer spanning tools, skills, and MCP servers — designed to do real work, continuously.

## What It Can Do

- Run agents locally through an interactive CLI or remotely through Telegram, Discord, Slack, WhatsApp, HTTP, and OpenAI-compatible APIs
- Perform coding tasks like a coding agent, similar to Claude Code: read files, search codebases, make precise edits, run and verify code through shell — all driven by the agent loop
- Execute broader automation tasks through full Playwright browser automation — navigate, click, fill forms, intercept network requests, evaluate JavaScript, and manage tabs, cookies, and storage
- Accept text and image inputs, including channel attachments and screenshots captured during tool execution
- Route screenshot outputs back through the agent loop for vision-capable models or a dedicated `vision_model`
- Transcribe Telegram voice messages into text when Whisper is configured
- Return streamed text responses, with optional voice output via TTS
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
  OPENROUTER_API_KEY: "sk-or-..."
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

## From Chatbot to Agent

Most LLM wrappers give you a chat interface. AGI gives you a runtime.

The difference is execution. When you ask AGI to research a topic, it searches the web and scrapes pages. When you ask it to fix a bug, it reads the codebase, makes precise edits, runs the tests, and iterates until they pass. When you ask it to monitor something overnight, it schedules a cron job, runs it while you sleep, and sends you the result in the morning.

This is the difference between a model that describes actions and a runtime that takes them.

## Core Capabilities

### Tools

AGI includes a comprehensive core toolset — web search, web scraping, file read/write/edit, shell execution, and full browser automation — and extends through MCP servers and custom skill scripts. Every tool follows the same OpenAI-compatible function schema, so they compose naturally and any tool can be replaced or extended.

The tool design philosophy is simple: tools should do one thing well. `web_search` finds; `web_fetch` retrieves and extracts clean text from any URL. Together they power deep research workflows without writing a single line of glue code. `shell` executes commands with approval gating; the `pty_*` family keeps persistent terminal sessions alive across turns. You can replace or add any of them.

```text
Core toolset
├── web_search           DuckDuckGo search with ranked results
├── web_fetch            HTTP fetch + HTML-to-text extraction (web scraping)
├── fs_read              Read files within the agent workspace
├── fs_write             Write or overwrite files
├── fs_list              List directory contents (recursive optional)
├── fs_delete            Delete files
├── shell                Execute shell commands with approval gating
├── pty_spawn/read/write Persistent pseudo-terminal sessions across turns
└── ports_list/probe     Inspect and probe running services

Browser automation (14 tools)
├── browser_navigate     Navigate to any URL
├── browser_snapshot     Full DOM snapshot with accessibility tree
├── browser_click        Click elements by selector or coordinates
├── browser_fill         Fill form inputs
├── browser_get_text     Extract visible text from elements
├── browser_screenshot   Capture viewport or element screenshots
├── browser_scroll       Scroll page or element
├── browser_eval         Execute arbitrary JavaScript
├── browser_tabs         Manage multiple browser tabs
├── browser_cookies      Get, set, and clear cookies
├── browser_storage      Read and write localStorage / sessionStorage
├── browser_network_rule Intercept and mock network requests
├── browser_wait         Wait for selectors, navigation, or network idle
└── browser_download_wait Wait for file downloads to complete

Agent orchestration
├── spawn_agent          Spawn a subagent in an isolated session
├── sessions_spawn       Create concurrent named agent sessions
├── sessions_list/kill   Inspect and terminate sessions
├── sessions_steer       Inject messages into running sessions
└── watchdog_spawn/stop  Monitor and guard long-running processes

Memory
├── memory_search        Hybrid dense + sparse retrieval with temporal decay
├── memory_store         Write memory as indexed Markdown
└── memory_delete        Remove memory entries

Scheduling
├── cron_add             Add cron / interval / one-shot jobs
├── cron_list            List scheduled jobs
└── cron_remove          Remove jobs

Skills & MCP
├── skill_list           List available skills
├── skill_run            Load and execute a skill
├── mcp                  Call a tool on any configured MCP server
└── mcp_list             List available MCP server tools

Communication
├── message_send         Send a message to any channel or peer
└── speak                Convert text to speech (TTS output)
```

### Browser Automation

AGI doesn't just fetch pages — it operates them. Fourteen browser tools built on Playwright give the agent full control over a real Chromium instance: navigate, interact with elements, intercept network traffic, evaluate JavaScript, capture screenshots, and manage tabs and storage. Vision-capable models receive screenshot outputs back into the agent loop and can reason about what they see.

This is the difference between a web scraper and a browser agent.

### Multi-Agent Delegation

Complex tasks rarely fit in a single turn. AGI breaks them down.

The lead agent can dynamically spawn subagents — each running in its own isolated session with its own context, tool set, and termination condition. Subagents run concurrently and propagate their results back to the parent asynchronously. Runtime-level limits cap depth and concurrency to prevent runaway agent trees.

```
Lead agent
├── spawns subagent A  (isolated session, concurrent)
├── spawns subagent B  (isolated session, concurrent)
└── synthesizes results → final response
```

This is how AGI handles tasks that take minutes or hours: decompose into parallel, isolated workflows, then converge into a single coherent output.

### Persistent Memory

Most agents forget everything when the conversation ends. AGI remembers.

Memory is written as Markdown files and indexed into SQLite, giving the agent a structured, queryable knowledge base that grows over time. Retrieval combines dense vector search (sqlite-vec cosine similarity), sparse BM25 (FTS5), reciprocal rank fusion, temporal decay, and MMR reranking — all in a single pipeline, with no external vector database required.

### Scheduling

`CronService` wraps APScheduler with agent-aware scheduling:

- Standard cron syntax: `0 9 * * 1-5`
- Interval shorthand: `interval:30m`, `interval:2h`
- One-shot jobs: `once`
- Jobs inject `InboundMessage` directly into the target session
- Job state persists in SQLite across restarts
- Schedules are manageable at runtime via the `cron_add` / `cron_list` / `cron_remove` tools

### Skills

A skill is a structured capability module — a Markdown file that defines a workflow, best practices, and references to supporting scripts. Skills are loaded on demand, not all at once, so the context window stays focused on the current task.

```text
skills/
└── your-skill/
    ├── SKILL.md          workflow instructions + {baseDir} substitution
    └── scripts/          helper scripts invoked from SKILL.md
```

Skills are loaded from `./skills/` for shared project-level capabilities, or `state/agents/<id>/skills/` for per-agent overrides. When a task matches a skill, the agent calls `skill_run("name")`, receives the full instructions, and executes them using whatever tools are needed.

### MCP Servers

External MCP servers are configured in `agi.yaml` under `mcp.servers`. Each server launches as a subprocess over stdio, its tools are auto-discovered, and the resulting capabilities are exposed through the same OpenAI-compatible schema used by native tools. You can replace or extend any part of the toolset — native tools, skills, and MCP servers all compose through the same runtime model.

## Architecture

```text
Telegram / Discord / Slack / WhatsApp / CLI / HTTP / OpenAI API
                            │
                   GatewayDispatcher
                            │
                       MessageQueue
                            │
                        AgentLoop
                            │
               ┌────────────┼────────────┐
             Tools        Memory      Skills / MCP
```

The system separates ingress, orchestration, execution, and capability:

- **GatewayDispatcher** normalizes all inbound traffic into a single `InboundMessage` format regardless of origin
- **MessageQueue** decouples ingress from agent execution
- **AgentLoop** owns reasoning, tool execution, turn control, and loop termination
- **Tools / Memory / Skills / MCP** form the runtime capability substrate

Each session is guarded by a per-session `asyncio.Lock`, keeping execution scoped to one turn at a time per session.

| Channel | Description |
|---|---|
| Telegram | Bot API, long polling |
| Discord | discord.py, mention-gated |
| Slack | Events API |
| WhatsApp | WhatsApp Business API |
| Email | IMAP/SMTP |
| Feishu / DingTalk / LINE | Webhook |
| CLI | Interactive terminal session |
| HTTP Gateway | Unified REST endpoint (`POST /v1/messages`) |
| OpenAI API | Compatible endpoint (`POST /v1/chat/completions`) |

## Execution Model

Each session runs an isolated ReAct loop:

1. Build context — system prompt, memory injection, and history compaction
2. Stream LLM output
3. Execute tool calls concurrently and collect results
4. Detect dead-loops from repeated identical tool patterns
5. Iterate until `end_turn` or the configured iteration limit

## Hooks

`HookManager` supports lifecycle hooks that fire at key points in the agent loop:

- `on_startup` — runtime initialization
- `on_shutdown` — graceful teardown
- `on_message` — every inbound message
- `on_reply` — every outbound reply
- `on_tool_call` — every tool invocation

## Configuration Reference

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
