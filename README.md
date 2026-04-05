# AGI — Autonomous Multi-Agent Runtime

English | [中文](README.zh.md)

Python

AGI is an open-source multi-agent runtime that orchestrates subagents, memory, and persistent execution to handle almost anything — driven by extensible skills.

## One-Line Agent Setup

If you use Claude Code, Cursor, Windsurf, or another coding agent, hand it this prompt:

```
Help me clone AGI if needed, then set it up for local development: git clone https://github.com/mikeqc007/agi && cd agi && pip install -e . && cp agi.yaml.example agi.yaml
```

## Quickstart

**1. Install**

```bash
git clone https://github.com/mikeqc007/agi.git
cd agi
pip install -e .
```

**2. Configure**

```bash
cp agi.yaml.example agi.yaml
```

Edit `agi.yaml` and set a model:

```yaml
agents:
  - id: default
    model:
      primary: "anthropic/claude-sonnet-4-5"
      fallbacks:
        - "openai/gpt-4o-mini"
        - "gemini/gemini-2.5-flash"
        - "openrouter/arcee-ai/trinity-large-preview:free"
```

Add your API keys:

```yaml
keys:
  ANTHROPIC_API_KEY: "sk-ant-..."
  OPENAI_API_KEY: "sk-..."
  GEMINI_API_KEY: "..."
  OPENROUTER_API_KEY: "sk-or-..."
```

**3. Run**

```bash
# Interactive CLI
agi run --cli

# Full runtime
agi run
```

## Channels

| Channel | Description |
|---|---|
| Telegram | Bot API, long polling |
| Discord | discord.py, mention-gated |
| CLI | Interactive terminal session |
| HTTP Gateway | `POST /v1/messages` |
| OpenAI API | `POST /v1/chat/completions` |

Configure Telegram in `agi.yaml`:

```yaml
telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
  allowed_users: []          # e.g. [123456789] — empty = allow all
  default_agent_id: telegram
```

## MCP Servers

AGI supports external MCP servers to extend its capabilities. Each server launches as a subprocess over stdio, its tools are auto-discovered, and the resulting capabilities are exposed through the same OpenAI-compatible schema used by native tools.

```yaml
mcp:
  servers:
    - name: weather
      command: /path/to/mcp_server
```

## From Chatbot to Agent

AGI started as a personal automation runtime — and quickly became something more. Developers use it to manage codebases, automate workflows, run scheduled tasks, and coordinate multi-step work across Telegram, Discord, and the web.

What makes an agent useful isn't the model — it's the infrastructure around the model. The ability to search the web and read a page. To open a file, edit it precisely, and run the tests. To remember what happened last week. To schedule a job and come back with the result.

AGI is that infrastructure, built in Python.

## Core Capabilities

### Skills and Tools

Skills are the reason AGI can do almost anything.

A skill is a structured capability module — a Markdown file that defines a workflow and references to supporting scripts. Skills are loaded on demand, not all at once, keeping the context window focused on the current task.

```text
skills/
└── summarize/
    ├── SKILL.md
    └── scripts/
        └── summarize.py
```

The tool design philosophy is the same. AGI includes a core toolset — web search, web scraping, file read/write/edit, shell execution, grep and glob for code search, and Playwright browser automation — and extends through MCP servers and skill scripts. You can replace or add any of them.

### Browser Automation

AGI doesn't just fetch pages — it operates them.

A Playwright-backed browser gives the agent real control over Chromium: navigate to any URL, interact with elements, run arbitrary JavaScript, and capture screenshots. Vision-capable models receive screenshots directly into the agent loop and can reason about what they see before deciding what to do next.

For desktop-level automation, AGI also supports screenshot capture, mouse control, keyboard input, and `computer_use` — a vision-guided instruction mode that takes a screenshot, reasons about it, and executes the appropriate action.

### Subagents

Complex tasks rarely fit in a single turn. AGI breaks them down.

The lead agent spawns subagents concurrently — each runs in its own isolated session with its own context, tool access, and termination condition. Once all subagents finish, their results are collected together and injected into the lead agent's context as a single message, including the raw tool call outputs from each subagent's execution. The lead agent then produces one final synthesized response.

```
Lead agent
├── spawn subagent A  ──────────────────────┐ concurrent
├── spawn subagent B  ──────────────────────┤ concurrent
│                                           ↓
└── wait for all → inject results + tool data → single final response
```

### Long-Term Memory

Most agents forget everything when the conversation ends. AGI remembers.

Memory is written as Markdown files and indexed into SQLite. Retrieval combines dense vector search (sqlite-vec), sparse BM25 (FTS5), reciprocal rank fusion, temporal decay, and MMR reranking — all in a single local pipeline, with no external vector database required.

### Scheduling

`CronService` wraps APScheduler with agent-aware scheduling:

- Standard cron syntax: `0 9 * * 1-5`
- Interval shorthand: `interval:30m`, `interval:2h`
- One-shot jobs: `once`

Jobs inject an `InboundMessage` directly into the target session and persist in SQLite across restarts. The agent can add, list, and remove jobs at runtime through the `cron_add`, `cron_list`, and `cron_delete` tools.

### Voice

When Whisper is configured, AGI transcribes Telegram voice messages into text before passing them to the agent loop. Responses can be converted back to speech via TTS (edge-tts by default).

```yaml
tts:
  provider: edge
  voice: zh-CN-XiaoxiaoNeural
```

### Execution Traces

Every turn is recorded as a structured JSON file under `logs/traces/{date}/{session_key}/`. Each file captures the full execution of one turn: the user query, the main agent's tool calls and reasoning, each subagent's tool trajectory and result, the final answer, and diagnostics.

```
logs/traces/2026-04-04/default_cli_direct_user1/
└── turn-a1b2c3d4.json
```

```json
{
  "meta": { "timestamp": "...", "session_key": "...", "turn_id": "turn-a1b2c3d4" },
  "turn": {
    "query": "...",
    "main": {
      "agent": "default",
      "model": "anthropic/claude-sonnet-4-5",
      "trajectory": [
        {
          "iter": 1,
          "think": "...",
          "action": { "type": "tool_call", "tool": "spawn_agent", "args": { "task": "..." } },
          "tool_result": { "status": "pending", "run_id": "sub-abc" }
        }
      ]
    },
    "subagents": [
      {
        "run_id": "sub-abc",
        "label": "...",
        "assigned_task": "...",
        "trajectory": [
          {
            "iter": 1,
            "think": "...",
            "action": { "type": "tool_call", "tool": "web_search", "args": { "query": "..." } },
            "tool_result": { "title": "...", "url": "..." }
          },
          {
            "iter": 2,
            "think": "...",
            "action": { "type": "finish", "result": "..." },
            "tool_result": null
          }
        ],
        "final_result": "...",
        "status": "ok",
        "elapsed_ms": 3200
      }
    ],
    "answer": "..."
  },
  "diagnostics": {
    "duration_ms": 6500,
    "total_tool_calls": 3,
    "subagent_count": 2,
    "no_tool_subagents": [],
    "error_subagents": []
  }
}
```

### Recommended Models

AGI is model-agnostic — it works with any LLM that implements an OpenAI-compatible API. It performs best on models that support:

- Long context windows for multi-step tasks
- Strong tool use for reliable function calling
- Multimodal input for screenshot and image reasoning

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
       ┌────────────┼────────────┐
     Tools        Memory      Skills / MCP
```

All inbound messages are normalized into a single `InboundMessage` format regardless of origin. Each session is guarded by a per-session `asyncio.Lock`, keeping execution scoped to one turn at a time per session.

## Configuration Reference

```yaml
agents:
  - id: default
    system_prompt: "You are a helpful AI assistant."
    tools_deny: []     # optional — e.g. [shell, browser] to restrict tool access
    model:
      primary: "anthropic/claude-sonnet-4-5"
      fallbacks:
        - "openai/gpt-4o-mini"
        - "gemini/gemini-2.5-flash"
      temperature: 0.7
      max_tokens: 8192

telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
  allowed_users: []
  default_agent_id: default

skills_dir: "./skills"

tts:
  provider: edge
  voice: zh-CN-XiaoxiaoNeural

mcp:
  servers:
    - name: example
      command: /path/to/mcp_server

keys:
  ANTHROPIC_API_KEY: "sk-ant-..."
  OPENAI_API_KEY: "sk-..."
  GEMINI_API_KEY: "..."
  OPENROUTER_API_KEY: "sk-or-..."
```

## HTTP API

### OpenAI-compatible endpoint

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

