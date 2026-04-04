# AGI — Autonomous Multi-Agent Runtime

Python · MIT License

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
      fallbacks: ["openai/gpt-4o-mini", "gemini/gemini-2.5-flash"]
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

# Full runtime (all channels)
agi run
```

## Channels

AGI accepts tasks from every messaging platform you already use. Channels start automatically once configured — none of them require a public IP.

| Channel | Transport |
|---|---|
| Telegram | Bot API, long polling |
| Discord | discord.py, mention-gated |
| Slack | Events API |
| WhatsApp | WhatsApp Business API |
| Email | IMAP / SMTP |
| Feishu / DingTalk / LINE | Webhook |
| CLI | Interactive terminal |
| HTTP Gateway | `POST /v1/messages` |
| OpenAI API | `POST /v1/chat/completions` |

Configure channels in `agi.yaml`:

```yaml
telegram:
  token: $TELEGRAM_BOT_TOKEN
  allowed_users: []          # empty = allow all

slack:
  bot_token: $SLACK_BOT_TOKEN
  app_token: $SLACK_APP_TOKEN

discord:
  token: $DISCORD_BOT_TOKEN
  allowed_users: []
```

## MCP Servers

AGI supports configurable MCP servers to extend its capabilities. Each server launches as a subprocess over stdio, its tools are auto-discovered, and the resulting capabilities are exposed through the same OpenAI-compatible schema used by native tools.

```yaml
mcp:
  servers:
    - name: weather
      command: /path/to/mcp_server
```

## From Assistant to Agent

AGI started as a personal automation runtime — but it quickly became something more. Developers have used it to run research pipelines, manage codebases across repositories, monitor services overnight, and coordinate multi-step workflows across Telegram, Discord, and the web. These weren't planned features. They emerged from giving an agent loop real execution power.

That told us something important: what makes an agent useful isn't the model — it's the infrastructure around the model. The ability to search the web and read a page. To open a file, edit it precisely, and run the tests. To remember what happened last week. To schedule a job and come back with the result.

So we built that infrastructure, carefully, in Python.

AGI is not a wrapper. It's a runtime — one that gives agents everything they need to do real work: file system access, web search and scraping, browser automation, persistent memory, cron scheduling, and the ability to spawn subagents for tasks that are too large for a single turn.

## Core Capabilities

### Skills and Tools

Skills are the reason AGI can do almost anything.

A standard agent skill is a structured capability module — a Markdown file that defines a workflow, best practices, and references to supporting scripts. You can add your own skills, replace built-in ones, or chain them into composite workflows.

Skills are loaded on demand, not all at once. Only the skill the current task needs is injected into context. This keeps the context window focused and makes AGI work well even with smaller models.

```text
skills/
├── research/SKILL.md
├── coding/SKILL.md
└── your-custom-skill/
    ├── SKILL.md
    └── scripts/
        └── helper.py          ← called from SKILL.md
```

The tool design philosophy is the same. AGI includes a core toolset — web search, web scraping, file read/write/edit, shell execution, and full browser automation — and supports custom tools through MCP servers and skill scripts. You can replace or add any of them.

### Browser Automation

AGI doesn't just fetch pages — it operates them.

A full Playwright-backed browser gives the agent real control over Chromium: navigate to any URL, click elements, fill forms, intercept network requests, run arbitrary JavaScript, capture screenshots, manage tabs and cookies. Vision-capable models receive screenshots directly into the agent loop and can reason about what they see before deciding what to do next.

This is the difference between a web scraper and a browser agent.

### Subagents

Complex tasks rarely fit in a single turn. AGI breaks them down.

The lead agent can dynamically spawn subagents — each running in its own isolated session with its own context, tool access, and termination condition. Subagents run concurrently and return structured results; the lead agent synthesizes them into a single coherent output.

This is how AGI handles tasks that take minutes or hours: decompose, parallelize, converge.

```
Lead agent
├── subagent A  — isolated session, concurrent
├── subagent B  — isolated session, concurrent
└── synthesizes results → final response
```

Recursive depth and concurrency are bounded by configurable limits, so delegation never becomes a runaway tree.

### Execution Environment

AGI doesn't just talk about actions — it takes them.

Every session has access to a real execution environment: a workspace directory for reading and writing files, shell execution with approval gating, and persistent pseudo-terminal sessions that stay alive across turns. The agent can run code, verify the output, and iterate — all within the same agent loop.

```text
workspace/
├── uploads/          ← files you provide
├── workspace/        ← agent working directory
└── outputs/          ← deliverables
```

This is the difference between a chatbot with tool access and an agent with an actual computer.

### Context Engineering

Each subagent runs in its own isolated context. It cannot see the lead agent's history or other subagents' work. This keeps subagents focused on their specific task without being distracted by irrelevant context from the broader session.

During long sessions, AGI actively manages the context window — summarizing completed subtasks, offloading intermediate results to the file system, and compressing information that is no longer relevant. This allows it to sustain multi-step tasks that would otherwise exceed the context limit.

### Long-Term Memory

Most agents forget everything when the conversation ends. AGI remembers.

Memory is written as Markdown files and indexed into SQLite. Retrieval combines dense vector search (sqlite-vec), sparse BM25 (FTS5), reciprocal rank fusion, temporal decay, and MMR reranking — all in a single local pipeline, with no external vector database required. The more you use it, the more it knows about your workflows, preferences, and context.

### Scheduling

`CronService` wraps APScheduler with agent-aware scheduling:

- Standard cron syntax: `0 9 * * 1-5`
- Interval shorthand: `interval:30m`, `interval:2h`
- One-shot jobs: `once`

Jobs inject an `InboundMessage` directly into the target session and persist in SQLite across restarts. The agent can add, list, and remove jobs at runtime through the cron tools.

### Recommended Models

AGI is model-agnostic — it works with any LLM that implements an OpenAI-compatible API. It performs best on models that support:

- Long context windows (100k+ tokens) for multi-step research and coding tasks
- Strong tool use for reliable function calling and structured output
- Multimodal input for image understanding and browser screenshot reasoning
- Reasoning capabilities for adaptive planning and task decomposition

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

All inbound messages are normalized into a single `InboundMessage` format regardless of origin. Each session is guarded by a per-session `asyncio.Lock`, keeping execution scoped to one turn at a time per session.

## Hooks

`HookManager` supports lifecycle hooks:

- `on_startup` / `on_shutdown`
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

## License

MIT
