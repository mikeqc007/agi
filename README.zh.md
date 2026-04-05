# AGI — 自主多智能体运行时

[English](README.md) | 中文

Python · MIT License

AGI 是一个开源多智能体运行时，通过协调子智能体、持久化记忆和持续执行，完成几乎任何任务——由可扩展的技能驱动。

## 一行代理设置

如果你使用 Claude Code、Cursor、Windsurf 或其他编程代理，把这段提示直接交给它：

```
Help me clone AGI if needed, then set it up for local development: git clone https://github.com/mikeqc007/agi && cd agi && pip install -e . && cp agi.yaml.example agi.yaml
```

## 快速开始

**1. 安装**

```bash
git clone https://github.com/mikeqc007/agi.git
cd agi
pip install -e .
```

**2. 配置**

```bash
cp agi.yaml.example agi.yaml
```

编辑 `agi.yaml`，设置模型：

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

填入 API 密钥：

```yaml
keys:
  ANTHROPIC_API_KEY: "sk-ant-..."
  OPENAI_API_KEY: "sk-..."
  GEMINI_API_KEY: "..."
  OPENROUTER_API_KEY: "sk-or-..."
```

**3. 运行**

```bash
# 交互式命令行
agi run --cli

# 完整运行时（所有渠道）
agi run
```

## 渠道

| 渠道 | 说明 |
|---|---|
| Telegram | Bot API，长轮询 |
| Discord | discord.py，@mention 触发 |
| CLI | 交互式终端会话 |
| HTTP Gateway | `POST /v1/messages` |
| OpenAI API | `POST /v1/chat/completions` |

在 `agi.yaml` 中配置 Telegram：

```yaml
telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
  allowed_users: []          # 例如 [123456789]，空 = 允许所有人
  default_agent_id: telegram
```

## MCP 服务器

AGI 支持外部 MCP 服务器扩展能力。每个服务器以子进程方式通过 stdio 启动，工具自动发现，并通过与原生工具相同的 OpenAI 兼容 schema 暴露出来。

```yaml
mcp:
  servers:
    - name: weather
      command: /path/to/mcp_server
```

## 从聊天机器人到智能体

AGI 最初是一个个人自动化运行时——很快变成了更多。开发者用它来管理代码库、自动化工作流、运行定时任务，以及通过 Telegram、Discord 和网络协调多步骤工作。

让智能体真正有用的不是模型本身——而是围绕模型的基础设施。能够搜索网络并读取页面。打开文件、精确编辑、运行测试。记住上周发生的事情。调度一个任务，带着结果回来。

AGI 就是这套基础设施，用 Python 构建。

## 核心能力

### 技能与工具

技能是 AGI 能做几乎任何事情的关键。

技能是一个结构化的能力模块——一个 Markdown 文件，定义工作流程并引用辅助脚本。技能按需加载，而非一次全部加载，让上下文窗口始终聚焦于当前任务。

```text
skills/
└── summarize/
    ├── SKILL.md
    └── scripts/
        └── summarize.py
```

工具的设计理念相同。AGI 包含一套核心工具集——网页搜索、网页抓取、文件读写编辑、shell 执行、grep 和 glob 代码搜索，以及 Playwright 浏览器自动化——并通过 MCP 服务器和技能脚本扩展。你可以随意替换或添加任何工具。

### 浏览器自动化

AGI 不只是抓取页面——它操作页面。

基于 Playwright 的浏览器让智能体真正控制 Chromium：导航到任意 URL、与元素交互、运行任意 JavaScript、截取屏幕截图。支持视觉的模型可以将截图直接输入到智能体循环中，在决定下一步之前先对页面进行推理。

对于桌面级自动化，AGI 还支持截图、鼠标控制、键盘输入，以及 `computer_use`——一种视觉引导的指令模式，截图后推理并执行相应操作。

### 子智能体

复杂任务很少能在单次对话中完成。AGI 会拆分它们。

主智能体并发派发子智能体——每个都在独立的会话中运行，有自己的上下文、工具访问权限和终止条件。所有子智能体完成后，结果统一收集并作为一条消息注入主智能体的上下文，包含每个子智能体的原始工具调用数据。主智能体随后一次性输出最终整合结果。

```
主智能体
├── 派发子智能体 A  ──────────────────────┐ 并发
├── 派发子智能体 B  ──────────────────────┤ 并发
│                                        ↓
└── 等待全部完成 → 注入结果 + 工具数据 → 单次最终输出
```

### 长期记忆

大多数智能体在对话结束后会忘记一切。AGI 会记住。

记忆以 Markdown 文件形式写入并索引到 SQLite。检索结合了密集向量搜索（sqlite-vec）、稀疏 BM25（FTS5）、倒数排名融合、时间衰减和 MMR 重排——全部在本地单一流水线中完成，无需外部向量数据库。

### 调度

`CronService` 将 APScheduler 与智能体感知的调度结合：

- 标准 cron 语法：`0 9 * * 1-5`
- 间隔简写：`interval:30m`、`interval:2h`
- 一次性任务：`once`

任务将 `InboundMessage` 直接注入目标会话，并在 SQLite 中持久化，重启后仍有效。智能体可在运行时通过 `cron_add`、`cron_list`、`cron_delete` 工具管理任务。

### 语音

配置 Whisper 后，AGI 会在传入智能体循环之前将 Telegram 语音消息转录为文字。回复可通过 TTS 转换为语音输出（默认 edge-tts）。

```yaml
tts:
  provider: edge
  voice: zh-CN-XiaoxiaoNeural
```

### 推荐模型

AGI 与模型无关——兼容任何实现了 OpenAI 兼容 API 的 LLM。以下能力会让它表现更好：

- 长上下文窗口，用于多步骤任务
- 强大的工具调用能力，确保可靠的函数调用
- 多模态输入，用于截图和图像推理

## 架构

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

所有入站消息无论来源，均被规范化为统一的 `InboundMessage` 格式。每个会话由独立的 `asyncio.Lock` 保护，确保每次只有一个 turn 在执行。

## 配置参考

```yaml
agents:
  - id: default
    system_prompt: "You are a helpful AI assistant."
    tools_deny: []     # 可选 — 例如 [shell, browser] 限制工具访问
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

### OpenAI 兼容接口

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "你好"}]}'
```

### 统一消息接口

```bash
curl http://localhost:8090/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "api",
    "agent_id": "default",
    "peer_id": "user-1",
    "content": "你好"
  }'
```

## 环境要求

- Python 3.11+
- SQLite with sqlite-vec extension
- Playwright（浏览器工具）：`playwright install chromium`

## 许可证

MIT
