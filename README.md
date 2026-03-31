# agi

Minimal agent runtime with tools, browser automation, hooks, TTS, and HTTP APIs.

## Ingress Architecture

All channels now go through one ingress dispatcher before reaching the agent runtime:

- `Telegram/Discord/CLI/OpenAI API/Gateway HTTP -> GatewayDispatcher -> submit_internal`

This is the minimal full-ingress unification layer.

## Gateway Quick Start

Configure `~/.agi/config.yaml`:

```yaml
gateway:
  enabled: true
  host: "0.0.0.0"
  port: 8090
  api_key: ""   # optional
```

Run:

```bash
agi run
```

Health check:

```bash
curl http://localhost:8090/health
```

## Frontend Connections

### Open WebUI

1. `Settings -> Connections -> OpenAI API`
2. Base URL: `http://<your-host>:8090/v1`
3. API Key: leave empty if `gateway.api_key` is empty; otherwise fill the same key.
4. Model: choose your agent id (for example `default`).

### Chatbox / LobeChat / Other OpenAI-compatible clients

1. Provider: `OpenAI API Compatible`
2. Base URL: `http://<your-host>:8090/v1`
3. API Key: same rule as above
4. Model: agent id

## API Examples

### OpenAI-compatible chat

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"你好"}],
    "stream": false
  }'
```

### Unified message endpoint

```bash
curl http://localhost:8090/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "webui",
    "peer_kind": "direct",
    "peer_id": "user-1",
    "sender": "user-1",
    "agent_id": "default",
    "content": "帮我总结这个页面",
    "stream": false
  }'
```

## Notes

- `gateway` is a new unified HTTP entry.
- Legacy `openai_api` channel can still be enabled independently.
- Browser/Hooks/TTS are runtime-level capabilities and can be used through any channel.
