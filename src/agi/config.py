from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    primary: str = "ollama/qwen3:8b"
    fallbacks: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    vision_model: str = ""   # separate vision-capable model for image description (optional)


class AgentConfig(BaseModel):
    id: str
    name: str = ""
    system_prompt: str = "You are a helpful assistant."
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory_enabled: bool = True
    require_mention: bool = False
    max_iterations: int = 30
    tool_profile: str = "default"   # default / safe / minimal
    tools_allow: list[str] = Field(default_factory=list)
    tools_deny: list[str] = Field(default_factory=list)
    compaction_threshold: float = 0.8   # fraction of max_tokens before compacting
    think_level: str = "off"            # off / minimal / low / medium / high
    permission_mode: str = "allow"      # allow / read_only / workspace_write / prompt


class TelegramConfig(BaseModel):
    token: str = ""
    allowed_users: list[int] = Field(default_factory=list)  # empty = allow all
    default_agent_id: str = "default"


class DiscordConfig(BaseModel):
    token: str = ""
    allowed_users: list[int] = Field(default_factory=list)  # empty = allow all
    default_agent_id: str = "default"
    require_mention: bool = True    # require @bot mention in servers


class OpenAIApiConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str = ""               # empty = no auth required


class GatewayConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8090
    api_key: str = ""               # empty = no auth required


class TtsConfig(BaseModel):
    provider: str = "none"          # none / edge / openai / elevenlabs
    voice: str = ""                 # provider-specific voice name / id
    api_key: str = ""               # for openai / elevenlabs
    model: str = ""                 # openai: tts-1 / tts-1-hd; elevenlabs: model id


class MemoryConfig(BaseModel):
    workspace: str = "~/.agi/state"
    embedding_model: str = "ollama/nomic-embed-text"
    embedding_dim: int = 768
    top_k: int = 6
    vector_weight: float = 0.7
    text_weight: float = 0.3
    half_life_days: float = 30.0
    mmr_lambda: float = 0.7
    reranker: str = "mmr"           # mmr / llm / none
    memory_dir: str = "state"       # relative to config_dir; stores persistent runtime state files
    flush_enabled: bool = True      # pre-compaction memory flush (openclaw style)
    flush_prompt: str = ""          # custom flush prompt (empty = use default)


class QueueConfig(BaseModel):
    mode: str = "drop"              # drop / collect
    cap: int = 5


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


class AppConfig(BaseModel):
    agents: list[AgentConfig] = Field(default_factory=list)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    openai_api: OpenAIApiConfig = Field(default_factory=OpenAIApiConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    keys: dict[str, str] = Field(default_factory=dict)  # injected into os.environ at startup
    skills_dir: str = "skills/shared"
    db_path: str = "~/.agi/agi.db"
    data_dir: str = "~/.agi"
    log_level: str = "INFO"
    max_subagent_concurrent: int = 4
    max_subagent_depth: int = 3
    max_subagent_children: int = 5
    subagent_timeout_seconds: int = 300
    session_reap_interval_hours: int = 24
    session_max_age_days: int = 90
    # Set by load_config; used to resolve relative paths relative to yaml location
    config_dir: str = ""

    def agent(self, agent_id: str) -> AgentConfig:
        for a in self.agents:
            if a.id == agent_id:
                return a
        if self.agents:
            return self.agents[0]
        return AgentConfig(id="default")

    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        if not path.is_absolute() and self.config_dir:
            path = Path(self.config_dir) / path
        return path

    def resolved_db_path(self) -> Path:
        return self._resolve(self.db_path)

    def resolved_data_dir(self) -> Path:
        return self._resolve(self.data_dir)

    def resolved_logs_dir(self) -> Path:
        return self._resolve(self.data_dir) / "logs"

    def resolved_skills_dir(self) -> Path:
        return self._resolve(self.skills_dir)

    def resolved_agent_skills_dir(self, agent_id: str) -> Path:
        return self._resolve(self.memory.memory_dir) / "agents" / agent_id / "skills"

    def resolved_memory_workspace(self) -> Path:
        return self._resolve(self.memory.workspace)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        import os
        env_path = os.environ.get("AGI_CONFIG")
        if env_path:
            path = Path(env_path)
    if path is None:
        # Check visible locations first, then fall back to hidden home dir
        for candidate in (
            Path("agi.yaml"),
            Path("config.yaml"),
            Path.home() / ".agi" / "config.yaml",
        ):
            if candidate.exists():
                path = candidate
                break
        else:
            path = Path.home() / ".agi" / "config.yaml"
    path = Path(path)

    data: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    # Env var overrides
    if token := os.getenv("TELEGRAM_TOKEN"):
        data.setdefault("telegram", {})["token"] = token
    if token := os.getenv("DISCORD_TOKEN"):
        data.setdefault("discord", {})["token"] = token
    if key := os.getenv("OPENAI_API_KEY_LOCAL"):
        data.setdefault("openai_api", {})["api_key"] = key
    if key := os.getenv("GATEWAY_API_KEY"):
        data.setdefault("gateway", {})["api_key"] = key
    if model := os.getenv("DEFAULT_MODEL"):
        for ag in data.get("agents", []):
            ag.setdefault("model", {})["primary"] = model

    cfg = AppConfig(**data)
    cfg.config_dir = str(path.parent.resolve())

    # Inject keys from config into os.environ (won't override existing env vars)
    for k, v in cfg.keys.items():
        if k not in os.environ:
            os.environ[k] = str(v)

    # Ensure at least one agent
    if not cfg.agents:
        cfg.agents = [AgentConfig(id="default")]

    return cfg
