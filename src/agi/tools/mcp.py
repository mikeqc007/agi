from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_LANG_MAP = {"zh": "zh-CN", "tw": "zh-TW", "en": "en-US",
             "ja": "ja-JP", "ko": "ko-KR", "vi": "vi-VN"}


def _normalize_args(arguments: dict) -> dict:
    """Common argument normalization applied to all MCP tool calls."""
    # language shorthand → BCP-47 (common convention across MCP servers)
    if "language" in arguments:
        arguments = {**arguments,
                     "language": _LANG_MAP.get(arguments["language"], arguments["language"])}
    return arguments



class MCPClient:
    """Single MCP server connection via stdio."""

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self._session: Any = None
        self._tools: list[dict] = []

    async def start(self) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client, StdioServerParameters
            import os

            env_full = {**os.environ, **(self.env or {})}
            params = StdioServerParameters(
                command=self.command, args=self.args, env=env_full
            )
            self._ctx = stdio_client(params)
            read, write = await self._ctx.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            await self._refresh_tools()
            logger.info("MCP server '%s' connected, %d tools", self.name, len(self._tools))
        except Exception as e:
            logger.warning("MCP server '%s' failed to start: %s", self.name, e)
            self._session = None

    async def stop(self) -> None:
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
            if hasattr(self, "_ctx"):
                await self._ctx.__aexit__(None, None, None)
        except Exception:
            pass
        self._session = None

    async def _refresh_tools(self) -> None:
        if not self._session:
            return
        result = await self._session.list_tools()
        self._tools = []
        for t in result.tools:
            self._tools.append({
                "name": f"mcp_{self.name}_{t.name}",
                "original_name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            })

    def get_schemas(self) -> list[dict]:
        schemas = []
        for t in self._tools:
            desc = t["description"] or t["original_name"]
            # Append parameter names to help small models match tool to user intent
            props = (t["input_schema"] or {}).get("properties", {})
            if props:
                param_hint = ", ".join(props.keys())
                desc = f"{desc} (params: {param_hint})"
            schemas.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": f"[MCP:{self.name}] {desc}",
                    "parameters": t["input_schema"],
                },
            })
        return schemas

    async def call(self, tool_name: str, arguments: dict) -> str:
        if not self._session:
            return f"Error: MCP server '{self.name}' not connected"
        original = tool_name.removeprefix(f"mcp_{self.name}_")
        arguments = _normalize_args(arguments)
        logger.info("MCP call: %s.%s args=%s", self.name, original, arguments)
        try:
            result = await self._session.call_tool(original, arguments)
            text = self._result_to_text(result)
            # Re-encode JSON with ensure_ascii=False so Chinese chars are readable to the model
            try:
                import json as _json
                parsed = _json.loads(text)
                text = _json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                pass
            logger.info("MCP result: %s.%s → %s", self.name, original, text[:200])
            return text
        except Exception as e:
            logger.info("MCP error: %s.%s → %s", self.name, original, e)
            return f"Error calling MCP tool {original}: {e}"

    @staticmethod
    def _result_to_text(result: Any) -> str:
        parts = []
        for c in result.content:
            if hasattr(c, "text"):
                parts.append(c.text)
            else:
                parts.append(str(c))
        return "\n".join(parts) if parts else "(no output)"


class MCPManager:
    """Manages multiple MCP server connections."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    def add_server(self, name: str, command: str,
                   args: list[str] | None = None,
                   env: dict[str, str] | None = None) -> None:
        self._clients[name] = MCPClient(name, command, args, env)

    async def start_all(self) -> None:
        await asyncio.gather(
            *[c.start() for c in self._clients.values()],
            return_exceptions=True,
        )

    async def stop_all(self) -> None:
        await asyncio.gather(
            *[c.stop() for c in self._clients.values()],
            return_exceptions=True,
        )

    def get_all_schemas(self) -> list[dict]:
        schemas = []
        for c in self._clients.values():
            schemas.extend(c.get_schemas())
        return schemas

    async def call(self, tool_name: str, arguments: dict) -> str:
        """Route a tool call to the correct MCP client."""
        for name, client in self._clients.items():
            if tool_name.startswith(f"mcp_{name}_"):
                return await client.call(tool_name, arguments)
        return f"Error: no MCP client found for tool '{tool_name}'"

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name.startswith("mcp_")
