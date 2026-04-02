from __future__ import annotations

# Tool permission levels — same model as Claude Code
# read_only:          safe read/search operations, no side effects
# workspace_write:    modifies files in workspace
# danger_full_access: shell execution, desktop control, browser automation

TOOL_LEVELS: dict[str, str] = {
    # read-only
    "read_file":        "read_only",
    "list_dir":         "read_only",
    "glob":             "read_only",
    "grep":             "read_only",
    "web_fetch":        "read_only",
    "web_search":       "read_only",
    "recall":           "read_only",
    "list_subagents":   "read_only",
    "read_skill":       "read_only",
    "say":              "read_only",
    "todo":             "read_only",
    # workspace write
    "write_file":       "workspace_write",
    "edit_file":        "workspace_write",
    "remember":         "workspace_write",
    "forget":           "workspace_write",
    "cron_add":         "workspace_write",
    "cron_delete":      "workspace_write",
    "spawn_agent":      "workspace_write",
    "kill_subagent":    "workspace_write",
    # danger
    "shell":            "danger_full_access",
    "screenshot":       "danger_full_access",
    "computer_use":     "danger_full_access",
    "mouse_click":      "danger_full_access",
    "keyboard_type":    "danger_full_access",
    "browser":          "danger_full_access",
}

_LEVEL_RANK = {"read_only": 0, "workspace_write": 1, "danger_full_access": 2}

# permission_mode → maximum allowed level
_MODE_MAX: dict[str, str] = {
    "read_only":          "read_only",
    "workspace_write":    "workspace_write",
    "allow":              "danger_full_access",
    "danger_full_access": "danger_full_access",
    # "prompt" is handled separately in the loop
}


def get_tool_level(tool_name: str) -> str:
    """Return the permission level required by a tool."""
    # MCP tools default to workspace_write
    if tool_name.startswith("mcp__"):
        return "workspace_write"
    return TOOL_LEVELS.get(tool_name, "workspace_write")


def is_allowed_by_mode(tool_name: str, permission_mode: str) -> bool:
    """Return True if the tool is allowed under the given permission mode."""
    if permission_mode == "prompt":
        return True  # prompt mode allows all; loop handles confirmation
    max_level = _MODE_MAX.get(permission_mode, "danger_full_access")
    tool_level = get_tool_level(tool_name)
    return _LEVEL_RANK.get(tool_level, 1) <= _LEVEL_RANK.get(max_level, 2)


def needs_prompt(tool_name: str, permission_mode: str) -> bool:
    """Return True if this tool requires user confirmation in prompt mode."""
    if permission_mode != "prompt":
        return False
    return get_tool_level(tool_name) == "danger_full_access"
