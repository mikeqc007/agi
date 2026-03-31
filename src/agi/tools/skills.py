from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Any

from agi.tools.registry import tool
from agi.types import ToolContext


@tool
async def read_skill(ctx: ToolContext, name: str) -> str:
    """Read a skill's instructions. Call this when a skill applies to the user's request.

    name: skill name to read
    """
    app = ctx.app
    skill_manager = getattr(app, "skill_manager", None) if app else None
    agent_dir: Path | None = None
    if app and hasattr(app, "cfg"):
        cfg = app.cfg
        agent_id = getattr(ctx, "agent_id", None)
        if agent_id and cfg.config_dir:
            agent_dir = Path(cfg.config_dir) / cfg.memory.memory_dir / "agents" / agent_id / "skills"
    if skill_manager is None:
        return f"Skill '{name}' not found."
    content = skill_manager.read_skill(name, agent_dir)
    if content is None:
        return f"Skill '{name}' not found."
    return content

logger = logging.getLogger(__name__)

MAX_SKILL_SIZE = 256 * 1024


class SkillManager:
    """Loads skills from skills_dir.

    Each skill is a subdirectory containing a SKILL.md and an optional
    scripts/ directory with executable helpers:

        skills/
          summarize/
            SKILL.md
          deploy/
            SKILL.md
            scripts/
              run.sh
              build.py
    """

    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        self._disabled: set[str] = set()

    def list_skills(self, agent_dir: Path | None = None) -> list[dict]:
        """Return metadata for all discovered skills (shared + agent-specific)."""
        skills: dict[str, dict] = {}
        self._dir.mkdir(parents=True, exist_ok=True)

        for entry in sorted(self._dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue
            name = entry.name
            desc, _frontmatter = _parse_skill_md(skill_file)
            skills[name] = {
                "name": name,
                "enabled": name not in self._disabled,
                "description": desc,
                "path": str(skill_file),
                "base_dir": str(entry.resolve()),
                "scripts": _list_scripts(entry),
            }

        if agent_dir and agent_dir.exists():
            for entry in sorted(agent_dir.iterdir()):
                if not entry.is_dir():
                    continue
                skill_file = entry / "SKILL.md"
                if not skill_file.exists():
                    continue
                name = entry.name
                desc, _frontmatter = _parse_skill_md(skill_file)
                skills[name] = {
                    "name": name,
                    "enabled": name not in self._disabled,
                    "description": desc,
                    "path": str(skill_file),
                    "base_dir": str(entry.resolve()),
                    "scripts": _list_scripts(entry),
                }

        return list(skills.values())

    def read_skill(self, name: str, agent_dir: Path | None = None) -> str | None:
        """Return SKILL.md content with {baseDir} resolved to the skill directory."""
        for skill in self.list_skills(agent_dir):
            if skill["name"] == name:
                path = Path(skill["path"])
                base_dir = skill["base_dir"]
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    return None
                return content.replace("{baseDir}", base_dir)
        return None

    def build_prompt(self, agent_dir: Path | None = None) -> str:
        skills = [s for s in self.list_skills(agent_dir) if s["enabled"]]
        if not skills:
            return ""
        lines = ["<available_skills>"]
        for s in skills:
            lines.append(f'<skill name="{s["name"]}">')
            lines.append(f'Description: {s["description"]}')
            lines.append(f'To use: call read_skill("{s["name"]}")')
            if s["scripts"]:
                lines.append(f'Scripts: {", ".join(s["scripts"])}')
            lines.append("</skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def toggle(self, name: str) -> bool:
        if name in self._disabled:
            self._disabled.discard(name)
            return True
        else:
            self._disabled.add(name)
            return False


def _list_scripts(skill_dir: Path) -> list[str]:
    """Return absolute paths of all scripts in skill_dir/scripts/, chmod +x each."""
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return []
    result = []
    for f in sorted(scripts_dir.iterdir()):
        if f.is_file():
            _ensure_executable(f)
            result.append(str(f.resolve()))
    return result


def _ensure_executable(path: Path) -> None:
    try:
        current = os.stat(path).st_mode
        os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP)
    except Exception:
        pass


def _parse_skill_md(path: Path) -> tuple[str, dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", {}

    frontmatter: dict[str, Any] = {}
    body = text

    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            import yaml
            try:
                frontmatter = yaml.safe_load(text[3:end]) or {}
            except Exception:
                pass
            body = text[end + 4:]

    desc = frontmatter.get("description", "")
    if not desc:
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("#"):
                desc = line.lstrip("#").strip()
                break
            if line:
                desc = line
                break

    return desc, frontmatter
