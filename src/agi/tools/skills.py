from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max SKILL.md size to include in skill listing
MAX_SKILL_SIZE = 256 * 1024


class SkillManager:
    """Loads skills from skills_dir.

    Each skill is a subdirectory containing a SKILL.md file:
        skills/
          menu/
            SKILL.md
          research/
            SKILL.md

    Skills appear in the agent system prompt so the model can decide when to
    read and follow a skill's SKILL.md via the read_file tool.
    """

    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        self._disabled: set[str] = set()

    def list_skills(self, agent_dir: Path | None = None) -> list[dict]:
        """Return metadata for all discovered skills (shared + agent-specific).

        Agent-specific skills override shared skills of the same name.
        """
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
                }

        return list(skills.values())

    def build_prompt(self, agent_dir: Path | None = None) -> str:
        """Build the skills block for the system prompt.

        openclaw-style: list each skill with its path so the model can
        call read_file to fetch the full SKILL.md when the skill applies.
        """
        skills = [s for s in self.list_skills(agent_dir) if s["enabled"]]
        if not skills:
            return ""
        lines = ["<available_skills>"]
        for s in skills:
            lines.append(
                f"<skill name=\"{s['name']}\">\n"
                f"Description: {s['description']}\n"
                f"To use: call read_file(\"{s['path']}\")\n"
                f"</skill>"
            )
        lines.append("</available_skills>")
        return "\n".join(lines)

    def toggle(self, name: str) -> bool:
        if name in self._disabled:
            self._disabled.discard(name)
            return True
        else:
            self._disabled.add(name)
            return False


def _parse_skill_md(path: Path) -> tuple[str, dict]:
    """Parse SKILL.md: extract description from frontmatter or first heading.
    Returns (description, frontmatter_dict).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", {}

    frontmatter: dict[str, Any] = {}
    body = text

    # Parse YAML frontmatter between --- markers
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            import yaml
            try:
                frontmatter = yaml.safe_load(text[3:end]) or {}
            except Exception:
                pass
            body = text[end + 4:]

    # Description: frontmatter > first # heading > first non-empty line
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
