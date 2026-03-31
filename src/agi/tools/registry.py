from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from agi.types import ToolContext

logger = logging.getLogger(__name__)

_registry: dict[str, tuple[Callable, dict]] = {}


def tool(func: Callable) -> Callable:
    """Decorator to register a function as an agent tool."""
    schema = _build_schema(func)
    _registry[func.__name__] = (func, schema)
    return func


def get_all_schemas() -> list[dict]:
    return [
        {"type": "function", "function": schema}
        for _, schema in _registry.values()
    ]


async def dispatch(name: str, ctx: ToolContext, args: dict) -> Any:
    entry = _registry.get(name)
    if entry is None:
        return {"error": f"Unknown tool: {name}"}
    fn, _ = entry
    try:
        return await fn(ctx, **args)
    except TypeError as e:
        return {"error": f"Tool call error: {e}"}


def _build_schema(func: Callable) -> dict:
    """Build OpenAI-style function schema from function signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n")[0] if doc else func.__name__

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "ctx":
            continue
        ann = param.annotation
        prop: dict[str, Any] = {"type": _py_type_to_json(ann)}

        # Extract param description from docstring
        for line in doc.split("\n"):
            if line.strip().startswith(f"{pname}:"):
                prop["description"] = line.strip()[len(pname) + 1:].strip()
                break

        if param.default is inspect.Parameter.empty:
            required.append(pname)

        properties[pname] = prop

    schema: dict[str, Any] = {
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required
    return schema


def _py_type_to_json(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return "string"
    origin = getattr(ann, "__origin__", None)
    if origin is list:
        return "array"
    MAP = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}
    return MAP.get(ann, "string")
