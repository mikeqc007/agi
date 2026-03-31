from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import sys
from io import BytesIO
from typing import Any

from agi.tools.registry import tool
from agi.types import ToolContext

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
VISION_MAX_SIZE = (1280, 800)  # resize screenshot before sending to LLM


@tool
async def screenshot(ctx: ToolContext) -> str:
    """Take a screenshot of the current screen and return description."""
    img_b64 = await _take_screenshot_b64()
    if not img_b64:
        return "Error: could not take screenshot"

    model = ctx.app.cfg.agents[0].model.primary
    description = await _vision_describe(img_b64, model,
        "Describe what is visible on screen: app names, window titles, UI elements, text content.")
    return description


@tool
async def computer_use(ctx: ToolContext, instruction: str) -> str:
    """Perform a computer action by describing what to do in natural language.

    instruction: What to do, e.g. 'click the Send button', 'type hello world', 'open Chrome'
    """
    for attempt in range(MAX_RETRIES):
        img_b64 = await _take_screenshot_b64()
        if not img_b64:
            return "Error: could not take screenshot"

        model = ctx.app.cfg.agents[0].model.primary
        action = await _vision_plan_action(img_b64, model, instruction)

        if not action:
            return "Error: could not determine action from screenshot"

        result = await _execute_action(action)

        if result.get("success"):
            # Verify with follow-up screenshot
            await asyncio.sleep(0.5)
            verify_b64 = await _take_screenshot_b64()
            if verify_b64:
                verification = await _vision_describe(
                    verify_b64, model,
                    f"Did this action succeed: '{instruction}'? Describe what changed."
                )
                return f"Action executed. Result: {verification}"
            return f"Action executed: {action.get('type')} {action.get('target', '')}"

        if attempt < MAX_RETRIES - 1:
            logger.debug("Action failed (attempt %d), retrying: %s", attempt + 1, result)
            await asyncio.sleep(0.5)

    return f"Failed to complete: {instruction}"


@tool
async def open_app(ctx: ToolContext, app_name: str) -> str:
    """Open an application by name.

    app_name: Application name, e.g. 'Chrome', 'Terminal', 'VSCode'
    """
    try:
        if sys.platform == "darwin":
            proc = await asyncio.create_subprocess_exec(
                "open", "-a", app_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        elif sys.platform == "win32":
            proc = await asyncio.create_subprocess_shell(
                f'start "" "{app_name}"',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        else:  # Linux
            # Try common launchers
            for cmd in [app_name.lower(), app_name.lower().replace(" ", "-"),
                        app_name.lower().replace(" ", "")]:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.sleep(1)
                    return f"Launched {app_name}"
                except FileNotFoundError:
                    continue
            return f"Could not find application: {app_name}"

        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            return f"Error opening {app_name}: {stderr.decode().strip()}"
        return f"Opened {app_name}"
    except Exception as e:
        return f"Error: {e}"


@tool
async def mouse_click(ctx: ToolContext, x: int, y: int, button: str = "left") -> str:
    """Click at specific screen coordinates.

    x: X coordinate in pixels
    y: Y coordinate in pixels
    button: Mouse button: left/right/double (default: left)
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        if button == "double":
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pyautogui.doubleClick(x, y)
            )
        elif button == "right":
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pyautogui.rightClick(x, y)
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pyautogui.click(x, y)
            )
        return f"Clicked ({x}, {y}) with {button} button"
    except Exception as e:
        return f"Error: {e}"


@tool
async def keyboard_type(ctx: ToolContext, text: str) -> str:
    """Type text using the keyboard.

    text: Text to type
    """
    try:
        import pyautogui
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.write(text, interval=0.03)
        )
        return f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}"
    except Exception as e:
        return f"Error: {e}"


@tool
async def keyboard_hotkey(ctx: ToolContext, keys: str) -> str:
    """Press a keyboard hotkey combination.

    keys: Keys separated by +, e.g. 'ctrl+c', 'cmd+space', 'alt+f4'
    """
    try:
        import pyautogui
        key_list = [k.strip().lower() for k in keys.split("+")]
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.hotkey(*key_list)
        )
        return f"Pressed hotkey: {keys}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _take_screenshot_b64() -> str | None:
    try:
        import mss
        from PIL import Image

        def _capture() -> str:
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # all screens combined
                img = sct.grab(monitor)
                pil = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
                # Resize for LLM
                pil.thumbnail(VISION_MAX_SIZE, Image.LANCZOS)
                buf = BytesIO()
                pil.save(buf, format="JPEG", quality=75)
                return base64.b64encode(buf.getvalue()).decode()

        return await asyncio.get_event_loop().run_in_executor(None, _capture)
    except Exception as e:
        logger.warning("Screenshot failed: %s", e)
        return None


async def _vision_describe(img_b64: str, model: str, prompt: str) -> str:
    try:
        import litellm
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Vision error: {e}"


async def _vision_plan_action(img_b64: str, model: str, instruction: str) -> dict | None:
    prompt = f"""You are controlling a computer. Look at the screenshot and determine the exact action needed.

Instruction: {instruction}

Reply with ONLY a JSON object (no explanation):
{{
  "type": "click" | "type" | "hotkey" | "scroll",
  "target": "description of target element",
  "x": <pixel x if clicking>,
  "y": <pixel y if clicking>,
  "text": "<text to type if typing>",
  "keys": "<keys if hotkey, e.g. ctrl+c>",
  "direction": "up|down",
  "amount": <scroll amount>
}}"""

    try:
        import litellm
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=256,
            stream=False,
        )
        raw = response.choices[0].message.content or "{}"
        import re
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        logger.debug("Vision plan failed: %s", e)
    return None


async def _execute_action(action: dict) -> dict:
    action_type = action.get("type", "")
    try:
        import pyautogui
        pyautogui.FAILSAFE = False

        loop = asyncio.get_event_loop()

        if action_type == "click":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            await loop.run_in_executor(None, lambda: pyautogui.click(x, y))

        elif action_type == "type":
            text = str(action.get("text", ""))
            await loop.run_in_executor(None, lambda: pyautogui.write(text, interval=0.03))

        elif action_type == "hotkey":
            keys = [k.strip() for k in str(action.get("keys", "")).split("+")]
            await loop.run_in_executor(None, lambda: pyautogui.hotkey(*keys))

        elif action_type == "scroll":
            direction = action.get("direction", "down")
            amount = int(action.get("amount", 3))
            clicks = amount if direction == "up" else -amount
            await loop.run_in_executor(None, lambda: pyautogui.scroll(clicks))

        return {"success": True, "action": action_type}
    except Exception as e:
        return {"success": False, "error": str(e)}
