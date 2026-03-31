from __future__ import annotations

"""Browser automation tool — mirrors openclaw's browser tool action set.

Single tool with action= discriminator, matching openclaw's:
  status / start / stop / profiles / tabs / open / focus / close /
  snapshot / screenshot / navigate / console / pdf / upload / dialog / act

act kinds: click / type / press / hover / scrollIntoView / drag / fill /
           select / evaluate / wait / resize / close
"""

from typing import Any

from agi.browser.manager import get_browser_manager
from agi.tools.registry import tool
from agi.types import ToolContext


@tool
async def browser(
    ctx: ToolContext,
    action: str = "status",
    profile: str = "openclaw",
    headless: bool = None,
    target_id: str = "",
    target_url: str = "",
    target: str = "",   # alias for target_url
    url: str = "",      # alias for target_url
    href: str = "",     # alias for target_url
    full_page: bool = False,
    max_chars: int = 20000,
    snapshot_format: str = "ai",
    refs: str = "role",
    interactive: bool = False,
    compact: bool = False,
    depth: int = -1,
    ref: str = "",
    element: str = "",
    image_type: str = "png",
    level: str = "",
    timeout_ms: int = 10000,
    accept: bool = True,
    prompt_text: str = "",
    paths: list[str] = None,
    input_ref: str = "",
    request: dict = None,
) -> Any:
    """Control the browser for web automation and information extraction.

    action values:
    - status    : get browser status and list of open tabs
    - start     : launch browser; headless auto-detected from DISPLAY env (pass headless=true/false to override)
    - stop      : close browser
    - profiles  : list supported browser profiles
    - tabs      : list open tabs with targetId, url, title
    - open      : open new tab at target_url; returns targetId
    - focus     : bring tab to front (target_id required)
    - close     : close tab (target_id optional; closes active tab if omitted)
    - snapshot  : read page content; snapshot_format=ai or aria, with refs map for automation
    - screenshot: capture page screenshot; supports element/ref and type=png/jpeg
    - navigate  : navigate current/specified tab to target_url
    - console   : get browser console messages (optional level filter)
    - pdf       : save current page as PDF; returns file path
    - upload    : set input files on file input element
    - dialog    : arm next dialog action (accept/dismiss)
    - act       : perform UI action via request object (see below)

    act request object fields:
      kind       : click / type / press / hover / scrollIntoView / drag / fill / select / evaluate / wait / resize / close
      ref        : CSS selector or text selector, e.g. 'button:text("Login")', '#username'
      text       : text to type (kind=type)
      submit     : bool, press Enter after typing (kind=type)
      slowly     : bool, type with delay (kind=type)
      key        : key name, e.g. 'Enter', 'Tab', 'Escape' (kind=press)
      delayMs    : key press delay in ms (kind=press)
      doubleClick: bool (kind=click)
      button     : left/right/middle (kind=click)
      modifiers  : list of 'Shift'/'Ctrl'/'Alt' (kind=click)
      startRef   : source selector (kind=drag)
      endRef     : destination selector (kind=drag)
      fields     : [{ref, value}, ...] for form fill (kind=fill)
      values     : [str, ...] for dropdown (kind=select)
      fn         : JavaScript expression string (kind=evaluate)
      timeMs     : milliseconds to wait (kind=wait)
      text       : text to wait for (kind=wait, waits until visible)
      textGone   : text to wait until hidden (kind=wait)
      selector   : CSS selector to wait for visible (kind=wait)
      url        : URL/pattern to wait_for_url (kind=wait)
      loadState  : load/domcontentloaded/networkidle (kind=wait)
      fn         : js expression to evaluate as wait condition (kind=wait)
      width/height: viewport size (kind=resize)
      timeoutMs  : action timeout ms (default 10000)

    Workflow example:
      1. action=start
      2. action=open target_url=https://example.com
      3. action=snapshot  → read page to find selectors
      4. action=act request={kind:click, ref:'button:text("Login")'}
      5. action=screenshot  → verify result visually
    """
    import re as _re

    mgr = get_browser_manager()

    # Clean up malformed action strings from models that embed XML-style args in the action field
    # e.g. action='navigate\n<arg_key>url</arg_key>\n<arg_value>https://...'
    if '\n' in action:
        lines = action.split('\n')
        action = lines[0].strip()
        rest = '\n'.join(lines[1:])
        # Extract all <arg_key>K</arg_key><arg_value>V</arg_value> pairs
        _xml_pairs = _re.findall(
            r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', rest, _re.DOTALL
        )
        for _k, _v in _xml_pairs:
            _k, _v = _k.strip(), _v.strip()
            if _k in ('url', 'target_url', 'href', 'target') and not (target_url or url or href or target):
                url = _v
            elif _k == 'snapshot_format':
                snapshot_format = _v
            elif _k == 'ref':
                ref = _v
            elif _k == 'max_chars' and _v.isdigit():
                max_chars = int(_v)
            elif _k == 'interactive':
                interactive = _v.lower() in ('true', '1', 'yes')
            elif _k == 'compact':
                compact = _v.lower() in ('true', '1', 'yes')
            elif _k == 'full_page':
                full_page = _v.lower() in ('true', '1', 'yes')
            elif _k == 'target_id' and not target_id:
                target_id = _v

    # Accept common aliases for target_url
    target_url = target_url or target or url or href

    # If element looks like a tab ID (e.g. "tab-2"), the model confused target_id with element
    if element and _re.match(r'^tab-\d+$', element) and not target_id:
        target_id = element
        element = ""

    tid = target_id or None

    # Normalize common action aliases the model uses FIRST, then check if browser needs starting
    _action_aliases = {
        "goto": "navigate",
        "go_to": "navigate",
        "visit": "navigate",
        "load": "navigate",
        "go": "navigate",
        "get": "navigate",
        "new_tab": "open",
        "new_page": "open",
        "capture": "screenshot",
        "screen": "screenshot",
        "install": None,   # not a real action — will fall through to error
    }
    action = _action_aliases.get(action, action)

    # Handle action='click'/'type'/'press' etc: redirect to act with appropriate request
    if action in ("click", "type", "press", "hover", "fill", "select", "evaluate", "wait"):
        if not request:
            request = {"kind": action}
            if ref:
                request["ref"] = ref
                ref = ""
        action = "act"

    # Handle action='search': construct DuckDuckGo Lite URL from prompt_text or target_url
    # DuckDuckGo Lite is plain HTML with no bot detection, ideal for browser automation
    if action == "search":
        query = prompt_text or target_url or target or url or href or ""
        if query:
            import urllib.parse as _urlparse
            target_url = f"https://lite.duckduckgo.com/lite/?q={_urlparse.quote(query)}"
            url = href = target = ""
        action = "navigate"

    # Auto-start browser if not running and a navigation/content action is requested
    _needs_browser = action in (
        "tabs", "open", "focus", "close", "navigate", "snapshot",
        "screenshot", "console", "pdf", "upload", "dialog", "act",
    )

    if _needs_browser and not mgr._started:
        await mgr.start(headless=headless, profile=profile)
        # If a URL is also provided with a content action, navigate automatically
        if target_url and action in ("snapshot", "screenshot"):
            await mgr.navigate(target_url, None)

    if action == "status":
        return await mgr.status()

    elif action == "start":
        return await mgr.start(headless=headless, profile=profile)

    elif action == "stop":
        return await mgr.stop()

    elif action == "profiles":
        return await mgr.profiles()

    elif action == "tabs":
        return {"tabs": await mgr.tabs()}

    elif action == "open":
        if not target_url:
            return {"error": "target_url required for action=open"}
        return await mgr.open_tab(target_url)

    elif action == "focus":
        if not tid:
            return {"error": "target_id required for action=focus"}
        return await mgr.focus_tab(tid)

    elif action == "close":
        return await mgr.close_tab(tid)

    elif action == "navigate":
        if not target_url:
            return {"error": "target_url required for action=navigate"}
        return await mgr.navigate(target_url, tid)

    elif action == "snapshot":
        return await mgr.snapshot(
            tid,
            max_chars,
            snapshot_format=snapshot_format,
            refs_mode=refs,
            interactive=interactive,
            compact=compact,
            depth=None if depth < 0 else depth,
        )

    elif action == "screenshot":
        result = await mgr.screenshot(
            tid,
            full_page=full_page,
            ref=ref,
            image_type=image_type,
            element=element,
        )
        # Return image data for vision models
        return result

    elif action == "console":
        return await mgr.console_messages(tid, level=level)

    elif action == "pdf":
        return await mgr.pdf(tid)

    elif action == "upload":
        if not paths:
            return {"error": "paths required for action=upload"}
        return await mgr.upload(
            paths=paths,
            target_id=tid,
            ref=ref,
            input_ref=input_ref,
            element=element,
            timeout_ms=timeout_ms,
        )

    elif action == "dialog":
        return await mgr.dialog(
            accept=accept,
            prompt_text=prompt_text,
            target_id=tid,
            timeout_ms=timeout_ms,
        )

    elif action == "act":
        if not request:
            return {
                "error": (
                    "request object required for action=act. "
                    "Example: {\"kind\": \"click\", \"ref\": \"button:text('Submit')\"}"
                )
            }
        return await mgr.act(request, tid)

    else:
        return {
            "error": (
                f"Unknown action: {action!r}. "
                "Valid: status/start/stop/profiles/tabs/open/focus/close/"
                "snapshot/screenshot/navigate/console/pdf/upload/dialog/act"
            )
        }
