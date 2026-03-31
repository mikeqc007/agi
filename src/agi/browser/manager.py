from __future__ import annotations

"""Playwright browser manager.

Manages a persistent browser + context for automation.
Mirrors openclaw's browser server actions:
status/start/stop/profiles/tabs/open/focus/close/snapshot/screenshot/
navigate/console/pdf/upload/dialog/act.

Setup: pip install playwright && playwright install chromium
"""

import asyncio
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Singleton instance
_instance: BrowserManager | None = None


def get_browser_manager() -> "BrowserManager":
    global _instance
    if _instance is None:
        _instance = BrowserManager()
    return _instance


class BrowserManager:
    """Wraps Playwright — one browser, one context, N pages (tabs)."""

    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._pages: dict[str, Any] = {}   # targetId -> Page
        self._page_counter = 0
        self._active_target_id: str | None = None
        self._console_logs: dict[str, list[dict[str, Any]]] = {}
        self._dialog_hooks: dict[str, dict[str, Any]] = {}
        self._role_refs_by_target: dict[str, dict[str, dict[str, Any]]] = {}
        self._role_refs_mode_by_target: dict[str, str] = {}
        self._profile = "openclaw"
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, headless: bool | None = None, profile: str = "openclaw") -> dict:
        if self._started:
            return await self.status()
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        self._profile = profile or "openclaw"
        if headless is None:
            # Auto-detect: headed only when a display is available (DISPLAY or WAYLAND_DISPLAY set)
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            headless = not has_display
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
        )
        page = await self._context.new_page()
        self._register_page(page)
        self._started = True
        logger.info("Browser started (headless=%s, profile=%s)", headless, self._profile)
        return await self.status()

    async def stop(self) -> dict:
        if not self._started:
            return {"running": False}
        try:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.debug("Browser stop error: %s", e)
        self._browser = None
        self._context = None
        self._pages = {}
        self._console_logs = {}
        self._dialog_hooks = {}
        self._role_refs_by_target = {}
        self._role_refs_mode_by_target = {}
        self._active_target_id = None
        self._started = False
        logger.info("Browser stopped")
        return {"running": False, "stopped": True}

    async def status(self) -> dict:
        tabs = []
        for tid, page in self._pages.items():
            tabs.append({
                "targetId": tid,
                "url": page.url,
                "title": await _safe_title(page),
            })
        return {
            "running": self._started,
            "profile": self._profile,
            "tabCount": len(self._pages),
            "activeTargetId": self._active_target_id,
            "tabs": tabs,
        }

    async def profiles(self) -> dict:
        # Keep contract aligned with openclaw action. We expose a managed profile
        # and explicitly mark chrome relay as unsupported.
        return {
            "profiles": [
                {"id": "openclaw", "running": self._started, "supported": True},
                {"id": "chrome", "running": False, "supported": False},
            ]
        }

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    async def tabs(self) -> list[dict]:
        result = []
        for tid, page in self._pages.items():
            result.append({
                "targetId": tid,
                "url": page.url,
                "title": await _safe_title(page),
            })
        return result

    async def open_tab(self, url: str) -> dict:
        self._ensure_started()
        page = await self._context.new_page()
        tid = self._register_page(page)
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        return {"targetId": tid, "url": page.url, "title": await _safe_title(page)}

    async def focus_tab(self, target_id: str) -> dict:
        page = self._get_page(target_id)
        await page.bring_to_front()
        self._active_target_id = target_id
        return {"ok": True, "targetId": target_id}

    async def close_tab(self, target_id: str | None = None) -> dict:
        self._ensure_started()
        if target_id:
            page = self._pages.pop(target_id, None)
            if page:
                await page.close()
                self._console_logs.pop(target_id, None)
                self._role_refs_by_target.pop(target_id, None)
                self._role_refs_mode_by_target.pop(target_id, None)
                if self._active_target_id == target_id:
                    self._active_target_id = None
        elif self._pages:
            tid, page = list(self._pages.items())[-1]
            del self._pages[tid]
            await page.close()
            self._console_logs.pop(tid, None)
            self._role_refs_by_target.pop(tid, None)
            self._role_refs_mode_by_target.pop(tid, None)
            if self._active_target_id == tid:
                self._active_target_id = None
        if not self._active_target_id and self._pages:
            self._active_target_id = list(self._pages.keys())[-1]
        return {"ok": True}

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate(self, url: str, target_id: str | None = None) -> dict:
        page = self._get_page(target_id)
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        resolved_target = self._resolve_target_id(target_id)
        return {
            "ok": True,
            "targetId": resolved_target,
            "url": page.url,
            "title": await _safe_title(page),
        }

    # ------------------------------------------------------------------
    # Snapshot — accessibility tree (like openclaw's "ai" format)
    # ------------------------------------------------------------------

    async def snapshot(
        self,
        target_id: str | None = None,
        max_chars: int = 20_000,
        snapshot_format: str = "ai",
        refs_mode: str = "role",
        interactive: bool = False,
        compact: bool = False,
        depth: int | None = None,
    ) -> dict:
        page = self._get_page(target_id)
        tid = self._resolve_target_id(target_id)
        options = {
            "interactive": bool(interactive),
            "compact": bool(compact),
            "max_depth": depth if isinstance(depth, int) and depth >= 0 else None,
        }

        if snapshot_format == "aria":
            ax = await page.accessibility.snapshot()
            nodes = _flatten_ax(ax, limit=2000)
            return {
                "ok": True,
                "format": "aria",
                "targetId": tid,
                "url": page.url,
                "title": await _safe_title(page),
                "nodes": nodes,
            }

        try:
            ax = await page.accessibility.snapshot()
            built = _build_role_snapshot_from_ax(ax, options)
            text = built["snapshot"]
            refs = built["refs"]
            self._role_refs_by_target[tid] = refs
            # Python Playwright does not expose aria-ref id flow like Node openclaw.
            self._role_refs_mode_by_target[tid] = "role" if refs_mode in ("role", "aria") else "role"
        except Exception:
            # Fallback: visible text
            try:
                text = await page.inner_text("body")
            except Exception:
                text = ""
            refs = {}
            built = {"stats": {"lines": len(text.splitlines()), "chars": len(text), "refs": 0, "interactive": 0}}

        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n[...truncated at {max_chars} chars]"
            truncated = True

        return {
            "ok": True,
            "format": "ai",
            "targetId": tid,
            "url": page.url,
            "title": await _safe_title(page),
            "snapshot": text,
            "truncated": truncated,
            "refs": refs,
            "stats": built.get("stats", {}),
        }

    # ------------------------------------------------------------------
    # Screenshot — returns base64 PNG
    # ------------------------------------------------------------------

    async def screenshot(
        self,
        target_id: str | None = None,
        full_page: bool = False,
        ref: str = "",
        image_type: str = "png",
        element: str = "",
    ) -> dict:
        import base64
        page = self._get_page(target_id)
        tid = self._resolve_target_id(target_id)

        selector = element or ref
        shot_type = "jpeg" if image_type == "jpeg" else "png"
        if selector:
            try:
                locator = self._ref_locator(page, tid, selector)
                data = await locator.screenshot(type=shot_type, timeout=5000)
            except Exception:
                # Selector not found or invalid — fall back to full-page screenshot
                logger.warning("Screenshot selector %r failed, falling back to full page", selector)
                data = await page.screenshot(full_page=full_page, type=shot_type)
        else:
            data = await page.screenshot(full_page=full_page, type=shot_type)

        # Match openclaw-style path return while preserving current base64 output.
        suffix = ".jpg" if shot_type == "jpeg" else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(data)
            image_path = f.name

        b64 = base64.b64encode(data).decode()
        mime = "image/jpeg" if shot_type == "jpeg" else "image/png"
        return {
            "ok": True,
            "targetId": tid,
            "base64": b64,
            "mime_type": mime,
            "path": image_path,
            "url": page.url,
        }

    # ------------------------------------------------------------------
    # Console messages
    # ------------------------------------------------------------------

    async def console_messages(self, target_id: str | None = None, level: str = "") -> dict:
        tid = self._resolve_target_id(target_id)
        messages = list(self._console_logs.get(tid, []))
        level = level.strip().lower()
        if level:
            messages = [m for m in messages if m.get("level", "").lower() == level]
        return {"ok": True, "targetId": tid, "messages": messages}

    # ------------------------------------------------------------------
    # PDF save
    # ------------------------------------------------------------------

    async def pdf(self, target_id: str | None = None) -> dict:
        page = self._get_page(target_id)
        tid = self._resolve_target_id(target_id)
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        await page.pdf(path=tmp)
        return {"ok": True, "targetId": tid, "path": tmp}

    # ------------------------------------------------------------------
    # Upload files
    # ------------------------------------------------------------------

    async def upload(
        self,
        paths: list[str],
        target_id: str | None = None,
        ref: str = "",
        input_ref: str = "",
        element: str = "",
        timeout_ms: int = 10_000,
    ) -> dict:
        page = self._get_page(target_id)
        tid = self._resolve_target_id(target_id)
        selector = input_ref or element or ref
        if not selector:
            raise RuntimeError("upload requires ref/input_ref/element selector")
        if not paths:
            raise RuntimeError("upload requires non-empty paths")

        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"Upload file not found: {missing[0]}")

        if input_ref:
            locator = self._ref_locator(page, tid, input_ref)
        elif ref:
            locator = self._ref_locator(page, tid, ref)
        else:
            locator = page.locator(selector).first
            if callable(locator):
                locator = locator()
        await locator.set_input_files(paths, timeout=_clamp_timeout(timeout_ms))
        # Best effort for apps that rely on change/input events.
        try:
            await locator.evaluate(
                "el => { el.dispatchEvent(new Event('input', {bubbles:true})); "
                "el.dispatchEvent(new Event('change', {bubbles:true})); }"
            )
        except Exception:
            pass
        return {"ok": True, "targetId": tid, "count": len(paths), "paths": paths}

    # ------------------------------------------------------------------
    # Dialog hook (next dialog)
    # ------------------------------------------------------------------

    async def dialog(
        self,
        accept: bool,
        prompt_text: str = "",
        target_id: str | None = None,
        timeout_ms: int = 10_000,
    ) -> dict:
        tid = self._resolve_target_id(target_id)
        ttl = _clamp_timeout(timeout_ms)
        self._dialog_hooks[tid] = {
            "accept": bool(accept),
            "prompt_text": prompt_text,
            "expires_at": asyncio.get_event_loop().time() + (ttl / 1000),
        }
        return {"ok": True, "targetId": tid, "armed": True, "timeoutMs": ttl}

    # ------------------------------------------------------------------
    # Act — execute UI actions (mirrors openclaw act kinds)
    # ------------------------------------------------------------------

    async def act(self, request: dict, target_id: str | None = None) -> dict:
        page = self._get_page(target_id)
        tid = self._resolve_target_id(target_id)
        kind = str(request.get("kind", ""))
        timeout = _clamp_timeout(request.get("timeoutMs", 10_000))

        if kind == "click":
            ref = _require(request, "ref")
            locator = self._ref_locator(page, tid, ref)
            if request.get("doubleClick"):
                await locator.dblclick(
                    timeout=timeout,
                    button=_button(request.get("button")),
                    modifiers=_modifiers(request.get("modifiers")),
                )
            else:
                await locator.click(
                    timeout=timeout,
                    button=_button(request.get("button")),
                    modifiers=_modifiers(request.get("modifiers")),
                )

        elif kind == "type":
            ref = _require(request, "ref")
            text = str(request.get("text", ""))
            locator = self._ref_locator(page, tid, ref)
            if request.get("slowly"):
                await locator.click(timeout=timeout)
                await locator.type(text, timeout=timeout, delay=75)
            else:
                await locator.fill(text, timeout=timeout)
            if request.get("submit"):
                await locator.press("Enter", timeout=timeout)

        elif kind == "press":
            key = _require(request, "key")
            delay = int(request.get("delayMs", 0))
            await page.keyboard.press(key, delay=max(0, delay))

        elif kind == "hover":
            ref = _require(request, "ref")
            await self._ref_locator(page, tid, ref).hover(timeout=timeout)

        elif kind == "scrollIntoView":
            ref = _require(request, "ref")
            await self._ref_locator(page, tid, ref).scroll_into_view_if_needed(timeout=timeout)

        elif kind == "drag":
            start = _require(request, "startRef")
            end = _require(request, "endRef")
            await self._ref_locator(page, tid, start).drag_to(
                self._ref_locator(page, tid, end),
                timeout=timeout,
            )

        elif kind == "fill":
            fields = request.get("fields") or []
            for field in fields:
                sel = str(field.get("ref") or "")
                val = str(field.get("value", ""))
                ftype = str(field.get("type", "")).lower()
                locator = self._ref_locator(page, tid, sel)
                if ftype in ("checkbox", "radio"):
                    checked = str(field.get("value", "")).lower() in ("1", "true", "yes", "on")
                    await locator.set_checked(checked, timeout=timeout)
                else:
                    await locator.fill(val, timeout=timeout)

        elif kind == "select":
            ref = _require(request, "ref")
            values = request.get("values") or []
            await self._ref_locator(page, tid, ref).select_option(values, timeout=timeout)

        elif kind == "evaluate":
            fn = _require(request, "fn")
            if request.get("ref"):
                ref = str(request["ref"])
                result = await asyncio.wait_for(
                    self._ref_locator(page, tid, ref).evaluate(fn),
                    timeout=timeout / 1000,
                )
            else:
                result = await asyncio.wait_for(page.evaluate(fn), timeout=timeout / 1000)
            return {"ok": True, "result": result}

        elif kind == "wait":
            await self._wait(page, request, timeout)

        elif kind == "resize":
            w = int(request.get("width", 1280))
            h = int(request.get("height", 800))
            await page.set_viewport_size({"width": w, "height": h})

        elif kind == "close":
            await self.close_tab(target_id)
            return {"ok": True, "targetId": tid}

        else:
            raise ValueError(
                "Unknown act kind: "
                f"{kind!r}. Valid: click/type/press/hover/scrollIntoView/"
                "drag/fill/select/evaluate/wait/resize/close"
            )

        return {"ok": True, "targetId": tid, "url": page.url}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register_page(self, page: Any) -> str:
        self._page_counter += 1
        tid = f"tab-{self._page_counter}"
        self._pages[tid] = page
        self._active_target_id = tid
        self._console_logs[tid] = []
        self._bind_page_listeners(tid, page)
        page.on("close", lambda: self._on_page_closed(tid))
        return tid

    def _bind_page_listeners(self, tid: str, page: Any) -> None:
        def _on_console(msg: Any) -> None:
            try:
                msg_type = msg.type() if callable(getattr(msg, "type", None)) else getattr(msg, "type", "log")
                msg_text = msg.text() if callable(getattr(msg, "text", None)) else getattr(msg, "text", "")
                msg_loc = msg.location() if callable(getattr(msg, "location", None)) else getattr(msg, "location", {})
                entry = {
                    "level": str(msg_type),
                    "text": str(msg_text),
                    "location": msg_loc or {},
                }
            except Exception:
                entry = {"level": "log", "text": str(msg)}
            bucket = self._console_logs.setdefault(tid, [])
            bucket.append(entry)
            if len(bucket) > 200:
                del bucket[: len(bucket) - 200]

        def _on_page_error(err: Any) -> None:
            bucket = self._console_logs.setdefault(tid, [])
            bucket.append({"level": "error", "text": str(err)})
            if len(bucket) > 200:
                del bucket[: len(bucket) - 200]

        def _on_request_failed(req: Any) -> None:
            failure = req.failure() if callable(getattr(req, "failure", None)) else getattr(req, "failure", {})
            method = req.method if isinstance(getattr(req, "method", None), str) else getattr(req, "method", lambda: "GET")()
            url = req.url if isinstance(getattr(req, "url", None), str) else getattr(req, "url", lambda: "")()
            error_text = ""
            if isinstance(failure, dict):
                error_text = str(failure.get("errorText", ""))
            elif failure is not None:
                error_text = str(failure)
            text = f"requestfailed {method} {url} {error_text}".strip()
            bucket = self._console_logs.setdefault(tid, [])
            bucket.append({"level": "warning", "text": text})
            if len(bucket) > 200:
                del bucket[: len(bucket) - 200]

        async def _on_dialog(dialog: Any) -> None:
            await self._handle_dialog(tid, dialog)

        page.on("console", _on_console)
        page.on("pageerror", _on_page_error)
        page.on("requestfailed", _on_request_failed)
        page.on("dialog", lambda d: asyncio.create_task(_on_dialog(d)))

    def _on_page_closed(self, tid: str) -> None:
        self._pages.pop(tid, None)
        self._console_logs.pop(tid, None)
        self._dialog_hooks.pop(tid, None)
        self._role_refs_by_target.pop(tid, None)
        self._role_refs_mode_by_target.pop(tid, None)
        if self._active_target_id == tid:
            self._active_target_id = list(self._pages.keys())[-1] if self._pages else None

    async def _handle_dialog(self, tid: str, dialog: Any) -> None:
        hook = self._dialog_hooks.get(tid)
        now = asyncio.get_event_loop().time()
        if hook and now <= float(hook.get("expires_at", 0)):
            try:
                if hook.get("accept", True):
                    prompt_text = str(hook.get("prompt_text") or "")
                    await dialog.accept(prompt_text or None)
                else:
                    await dialog.dismiss()
            finally:
                self._dialog_hooks.pop(tid, None)
            return

        # Default behavior: dismiss unexpected dialog so automation can continue.
        try:
            await dialog.dismiss()
        except Exception:
            pass

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("Browser not started. Use action='start' first.")

    def _get_page(self, target_id: str | None = None) -> Any:
        self._ensure_started()
        if not self._pages:
            raise RuntimeError("No open tabs.")
        if target_id:
            page = self._pages.get(target_id)
            if not page:
                raise RuntimeError(f"Tab not found: {target_id!r}. Use action='tabs' to list open tabs.")
            return page
        # Default: active tab, fallback to last.
        if self._active_target_id and self._active_target_id in self._pages:
            return self._pages[self._active_target_id]
        return list(self._pages.values())[-1]

    def _resolve_target_id(self, target_id: str | None) -> str:
        if target_id:
            return target_id
        if self._active_target_id:
            return self._active_target_id
        if self._pages:
            return list(self._pages.keys())[-1]
        raise RuntimeError("No open tabs.")

    async def _wait(self, page: Any, request: dict, timeout: int) -> None:
        if text_gone := request.get("textGone"):
            await page.wait_for_selector(f"text={text_gone}", state="hidden", timeout=timeout)
            return
        if text := request.get("text"):
            await page.wait_for_selector(f"text={text}", state="visible", timeout=timeout)
            return
        if selector := request.get("selector"):
            await page.wait_for_selector(str(selector), state="visible", timeout=timeout)
            return
        if url := request.get("url"):
            await page.wait_for_url(str(url), timeout=timeout)
            return
        if load_state := request.get("loadState"):
            await page.wait_for_load_state(str(load_state), timeout=timeout)
            return
        if fn := request.get("fn"):
            await page.wait_for_function(str(fn), timeout=timeout)
            return
        if time_ms := request.get("timeMs"):
            await asyncio.sleep(max(0, float(time_ms)) / 1000)
            return
        await page.wait_for_timeout(200)

    def _ref_locator(self, page: Any, target_id: str, ref: str) -> Any:
        normalized = _normalize_ref(ref)
        if normalized.startswith("e") and normalized[1:].isdigit():
            mode = self._role_refs_mode_by_target.get(target_id, "role")
            if mode == "aria":
                # Reserved for future parity with Playwright aria-ref mode.
                return page.locator(f"aria-ref={normalized}")
            refs = self._role_refs_by_target.get(target_id, {})
            info = refs.get(normalized)
            if not info:
                raise RuntimeError(
                    f'Unknown ref "{normalized}". Run action="snapshot" again and use latest refs.'
                )
            role = str(info.get("role") or "").strip()
            name = str(info.get("name") or "").strip()
            nth = info.get("nth")
            if not role:
                raise RuntimeError(f'Invalid ref mapping for "{normalized}"')
            locator = page.get_by_role(role, name=name, exact=True) if name else page.get_by_role(role)
            if isinstance(nth, int) and nth >= 0:
                locator = locator.nth(nth)
            return locator

        return page.locator(ref)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

async def _safe_title(page: Any) -> str:
    try:
        return await page.title()
    except Exception:
        return ""


def _require(request: dict, key: str) -> str:
    v = request.get(key)
    if not v:
        raise ValueError(f"act request missing required field: {key!r}")
    return str(v)


def _clamp_timeout(value: Any, default: int = 10_000) -> int:
    try:
        timeout = int(value)
    except Exception:
        timeout = default
    return max(500, min(120_000, timeout))


def _button(value: Any) -> str:
    raw = str(value or "left").lower()
    return raw if raw in ("left", "right", "middle") else "left"


def _modifiers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    valid = {
        "alt": "Alt",
        "control": "Control",
        "ctrl": "Control",
        "controlormeta": "ControlOrMeta",
        "meta": "Meta",
        "shift": "Shift",
    }
    for item in value:
        key = str(item).strip().lower()
        if key in valid:
            normalized.append(valid[key])
    return normalized


def _serialize_ax(node: Any, depth: int = 0, max_depth: int = 10) -> str:
    """Serialize Playwright accessibility snapshot to readable text."""
    if not node or depth > max_depth:
        return ""
    indent = "  " * depth
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = [role]
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f"= {value!r}")

    lines = [indent + " ".join(p for p in parts if p)]
    for child in node.get("children") or []:
        child_text = _serialize_ax(child, depth + 1, max_depth)
        if child_text:
            lines.append(child_text)
    return "\n".join(lines)


def _normalize_ref(value: str) -> str:
    ref = str(value or "").strip()
    if ref.startswith("@"):
        ref = ref[1:]
    if ref.startswith("ref="):
        ref = ref[4:]
    return ref


def _flatten_ax(node: Any, limit: int = 2000) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    out: list[dict[str, Any]] = []
    stack: list[tuple[dict[str, Any], int]] = [(node, 0)]
    while stack and len(out) < limit:
        cur, depth = stack.pop()
        role = str(cur.get("role", "") or "unknown")
        name = str(cur.get("name", "") or "")
        value = cur.get("value")
        item = {"ref": f"ax{len(out)+1}", "role": role, "name": name, "depth": depth}
        if value not in (None, ""):
            item["value"] = str(value)
        out.append(item)
        children = cur.get("children") or []
        if isinstance(children, list):
            for child in reversed(children):
                if isinstance(child, dict):
                    stack.append((child, depth + 1))
    return out


def _build_role_snapshot_from_ax(node: Any, options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    interactive_only = bool(options.get("interactive"))
    compact = bool(options.get("compact"))
    max_depth = options.get("max_depth")
    max_depth = int(max_depth) if isinstance(max_depth, int) and max_depth >= 0 else None

    refs: dict[str, dict[str, Any]] = {}
    counts: dict[tuple[str, str], int] = {}
    refs_by_key: dict[tuple[str, str], list[str]] = {}
    lines: list[str] = []
    counter = 0

    def walk(cur: Any, depth: int) -> None:
        nonlocal counter
        if not isinstance(cur, dict):
            return
        if max_depth is not None and depth > max_depth:
            return

        role_raw = str(cur.get("role", "") or "").strip()
        if not role_raw:
            role_raw = "generic"
        role = role_raw.lower()
        name = str(cur.get("name", "") or "").strip()
        value = cur.get("value")
        is_interactive = role in _INTERACTIVE_ROLES
        is_content = role in _CONTENT_ROLES
        is_structural = role in _STRUCTURAL_ROLES
        should_show = not (compact and is_structural and not name)
        should_ref = is_interactive or (is_content and bool(name))

        if interactive_only and not is_interactive:
            should_show = False

        if should_show:
            entry = f'{"  " * depth}- {role_raw}'
            if name:
                entry += f' "{name}"'
            if value not in (None, "") and not isinstance(value, dict):
                entry += f" = {value!r}"

            if should_ref:
                counter += 1
                ref = f"e{counter}"
                key = (role, name)
                nth = counts.get(key, 0)
                counts[key] = nth + 1
                refs[ref] = {"role": role}
                if name:
                    refs[ref]["name"] = name
                refs[ref]["nth"] = nth
                refs_by_key.setdefault(key, []).append(ref)
                entry += f" [ref={ref}]"
                if nth > 0:
                    entry += f" [nth={nth}]"

            lines.append(entry)

        for child in cur.get("children") or []:
            walk(child, depth + 1)

    walk(node, 0)

    # Keep nth only for duplicates, same as openclaw behavior.
    for key, refs_for_key in refs_by_key.items():
        if len(refs_for_key) <= 1:
            only = refs_for_key[0]
            refs[only].pop("nth", None)

    snapshot = "\n".join(lines)
    interactive_count = sum(1 for v in refs.values() if v.get("role") in _INTERACTIVE_ROLES)
    return {
        "snapshot": snapshot,
        "refs": refs,
        "stats": {
            "lines": len(lines),
            "chars": len(snapshot),
            "refs": len(refs),
            "interactive": interactive_count,
        },
    }


_INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "option",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "treeitem",
}

_CONTENT_ROLES = {
    "heading",
    "cell",
    "gridcell",
    "columnheader",
    "rowheader",
    "listitem",
    "article",
    "region",
    "main",
    "navigation",
}

_STRUCTURAL_ROLES = {
    "generic",
    "group",
    "list",
    "table",
    "row",
    "rowgroup",
    "grid",
    "treegrid",
    "menu",
    "menubar",
    "toolbar",
    "tablist",
    "tree",
    "directory",
    "document",
    "application",
    "presentation",
    "none",
}
