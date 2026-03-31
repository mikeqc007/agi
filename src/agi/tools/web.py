from __future__ import annotations

import logging
import re

import aiohttp

from agi.tools.registry import tool
from agi.types import ToolContext

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 15
MAX_CONTENT = 10_000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; agi/1.0)",
    "Accept-Encoding": "gzip, deflate",   # exclude brotli (not supported by aiohttp without extra deps)
}


@tool
async def web_fetch(ctx: ToolContext, url: str) -> str:
    """Fetch a web page and return its text content.

    url: URL to fetch
    """
    try:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
                allow_redirects=True,
            ) as resp:
                if resp.status >= 400:
                    return f"Error: HTTP {resp.status}"
                ct = resp.content_type or ""
                raw = await resp.text(errors="replace")

        if "html" in ct or raw.strip().startswith("<"):
            text = _html_to_text(raw)
        else:
            text = raw

        if len(text) > MAX_CONTENT:
            text = text[:MAX_CONTENT] + "\n[...truncated]"
        return text.strip() or "(empty)"
    except Exception as e:
        return f"Error: {e}"


@tool
async def web_search(ctx: ToolContext, query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return results.

    query: Search query
    num_results: Number of results to return (default 5)
    """
    try:
        results = await _ddg_search(query, max(1, min(10, int(num_results))))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


async def _ddg_search(query: str, limit: int) -> list[dict]:
    """DuckDuckGo HTML search (no API key needed)."""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "b": ""}

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.post(
            url,
            data=params,
            timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
        ) as resp:
            html = await resp.text(errors="replace")

    results = []
    # Parse result blocks
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )
    for href, title_html, snippet_html in blocks[:limit]:
        title = _strip_tags(title_html).strip()
        snippet = _strip_tags(snippet_html).strip()
        # DDG redirects; extract real URL
        real_url = _extract_ddg_url(href)
        if title and real_url:
            results.append({"title": title, "url": real_url, "snippet": snippet})

    return results


def _extract_ddg_url(href: str) -> str:
    m = re.search(r'uddg=([^&]+)', href)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1))
    return href


def _strip_tags(html: str) -> str:
    return re.sub(r'<[^>]+>', '', html)


def _html_to_text(html: str) -> str:
    """Very simple HTML → text: strip tags, decode entities."""
    # Remove scripts and styles
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Block elements → newlines
    html = re.sub(r'<(br|p|div|li|h[1-6]|tr)[^>]*>', '\n', html, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r'<[^>]+>', '', html)
    # Decode common entities
    for ent, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                       ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(ent, char)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
