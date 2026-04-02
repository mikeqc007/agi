from __future__ import annotations

import re


def expand_query(text: str) -> str:
    """Light query expansion for FTS5: add OR variants for common synonyms."""
    text = text.strip()
    if not text:
        return text

    # Escape FTS5 special chars
    safe = re.sub(r'["\*\(\)\:]+', " ", text).strip()

    # Split into terms
    terms = safe.split()
    if not terms:
        return safe

    # Build FTS5 query: each term as prefix match, joined with OR for recall
    parts = [f'"{t}"*' for t in terms if len(t) > 1]
    if not parts:
        return safe

    # Also include the full phrase as a boost
    phrase = f'"{safe}"'
    return f"{phrase} OR {' OR '.join(parts)}"


def expand_query_for_fts(text: str) -> str:
    """Safe wrapper — returns plain text on any error."""
    try:
        return expand_query(text)
    except Exception:
        return text
