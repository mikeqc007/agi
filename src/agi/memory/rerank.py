from __future__ import annotations

import logging

from agi.types import MemoryEntry

logger = logging.getLogger(__name__)


async def llm_rerank(
    entries: list[MemoryEntry],
    query: str,
    model: str,
    top_k: int,
) -> list[MemoryEntry]:
    """LLM pointwise reranking: ask LLM to score relevance 0-10 for each entry."""
    if not entries:
        return entries

    # Build prompt
    lines = ["Rate the relevance of each passage to the query. Reply with JSON array of scores (0-10)."]
    lines.append(f'\nQuery: "{query}"\n')
    for i, e in enumerate(entries):
        snippet = e.content[:300].replace("\n", " ")
        lines.append(f"[{i}] {snippet}")
    lines.append('\nReply ONLY with a JSON array like: [8, 3, 7, ...]')

    prompt = "\n".join(lines)

    try:
        import json
        import litellm

        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            stream=False,
        )
        raw = response.choices[0].message.content or "[]"
        # Extract JSON array from response
        import re
        m = re.search(r'\[[\d\s,\.]+\]', raw)
        if m:
            scores = json.loads(m.group())
            if len(scores) == len(entries):
                for entry, score in zip(entries, scores):
                    entry.score = float(score) / 10.0
                entries.sort(key=lambda e: e.score, reverse=True)
                return entries[:top_k]
    except Exception as e:
        logger.debug("LLM rerank failed: %s — falling back to score order", e)

    return entries[:top_k]
