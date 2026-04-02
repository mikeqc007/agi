from __future__ import annotations

from agi.memory.embeddings import cosine_similarity
from agi.types import MemoryEntry


def mmr_rerank(
    entries: list[MemoryEntry],
    k: int,
    lambda_: float = 0.7,
) -> list[MemoryEntry]:
    """Maximal Marginal Relevance using embedding cosine similarity.

    lambda_=1.0  → pure relevance (no diversity)
    lambda_=0.0  → pure diversity
    """
    if len(entries) <= k:
        return entries

    # Entries without embeddings fall back to relevance-only order
    has_emb = [e for e in entries if e.embedding]
    no_emb = [e for e in entries if not e.embedding]

    if not has_emb:
        return entries[:k]

    selected: list[MemoryEntry] = []
    remaining = list(has_emb)

    while len(selected) < k and remaining:
        if not selected:
            # First: pick highest relevance score
            best = max(remaining, key=lambda e: e.score)
            selected.append(best)
            remaining.remove(best)
            continue

        best_entry = None
        best_mmr = float("-inf")

        for cand in remaining:
            rel = cand.score
            # Maximum similarity to any already-selected entry
            max_sim = max(
                cosine_similarity(cand.embedding, sel.embedding)
                for sel in selected
                if sel.embedding
            )
            mmr_score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_entry = cand

        if best_entry is None:
            break
        selected.append(best_entry)
        remaining.remove(best_entry)

    # Append no-embedding entries if we still need more
    needed = k - len(selected)
    if needed > 0:
        selected.extend(no_emb[:needed])

    return selected
