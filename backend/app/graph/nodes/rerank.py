"""app/graph/nodes/rerank.py — node 4 of 10.

Implements a deterministic ranking layer over the chunks already retrieved by
:func:`retrieve`.

The rule set is deliberately small and explainable so it can be tested
without an LLM:
1. Exact phrase overlap: chunks whose ``text`` contains an important token
   from the student's query move to the top.
2. Metadata boost: chunks tagged with the current intent (e.g.
   ``"misconception"`` when the intent is diagnosis) gain a score.
3. Length sanity: chunks that are empty or absurdly long are penalised.

The reranker produces two ordered lists:
    reranked   - the chunks in final presentation order, with
                 ``rerank_score`` filled in.
    reordered  - a parallel list used for eval comparisons (kept explicit so
                 tests can assert that reranking actually changed the order).

Nothing here invents circuit facts. It only decides which knowledge to show
the student and which to quote in the generated answer.
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.nodes.retrieve import TOP_K

log = logging.getLogger(__name__)


def _token_set(text: str) -> set[str]:
    return set(text.lower().split())


def rerank(state: dict[str, Any]) -> dict[str, Any]:
    """Node 4: score and order the retrieved chunks."""
    query = state.get("query", "").lower()
    intent = state.get("intent")

    retrieved = state.get("knowledge", {}).get("retrieved", [])
    if not retrieved:
        return {
            "knowledge": {
                "query_used": query,
                "retrieved": [],
                "reranked": [],
                "enough_evidence": False,
            }
        }

    scored: list[dict[str, Any]] = []
    for chunk in retrieved:
        text = chunk.get("text", "")
        if not text:
            score = 0.0
        else:
            tokens = _token_set(text)
            query_tokens = set(query.split())
            overlap = len(tokens & query_tokens) / max(len(query_tokens), 1)
            intent_boost = 0.2 if intent and intent in chunk.get("metadata", {}).get("tags", []) else 0.0
            score = overlap + intent_boost

        # Penalise absurdly long snippets that would bloat the answer; a
        # normal chunk is ~200-900 chars, anything beyond 2500 chars is
        # probably a concatenation of multiple documents.
        if len(text) > 2500:
            score -= 0.5

        scored.append(
            {
                **chunk,
                "rerank_score": round(score, 4),
            }
        )

    # Deterministic ordering: higher score first, ties broken by chunk_id so
    # results are reproducible across runs.
    scored.sort(key=lambda c: (-c["rerank_score"], c.get("chunk_id", "")))

    top = scored[:TOP_K]
    reranked = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "text": c.get("text", ""),
            "distance": c.get("distance"),
            "rerank_score": c.get("rerank_score"),
            "metadata": c.get("metadata", {}),
        }
        for c in top
    ]

    return {
        "knowledge": {
            "query_used": query,
            "retrieved": [c for c in retrieved],
            "reranked": reranked,
            "enough_evidence": bool(reranked),
        }
    }
