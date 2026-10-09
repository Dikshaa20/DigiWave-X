"""app/graph/nodes/retrieve.py — node 3 of 10.

Reads the deterministic Engine + Oracle output and the student's query from
AgentState, then asks the knowledge base for the most relevant chunks.

The node does NOT call the LLM. It only:
1. normalises the query (the diagnosis often produces a topic seed the
   student never typed),
2. delegates to :class:`KnowledgeBase.retrieve` (ChromaDB + sentence
   embeddings, independent of Gemini), and
3. stores the ranked results so the generator and verifier can quote them.

The node is pure I/O over the knowledge layer and is trivially testable
with a stubbed knowledge base.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.ingestion.ingestion import KnowledgeBase

log = logging.getLogger(__name__)

# The number of chunks the generator may cite. Keep it small so answers stay
# grounded and fast.
TOP_K = 5

# Singleton RAG cache. SentenceTransformer (all-MiniLM-L6-v2) is loaded    extensively here; once per graph node invocation so a module-level cache
# keeps the first-turn latency reasonable and makes the evaluation batch
# runnable in a sane time. The knowledge base is immutable after bootstrap,
# so sharing across requests is safe.
_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def _get_knowledge_base() -> KnowledgeBase:
    """Return the module-level KnowledgeBase, bootstrapping it once."""
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = KnowledgeBase.from_manifest()
                try:
                    _kb.add()
                except Exception as exc:  # pragma: no cover
                    log.warning("knowledge base bootstrap failed: %s", exc)
    return _kb


def _topic_seed(query: str, intent: str | None) -> str:
    """Build a ChromaDB query from the student's message + intent.

    The router's intent is useful: a "diagnosis" question should be
    re-formulated as a topic query about the relevant code or fault, while
    a "concept" question can be searched more literally. This keeps the
    retrieval grounded in what the student is actually trying to understand.
    """
    seed = query.strip()
    if intent == "diagnosis":
        # Turn a diagnosis request into a topic query. E.g. "why is my ami
        # wrong" -> "AMI polarity diagnosis explanation"
        seed = f"explanation of {seed}"
    elif intent == "what_if":
        seed = f"predict {seed}"
    elif intent == "concept":
        seed = query.strip()
    return seed


def retrieve(state: dict[str, Any]) -> dict[str, Any]:
    """Node 3: populate the knowledge section from ChromaDB."""
    query = state.get("query", "")
    intent = state.get("intent")
    kb = _get_knowledge_base()
    results = kb.retrieve(_topic_seed(query, intent), n_results=TOP_K)

    # Normalise ChromaDB hits into the state's RetrievedChunk shape.
    retrieved = []
    for chunk in results:
        retrieved.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "distance": chunk.get("distance"),
                "metadata": chunk.get("metadata", {}),
            }
        )

    return {
        "knowledge": {
            "query_used": _topic_seed(query, intent),
            "retrieved": retrieved,
            "reranked": [],  # populated later by rerank.py
            "enough_evidence": bool(retrieved),
        }
    }
