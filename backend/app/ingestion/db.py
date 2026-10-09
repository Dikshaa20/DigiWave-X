"""app/ingestion/db.py — ChromaDB collection manager.

Keeps the persistence layer separate from the ingest/retrieve logic so the
graph nodes never import ``chromadb`` directly. The collection is created
once and reused across turns.

Public surface:
    create_collection(client, collection_name, persist_directory, *, overwrite=False)
    reset_collection(collection)

All functions raise ``ImportError`` if ``chromadb`` is not installed, so a
test runner can stub the module without a hard dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

try:  # pragma: no cover
    import chromadb
    from chromadb.config import Settings
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]


def create_collection(
    *,
    client: Any | None = None,
    collection_name: str = "digiwave_x_knowledge",
    persist_directory: str | None = None,
    overwrite: bool = False,
) -> Any:
    """Create (or reuse) a collection and return the ChromaDB collection object."""
    if chromadb is None:
        raise ImportError("chromadb is required; run: pip install chromadb")

    persist = persist_directory or ".chroma"
    if os.path.exists(persist) and not os.path.isdir(persist):
        os.makedirs(persist, exist_ok=True)
    elif not os.path.exists(persist):
        os.makedirs(persist, exist_ok=True)

    base: Any
    if client is not None:
        base = client
    else:
        base = chromadb.PersistentClient(
            path=persist,
            settings=Settings(allow_reset=True),
        )

    try:
        if overwrite and base.collection_exists(collection_name):
            base.delete_collection(collection_name)
        return base.get_or_create_collection(collection_name)
    except Exception:
        if base.collection_exists(collection_name):
            base.delete_collection(collection_name)
        return base.get_or_create_collection(collection_name)


def reset_collection(collection: Any) -> None:
    """Drop all rows from an existing collection without deleting it."""
    try:
        collection.delete(where={})  # clear every document
    except Exception:
        # Some versions of the client don't support `where={}` deletes; fall
        # back to a full reset by recreating the collection.
        try:
            collection.delete(ids=collection.get(include=[])["ids"])
        except Exception:
            log.warning("Could not reset ChromaDB collection; it may need manual recreation.")


def collection_exists(client: Any, collection_name: str) -> bool:
    """Return True when a collection with *collection_name* already exists."""
    try:
        return client.collection_exists(collection_name)
    except Exception:
        return False
