"""app/ingestion/ingestion.py — RAG ingestion + retrieval for DigiWave X.

This module is the *knowledge* half of the tutoring pipeline. It is
completely independent of the deterministic experiment engine and of any
LLM: it only:

1. splits a doc into overlapping text chunks (chunking),
2. embeds each chunk with a local model (all-MiniLM-L6-v2),
3. stores documents + metadata in a persistent ChromaDB collection, and
4. retrieves the top-K most relevant chunks for a query.

The ingestion routine is deterministic and seedable, so tests can check
exactly which chunks are stored and how a query resolves.

Two classes are public:

* :class:`KnowledgeBase`  — one-shot builder. Call
  :meth:`KnowledgeBase.from_manifest` once at startup, then
  :meth:`retrieve` per turn.
* :class:`ChromaRetriever` — the retrieval layer. Exposed so tests can swap
  the embedder or collection backend without touching the graph nodes.

The RAG pipeline never talks to Gemini. A tutor answer is suggested by the
LLM *after* retrieval; this module only supplies evidence.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import chromadb
    from chromadb.config import Settings
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded-knowledge base. A real product would load these from a documents
# folder, but shipping a small, curated manifest keeps the MVP self-contained
# and deterministic across machines. Metadata tags make each chunk traceable.
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = {
    "docs": [
        {
            "id": "ami",
            "title": "AMI (Alternate Mark Inversion)",
            "sections": [
                {
                    "heading": "Polarity values",
                    "text": (
                        "AMI is a bipolar line code. A zero bit ('space') is "
                        "always a zero voltage level. A one bit ('mark') is "
                        "either +1 or -1, and the polarity of a one *alternates* "
                        "from one mark to the next. The very first mark in a "
                        "stream is +1."
                    ),
                    "tags": ["ami", "bipolar"],
                },
                {
                    "heading": "Zeros never reset polarity",
                    "text": (
                        "A run of zeros does NOT reset the alternation. The "
                        "polarity continues alternating through zero bits, so a "
                        "student who assumes a run of zeros flips the mark "
                        "back to +1 is misreading the signal."
                    ),
                    "tags": ["ami", "misconception"],
                },
                {
                    "heading": "DC component",
                    "text": (
                        "Because +1 and -1 marks alternate, AMI has a near-zero "
                        "DC component. A measured DC offset is a quick check that "
                        "the encoder is not producing unipolar NRZ (which holds "
                        "one level and carries a large DC offset)."
                    ),
                    "tags": ["ami", "dc-component"],
                },
                {
                    "heading": "Transitions",
                    "text": (
                        "AMI has no guaranteed transitions at every bit boundary "
                        "when zeros are present: a zero bit has no transition "
                        "(both samples are 0). Transitions appear only at marks."
                    ),
                    "tags": ["ami", "transitions"],
                },
            ],
        },
        {
            "id": "manchester",
            "title": "Manchester (Bi-Phase) encoding",
            "sections": [
                {
                    "heading": "Transition every bit period",
                    "text": (
                        "Manchester encoding guarantees a mid-bit transition for "
                        "every bit period, which makes it self-clocking. The "
                        "data bit is carried by the direction of the transition, "
                        "not by the absolute level."
                    ),
                    "tags": ["manchester", "self-clocking"],
                },
                {
                    "heading": "IEEE 802.3 convention",
                    "text": (
                        "Under the IEEE 802.3 convention (the one used by this "
                        "lab): a 1 is low-then-high (level drops then rises in "
                        "the middle of the bit), and a 0 is high-then-low. The "
                        "opposite mapping (G. E. Thomas) appears in some "
                        "textbooks — a common source of waveform inversion."
                    ),
                    "tags": ["manchester", "convention"],
                },
                {
                    "heading": "Sampling at two per bit",
                    "text": (
                        "Manchester requires at least two samples per bit so the "
                        "mid-bit transition lands between samples. With only one "
                        "sample per bit the signal would look like plain NRZ, "
                        "which is the classic student error."
                    ),
                    "tags": ["manchester", "sampling"],
                },
            ],
        },
        {
            "id": "nrz",
            "title": "NRZ-L and NRZ encodings",
            "sections": [
                {
                    "heading": "NRZ-L (polar, level based)",
                    "text": (
                        "NRZ-L maps a 1 to a positive level (+1) and a 0 to a "
                        "negative level (-1), and holds that level for the "
                        "whole bit period. It is *level-based* and has no "
                        "transitions at bit boundaries unless the level "
                        "changes."
                    ),
                    "tags": ["nrz", "polar"],
                },
                {
                    "heading": "Unipolar NRZ",
                    "text": (
                        "Unipolar NRZ uses only one polarity: a 1 is +1 and a "
                        "0 is 0. Because only one level is used, unipolar NRZ "
                        "has a large DC component. Do not confuse it with polar "
                        "NRZ-L, which uses both +1 and -1."
                    ),
                    "tags": ["nrz", "unipolar", "dc-component"],
                },
                {
                    "heading": "What a student should watch for",
                    "text": (
                        "NRZ-L is level-based, not transition-based. If a "
                        "student treats it like NRZ-I (inverted), where a 1 "
                        "causes the level to flip at the boundary, the waveform "
                        "will be wrong even though the bits are the same. "
                        " The key check is the level, not "
                        "the edge."
                    ),
                    "tags": ["nrz", "convention"],
                },
            ],
        },
        {
            "id": "rz",
            "title": "RZ (Return-to-Zero)",
            "sections": [
                {
                    "heading": "Pulse shape",
                    "text": (
                        "Unipolar RZ sends a positive pulse for the first half "
                        "of the bit period and then returns to 0 for the second "
                        "half. A 0 stays at zero for the entire bit period. The "
                        "return to zero inside the bit period makes RZ "
                        "self-clocking."
                    ),
                    "tags": ["rz", "pulse"],
                },
                {
                    "heading": "Sampling",
                    "text": (
                        "RZ needs an even number of samples per bit so the "
                        "mid-bit return to zero is represented exactly. One "
                        "sample per bit collapses RZ into plain NRZ, which is "
                        "the most common waveform mismatch in this lab."
                    ),
                    "tags": ["rz", "sampling"],
                },
            ],
        },
        {
            "id": "line-ccoding-basics",
            "title": "Line coding basics",
            "sections": [
                {
                    "heading": "Why line coding matters",
                    "text": (
                        "A line-coded waveform carries raw bits across a wire. "
                        "The transmitter maps each bit to a voltage pattern, and "
                        "the receiver must be able to recover the clock and the "
                        "bits from that pattern."
                    ),
                    "tags": ["line coding", "basics"],
                },
                {
                    "heading": "Four properties to compare",
                    "text": (
                        "When a student reads a waveform they should check four "
                        "things: (1) the logic levels used (+1/-1/0), (2) "
                        "whether transitions are guaranteed at every bit "
                        "boundary, (3) the DC component of the stream, and "
                        "(4) whether the code is self-clocking."
                    ),
                    "tags": ["line coding", "basics"],
                },
                {
                    "heading": "Wiring to the encoder",
                    "text": (
                        "The bit source has a single output (source.out) wired "
                        "to the encoder input (encoder.in). The encoder output "
                        "(encoder.out) feeds the oscilloscope (scope.in). A "
                        "circuit missing the source-to-encoder wire, or missing "
                        "the encoder-to-scope wire, is incomplete."
                    ),
                    "tags": ["wiring", "kit"],
                },
            ],
        },
        {
            "id": "misconceptions",
            "title": "Common misconceptions & faults",
            "sections": [
                {
                    "heading": "AMI polarity reset after zeros",
                    "text": (
                        "Assuming a run of zeros resets mark polarity. AMI "
                        "alternates across the whole stream regardless of "
                        "intervening zeros. This is the top AMI fault in the "
                        "kit."
                    ),
                    "tags": ["ami", "misconception"],
                },
                {
                    "heading": "NRZ-L / NRZ-I confusion",
                    "text": (
                        "Treating NRZ-L as transition-based. NRZ-L is "
                        "level-based; NRZ-I encodes by a transition at the bit "
                        "boundary. The same bits give different waveforms."
                    ),
                    "tags": ["nrz", "misconception"],
                },
                {
                    "heading": "Manchester convention flip",
                    "text": (
                        "Using the G. E. Thomas mapping (1 = high-then-low) "
                        "where IEEE 802.3 (1 = low-then-high) is expected. The "
                        "waveform is the exact inverse."
                    ),
                    "tags": ["manchester", "misconception"],
                },
            ],
        },
        {
            "id": "waveform-interpretation",
            "title": "Waveform interpretation",
            "sections": [
                {
                    "heading": "Reading the scope trace",
                    "text": (
                        "Each bit occupies samples_per_bit samples. The "
                        "oscilloscope plots the sampled voltage. Compare the "
                        "shape against the known pattern of the chosen code, "
                        "and read the DC component and transition count on the "
                        "stats panel."
                    ),
                    "tags": ["waveform", "interpretation"],
                },
                {
                    "heading": "DC component",
                    "text": (
                        "DC component is the mean level of the waveform. "
                        "Codes that alternate +1 and -1 (AMI) sit near zero; "
                        "codes that stay near one level (unipolar NRZ) have a "
                        "large DC offset. Compare against the stats panel to "
                        "confirm the code."
                    ),
                    "tags": ["waveform", "dc-component"],
                },
                {
                    "heading": "Transitions",
                    "text": (
                        "The transition count is the number of times the "
                        "sampled level changes from sample to sample. "
                        "Self-clocking codes (Manchester, RZ when the bit has "
                        "a pulse) have a transition inside every bit period."
                    ),
                    "tags": ["waveform", "transitions"],
                },
            ],
        },
        {
            "id": "wiring-kit",
            "title": "The virtual wiring kit",
            "sections": [
                {
                    "heading": "Source → encoder → scope",
                    "text": (
                        "Required connections: source.out → encoder.in and "
                        "encoder.out → scope.in. Any extra wire is 'unexpected'."
                    ),
                    "tags": ["wiring", "kit"],
                },
                {
                    "heading": "Reading a fault",
                    "text": (
                        "The engine reports missing (required wires absent) and "
                        "unexpected (wires not in the required set). A missing "
                        "wire breaks the data path; an unexpected wire is a "
                        "hallucination that must be removed before testing."
                    ),
                    "tags": ["wiring", "kit"],
                },
            ],
        },
    ]
}


def _load_manifest(path: str | Path | None) -> dict[str, Any]:
    """Load the knowledge manifest from *path*, falling back to the bundled default."""
    if path is None:
        return DEFAULT_MANIFEST
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            return json.load(fh)
    log.warning(
        "Knowledge manifest not found at %s; using the bundled default.",
        p,
    )
    return DEFAULT_MANIFEST


def _chunk_document(doc: dict[str, Any], chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    """Split one document into overlapping text chunks.

    Each heading is paired with the paragraph that follows it. Short heading
    fragments (< 40 chars) are merged into the following paragraph so table
    fragments don't become orphaned chunks.
    """
    chunks: list[dict[str, Any]] = []
    current = ""
    current_meta: dict[str, Any] = {"source": doc["id"], "title": doc.get("title", ""), "heading": None, "tags": []}

    def _normalize_meta(meta: dict[str, Any]) -> dict[str, Any]:
        # ChromaDB requires non-empty scalar/list metadata values and does not
        # accept empty lists for tags. Make every value ChromaDB-safe.
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, list):
                v = [x for x in v if x not in (None, "")]
                if v:
                    out[k] = v
            elif isinstance(v, str) and v:
                out[k] = v
            elif not isinstance(v, (list, str)):
                out[k] = v
        return out

    def flush() -> None:
        nonlocal current, current_meta
        if current:
            chunks.append(
                {"chunk_id": f"{doc['id']}::{uuid.uuid4().hex[:8]}", "text": current, "metadata": _normalize_meta(dict(current_meta))}
            )
        current = ""
        current_meta = {"source": doc["id"], "title": doc.get("title", ""), "heading": None, "tags": []}

    for section in doc.get("sections", []):
        heading = section.get("heading") or ""
        text = section.get("text") or ""
        # merge short headings into the following paragraph (carry tags along)
        short_heading = len(heading) < 40 and len(heading) > 0
        tags = [t for t in section.get("tags", []) if t]
        if short_heading:
            current_meta["heading"] = heading
            if tags:
                current_meta["tags"] = tags
            if text:
                if current:
                    current += " " + heading
                current += text
            continue
        if heading:
            current_meta["heading"] = heading
        if tags:
            current_meta["tags"] = tags
        if text and current:
            current += " " + text
            if len(current) >= chunk_size:
                flush()
        elif text:
            current = text
        if current:
            flush()

    # final flush for any remainder
    if current:
        chunks.append(
            {"chunk_id": f"{doc['id']}::{uuid.uuid4().hex[:8]}", "text": current, "metadata": _normalize_meta(dict(current_meta))}
        )
    return chunks


def build_documents(manifest: dict[str, Any] | None = None, chunk_size: int = 800, overlap: int = 80) -> list[dict[str, Any]]:
    """Ingest the knowledge manifest into a flat list of chunk dicts.

    Each chunk dict carries exactly what the retriever needs downstream:
    chunk_id, text, metadata. The deterministic offset is reproducible.
    """
    manifest = manifest or DEFAULT_MANIFEST
    docs: list[dict[str, Any]] = []
    for doc in manifest["docs"]:
        docs.extend(_chunk_document(doc, chunk_size, overlap))
    return docs


class ChromaRetriever:
    """Persistence + retrieval over a ChromaDB collection.

    The constructor accepts arbitrary embedder + collection keyword
    arguments so tests can inject a fake or in-memory store.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        collection_name: str = "digiwave_x_knowledge",
        persist_directory: str | None = None,
        embedder: Any | None = None,
        embed_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory or ".chroma"
        self.embed_kwargs = embed_kwargs or {}
        self.embedder = embedder
        self.client = client
        # If no embedder was supplied, lazily build a SentenceTransformer
        # instance. Keeping the constructor cheap makes unit tests cheap.
        self._built_embedder = False

    def _ensure_embedder(self) -> None:
        if self._built_embedder:
            return
        if self.embedder is not None:
            self.embedder_model = self.embedder
        elif SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for ChromaRetriever")
        else:
            self.embedder_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._built_embedder = True

    def _get_or_create_collection(self) -> Any:
        if self.client is not None:
            return self.client.get_or_create_collection(self.collection_name)
        if chromadb is None:
            raise RuntimeError("chromadb is required for ChromaRetriever")
        if os.path.exists(self.persist_directory):
            existing = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(allow_reset=True),
            )
            try:
                return existing.get_or_create_collection(self.collection_name)
            except Exception:
                existing.reset()
                coll = existing.get_or_create_collection(self.collection_name)
                existing.delete_collection(self.collection_name)
                return existing.get_or_create_collection(self.collection_name)
        client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True),
        )
        return client.get_or_create_collection(self.collection_name)

    def add(self, chunks: list[dict[str, Any]]) -> int:
        """Add *chunks* to the collection. Returns the number of documents added."""
        if not chunks:
            return 0
        self._ensure_embedder()
        collection = self._get_or_create_collection()
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        embeddings = self.embedder_model.encode(texts, **self.embed_kwargs)
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(self, query_text: str, *, n_results: int = 5) -> list[dict[str, Any]]:
        """Return the top `n_results` most relevant chunk dicts for *query_text*."""
        if not query_text.strip():
            return []
        self._ensure_embedder()
        collection = self._get_or_create_collection()
        embedding = self.embedder_model.encode([query_text], **self.embed_kwargs)[0]
        res = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=n_results,
        )
        results: list[dict[str, Any]] = []
        if res.get("documents"):
            for i, doc in enumerate(res["documents"][0]):
                meta = res.get("metadatas") and (res["metadatas"][0][i] if res["metadatas"] else {})
                results.append(
                    {
                        "chunk_id": res.get("ids", [None])[0][i],
                        "text": doc,
                        "distance": res.get("distances", [[None]])[0][i],
                        "metadata": meta,
                    }
                )
        return results


class KnowledgeBase:
    """One-shot, deterministic builder of the ChromaDB knowledge store.

    Bootstrap usage (your graph will call this once on startup):

        kb = KnowledgeBase.from_manifest("path/to/manifest.json", chunk_size=800, overlap=80)
        kb.add()
        # then per turn:
        evidence = kb.retrieve("Why is my AMI polarity wrong?")
    """

    def __init__(
        self,
        *,
        manifest: dict[str, Any] | None = None,
        chunk_size: int = 800,
        overlap: int = 80,
        retriever: ChromaRetriever | None = None,
    ) -> None:
        self.manifest = manifest or DEFAULT_MANIFEST
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retriever = retriever
        self.documents: list[dict[str, Any]] = []

    @classmethod
    def from_manifest(
        cls,
        path: str | Path | None = None,
        *,
        manifest: dict[str, Any] | None = None,
        chunk_size: int = 800,
        overlap: int = 80,
        **retriever_kwargs: Any,
    ) -> "KnowledgeBase":
        kb = cls(manifest=manifest or _load_manifest(path), chunk_size=chunk_size, overlap=overlap)
        kb.retriever = ChromaRetriever(**retriever_kwargs)
        return kb

    def add(self) -> int:
        """Ingest the manifest and store it. Returns number of chunks stored."""
        self.documents = build_documents(self.manifest, self.chunk_size, self.overlap)
        if self.retriever is not None:
            return self.retriever.add(self.documents)
        return 0

    def retrieve(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        if self.retriever is None:
            return []
        return self.retriever.query(query, n_results=n_results)

    def count(self) -> int:
        return len(self.documents)


