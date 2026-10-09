"""app/graph/nodes/generate.py — node 5 of 10.

Calls Gemini 2.5 Flash to draft an answer. The model is invoked with a
structured prompt: deterministic circuit state, the oracle result, the
diagnosis, the retrieved knowledge, and the current student context. The
prompt explicitly forbids the model from inventing waveform numbers — it
must quote the injected oracle facts and explain them.

This node is the only one that talks to Gemini. Everything else guards
against the LLM drifting from ground truth.

The client boundary is in ``app.llm.client`` so the node can be tested by
monkeypatching ``client.generate`` without a real API key.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

from app.llm.client import GeminiClient, GeminiNotConfigured

# The model is configurable through an environment variable at import time.
GEMINI_MODEL = __import__("os").environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _prompt(state: dict[str, Any]) -> str:
    """Build the prompt that the generator sees.

    Facts are injected verbatim into the prompt so the model is anchored to
    the deterministic engine. The LLM's role is explanation and
    pedagogy, not computation.
    """
    query = state.get("query", "")
    ctx = state.get("circuit", {})
    wiring = ctx.get("wiring_status") or {}
    oracle = ctx.get("oracle") or {}
    # The engine's oracle is a dict, but guard against any corruption
    # upstream so we never call methods on a non-dict.
    if not isinstance(oracle, dict):
        oracle = {}
    diagnosis = ctx.get("diagnosis") or {}
    faults = diagnosis.get("faults") or []
    if not isinstance(faults, list):
        faults = []
    knowledge = state.get("knowledge", {})
    if not isinstance(knowledge, dict):
        knowledge = {}
    reranked = knowledge.get("reranked") or []

    # --- Deterministic facts, quoted verbatim into the prompt ---
    facts = []
    if wiring.get("valid") is not True:
        if wiring.get("missing"):
            facts.append("MISSING: " + ", ".join(f"{m['from']} -> {m['to']}" for m in wiring["missing"]))
        if wiring.get("unexpected"):
            facts.append("UNEXPECTED: " + ", ".join(f"{u['from']} -> {u['to']}" for u in wiring["unexpected"]))
    if oracle:
        facts.append(f"scheme={oracle.get('scheme')}")
        if oracle.get("levels") is not None:
            facts.append(f"levels={oracle['levels']}")
        if oracle.get("transitions") is not None:
            facts.append(f"transitions={oracle['transitions']}")
        if oracle.get("dc_component") is not None:
            facts.append(f"dc_component={oracle['dc_component']:.4f}")

    # --- The knowledge the student is citing ---
    kb_notes = ""
    if reranked:
        items = []
        for c in reranked:
            tags = c.get("metadata", {}).get("tags", [])
            title = c.get("metadata", {}).get("title", "")
            heading = c.get("metadata", {}).get("heading", "")
            label = f"[{', '.join(tags)}] {title}" if title else f"[{heading}]" if heading else ""
            items.append(f"{label or '?'} {c.get('text', '')[:400]}")
        kb_notes = "The following knowledge refers to this student question:\n" + "\n".join(items)

    # --- Diagnosis wording ---
    diag_text = ""
    if faults:
        diag_text = "The deterministic diagnosis found the following faults:\n" + "\n".join(f"  - {f['name']} ({f['evidence']})" for f in faults[:4])

    prompt = f"""You are a friendly digital-communication tutor helping a student in the DigiWave X line-coding lab.

The student asked this question:
"""
    prompt += f"  {query}\n\n"

    if facts:
        prompt += "Ground-truth circuit facts (deterministic, from the engine):\n"
        for f in facts:
            prompt += f"  {f}\n"
        prompt += "\n"

    if diag_text:
        prompt += "Deterministic diagnosis:\n" + diag_text + "\n\n"

    if kb_notes:
        prompt += "Knowledge that was retrieved for this question:\n" + kb_notes + "\n\n"

    prompt += """Rules
-----
1. You are a tutor. Explain the concept, point out misconceptions, and give a hint if asked. Do NOT calculate circuit truth.
2. You may ONLY use the numbers in "Ground-truth circuit facts" for waveform statistics. Never invent a new value for levels, dc_component, transitions, or scheme.
3. If the circuit is not wired correctly, explain the wiring faults first, then explain the concept they reveal.
4. When quoting levels, use the list exactly as printed (e.g. [1.0, 1.0, 0.0, 0.0, -1.0, -1.0]).
5. If there is no relevant knowledge, say so and lean on the ground-truth facts.
6. Write clearly for a student who is new to line coding.

Answer the student's question directly and concisely, but thoroughly enough to be useful.
"""

    return prompt


def generate(state: dict[str, Any]) -> dict[str, Any]:
    """Node 5: call Gemini with the grounded prompt and return a draft answer."""
    # Debug: print the circuit section shape so we can see what the graph
    # actually carried in the channel.
    _circuit = state.get("circuit")
    oracle = (_circuit or {}).get("oracle") if isinstance(_circuit, dict) else None
    if not isinstance(oracle, dict):
        oracle = {}
    prompt = _prompt(state)

    try:
        client: GeminiClient = GeminiClient()
        # Raise a clear error if the key is missing rather than a confusing
        # failure deep inside the SDK.
        if not client.ready:
            raise GeminiNotConfigured(
                "GEMINI_API_KEY is not set. The generate step needs it."
            )
        user_message = str(state.get("query", "")).strip()
        if not user_message:
            user_message = "Please explain the current circuit state."

        response = client.generate(
            system_prompt=prompt,
            user_message=user_message,
        )
    except GeminiNotConfigured as exc:
        # Graceful degradation: return a plain-text message the API can show
        # to the student instead of a 500.
        log.warning("generate() failed because Gemini is not configured: %s", exc)
        # Graceful fallback: restate the deterministic engine facts (they are
        # the canonical source of truth) so the student still gets a useful
        # answer, then report the missing key as an error flag.
        _circuit = state.get("circuit") or {}
        _oracle = _circuit.get("oracle") if isinstance(_circuit, dict) else None
        _oracle = _oracle if isinstance(_oracle, dict) else {}
        _draft = (
            "I couldn't load the tutor model right now (the API key is "
            "missing in this environment). Please set GEMINI_API_KEY and "
            "try again. The circuit facts below are still deterministic."
        )
        return {
            "tutor": {
                "draft_answer": _draft,
                "fallback": True,
                "facts_used": [
                    "scheme=" + str(_oracle.get("scheme")),
                    "transitions=" + str(_oracle.get("transitions")),
                ],
                "hint_level": 0,
                "hint_text": None,
                "what_if": None,
                "error": "GEMINI_API_KEY is not set",
            }
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("Gemini generate failed")
        raise

    return {
        "tutor": {
            "draft_answer": response.strip(),
            "facts_used": [],
            "citations": [],
            "hint_level": 0,
            "hint_text": None,
            "what_if": None,
        }
    }
