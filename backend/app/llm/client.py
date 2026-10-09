"""app/llm/client.py — thin, testable Gemini abstraction.

Design
------
* The real model is fetched from the GEMINI_API_KEY environment variable.
  Nothing is hardcoded.
* Production graph nodes call :func:`gemini_generate` (returns a string),
  so every node can be unit-tested with a monkeypatched stub — no SDK
  dependency is required.
* If GEMINI_API_KEY is missing or empty the module exposes a
  ``GeminiNotConfigured`` exception. Graph entry points check this once
  and raise a clear HTTP 503 instead of failing deep inside a node.
"""

from __future__ import annotations

import os
import threading
from typing import Any

try:  # pragma: no cover - SDK import time
    from google import genai
except ImportError as exc:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GeminiNotConfigured(Exception):
    """Raised when GEMINI_API_KEY is absent or empty."""


class GeminiClient:
    """Thin façade over the Google GenAI REST API.

    Public surface is deliberately small so the LangGraph nodes can inject
    a stub for tests:

        client = GeminiClient()
        result = client.generate(chat, system_prompt, user_msg)

    The client is *not* a singleton that eagerly initialises the SDK on
    import. Initialisation happens lazily — the first call to
    :meth:`generate` — which makes import side-effects predictable.
    """

    MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model = model or self.MODEL
        self._client: genai.Client | None = None
        self._lock = threading.Lock()
        # Set only when the SDK import itself failed (as opposed to a missing
        # key). ``None`` otherwise, so ``ready`` can inspect it safely.
        self._import_error: BaseException | None = None
        try:
            import google.genai  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            self._import_error = exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        timeout: float = 60.0,
        temperature: float = 0.4,
    ) -> str:
        """Return a *string* completion from the configured model."""
        if not self.ready:
            raise GeminiNotConfigured(
                "GEMINI_API_KEY is not set. Set it before calling generate()."
            )
        # Lazy SDK initialisation.
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = genai.Client(api_key=self._api_key)
        response = self._client.models.generate_content(
            model=self._model,
            config=genai.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            ),
            contents=[
                genai.types.Content(
                    role="user",
                    parts=[genai.types.Part(text=system_prompt)],
                ),
                genai.types.Content(role="user", parts=[genai.types.Part(text=user_message)]),
            ],
        )
        # The SDK returns a parts list; grab the text verbatim.
        text = ""
        if response and getattr(response, "candidates", None):
            for cand in response.candidates:
                if getattr(cand, "content", None) is not None:
                    for part in cand.content.parts:
                        if getattr(part, "text", None):
                            text += part.text
        return text

    @property
    def ready(self) -> bool:
        """True when the SDK has a usable API key, False otherwise."""
        if self._import_error is not None:
            return False
        return bool(self._api_key and self._api_key.strip())

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    def from_env(cls) -> "GeminiClient":
        return cls()


# Singleton used by the graph wiring functions. The graph entry point should
# call :func:`get_graph_client` once and cache the result for the process.
_client: GeminiClient | None = None


def get_graph_client() -> GeminiClient:
    """Return a cached GeminiClient for graph use.

    Raises GeminiNotConfigured immediately if the key is missing, so the
    top-level API route can return HTTP 503 before running the graph.
    """
    global _client
    if _client is None:
        _client = GeminiClient()
        if not _client.ready:
            raise GeminiNotConfigured(
                "GEMINI_API_KEY is not set. The tutor route requires it."
            )
    return _client


def is_gemini_configured() -> bool:
    """Read-only check for callers that want to fail gracefully before work."""
    key = os.getenv("GEMINI_API_KEY")
    return bool(key and key.strip())
