"""Unified S2S adapter for the Gemini Live API (``gemini`` extra).

Import-guarded scaffolding: constructing :class:`GeminiS2S` without the
``gemini`` extra raises a clear
:class:`~interpret_live.backends.guard.MissingExtraError`. As with the Realtime
adapter, the provider does S2S internally; our stabilizer is bypassed on this
path. :meth:`interrupt` issues the provider's cancellation on barge-in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..types import AudioFrame, TtsChunk
from .guard import require

__all__ = ["GeminiS2S"]


class GeminiS2S:
    """Speech-to-speech via the Gemini Live API.

    Args:
        api_key: Google AI API key.
        model: Gemini Live model id.
        target_lang: Target language for the translation instructions.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.0-flash-live",
        target_lang: str = "es",
    ) -> None:
        genai = require("google.genai", backend="gemini", extra="gemini")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._target_lang = target_lang

    def stream(
        self, audio: AsyncIterator[AudioFrame], *, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:  # pragma: no cover - requires network/SDK
        """Stream translated target audio for the incoming ``audio``."""
        raise NotImplementedError(
            "GeminiS2S streaming requires a live Gemini Live session; wire the "
            "bidirectional session here. The interface and import-guard are in "
            "place so the pipeline/Session can target it."
        )

    async def interrupt(self) -> None:  # pragma: no cover - requires network/SDK
        """Cancel the in-flight Gemini Live turn."""
        raise NotImplementedError("interrupt requires a live Gemini Live session")
