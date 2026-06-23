"""Unified S2S adapter for the OpenAI Realtime API (``openai`` extra).

Import-guarded scaffolding: constructing :class:`RealtimeS2S` without the
``openai`` extra raises a clear
:class:`~interpret_live.backends.guard.MissingExtraError`. On this path the
provider does STT+MT+TTS internally; our LocalAgreement stabilizer is bypassed
(documented in the capability matrix). The harness still sends the provider's
``response.cancel`` on barge-in via :meth:`interrupt`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..types import AudioFrame, TtsChunk
from .guard import require

__all__ = ["RealtimeS2S"]


class RealtimeS2S:
    """Speech-to-speech via the OpenAI Realtime API.

    Args:
        api_key: OpenAI API key.
        model: Realtime model id.
        voice: Output voice name.
        target_lang: Target language for the translation instructions.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-realtime-preview",
        voice: str = "alloy",
        target_lang: str = "es",
    ) -> None:
        openai = require("openai", backend="realtime", extra="openai")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._voice = voice
        self._target_lang = target_lang
        self._connection: object | None = None

    def stream(
        self, audio: AsyncIterator[AudioFrame], *, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:  # pragma: no cover - requires network/SDK
        """Stream translated target audio for the incoming ``audio``."""
        raise NotImplementedError(
            "RealtimeS2S streaming requires a live OpenAI Realtime connection; "
            "wire the websocket session here. The interface and import-guard are "
            "in place so the pipeline/Session can target it."
        )

    async def interrupt(self) -> None:  # pragma: no cover - requires network/SDK
        """Send ``response.cancel`` to halt in-flight synthesis."""
        raise NotImplementedError("interrupt requires a live Realtime connection")
