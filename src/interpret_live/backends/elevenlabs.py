"""TTS adapter for ElevenLabs with optional voice preservation (``elevenlabs``).

Import-guarded scaffolding: constructing :class:`ElevenLabsTTS` without the
``elevenlabs`` extra raises a clear
:class:`~interpret_live.backends.guard.MissingExtraError`. A drop-in
:class:`~interpret_live.backends.TTS` that can clone the speaker's voice into the
target language; off by default. The streaming interface is in place so it
slots into the pipeline like any other TTS.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..types import TtsChunk
from .guard import require

__all__ = ["ElevenLabsTTS"]


class ElevenLabsTTS:
    """Streaming TTS via ElevenLabs, optionally using a cloned voice.

    Args:
        api_key: ElevenLabs API key.
        voice_id: Target voice id (a cloned voice for preservation, or a preset).
        model_id: ElevenLabs model id.
        sample_rate: Output sample rate.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        sample_rate: int = 22050,
    ) -> None:
        el = require("elevenlabs", backend="elevenlabs", extra="elevenlabs")
        self._client = el.client.ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._model_id = model_id
        self._sample_rate = sample_rate

    def synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:  # pragma: no cover - requires network/SDK
        """Return the streamed chunk iterator for ``text``."""
        raise NotImplementedError(
            "ElevenLabsTTS streaming requires a live ElevenLabs connection; wire "
            "the streaming synthesis here. The interface and import-guard are in "
            "place so it drops into the pipeline as a TTS."
        )
