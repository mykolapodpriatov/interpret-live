"""Offline streaming TTS adapter backed by Piper (``piper`` extra).

Import-guarded: constructing :class:`PiperTTS` without the ``piper`` extra raises
a clear :class:`~interpret_live.backends.guard.MissingExtraError`. Synthesizes
target text into :class:`~interpret_live.types.TtsChunk`s, streamed so the
pipeline can begin playback (and abort on barge-in) before the whole utterance is
rendered.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from ..types import TtsChunk
from .guard import require

__all__ = ["PiperTTS"]


class PiperTTS:
    """Streaming TTS over a Piper voice model.

    Args:
        model_path: Path to a Piper ``.onnx`` voice model.
        config_path: Optional path to the voice's ``.json`` config.
        sample_rate: Output sample rate (must match the voice model).
    """

    def __init__(
        self,
        *,
        model_path: str,
        config_path: str | None = None,
        sample_rate: int = 22050,
    ) -> None:
        piper = require("piper", backend="piper", extra="piper")
        self._voice = piper.PiperVoice.load(model_path, config_path=config_path)
        self._sample_rate = sample_rate

    async def _synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        audio_chunks = list(self._voice.synthesize_stream_raw(text))
        for i, raw in enumerate(audio_chunks):
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            yield TtsChunk(
                samples=samples,
                sample_rate=self._sample_rate,
                segment_index=segment_index,
                utterance_id=utterance_id,
                final=(i == len(audio_chunks) - 1),
            )

    def synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        """Return the streamed chunk iterator for ``text``."""
        return self._synthesize(text, segment_index=segment_index, utterance_id=utterance_id)
