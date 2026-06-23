"""Offline streaming STT adapter backed by faster-whisper (``whisper`` extra).

Import-guarded: constructing :class:`WhisperSTT` without the ``whisper`` extra
raises a clear :class:`~interpret_live.backends.guard.MissingExtraError` rather
than a raw ``ImportError``. The adapter buffers incoming :class:`AudioFrame`s,
runs ``faster_whisper`` over a growing window, and yields word-level
:class:`Hypothesis` objects (partials, then a final) for the stabilizer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from ..clock import Clock, RealClock
from ..types import AudioFrame, Hypothesis, Token
from .guard import require

__all__ = ["WhisperSTT"]


class WhisperSTT:
    """Streaming STT over faster-whisper.

    Args:
        model_size: faster-whisper model id (e.g. ``"small"``, ``"base"``).
        language: Source language hint (e.g. ``"en"``).
        device: ``"cpu"`` / ``"cuda"``.
        compute_type: faster-whisper compute type (e.g. ``"int8"``).
        window_ms: Re-decode window length while streaming.
        clock: Injected clock (defaults to a real clock in production).
    """

    def __init__(
        self,
        *,
        model_size: str = "small",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        window_ms: int = 4000,
        clock: Clock | None = None,
    ) -> None:
        fw = require("faster_whisper", backend="whisper", extra="whisper")
        self._model = fw.WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language
        self._window_ms = window_ms
        self._clock = clock or RealClock()

    async def _stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        buffer: list[float] = []
        sample_rate = 16000
        async for frame in audio:
            sample_rate = frame.sample_rate
            buffer.extend(frame.samples.tolist())
            window_samples = int(self._window_ms * sample_rate / 1000)
            window = np.asarray(buffer[-window_samples:], dtype=np.float32)
            segments, _info = self._model.transcribe(
                window, language=self._language, word_timestamps=True
            )
            tokens = _segments_to_tokens(segments)
            if tokens:
                yield Hypothesis(tokens=tuple(tokens), is_final=False)
        # Final pass over the full buffer.
        full = np.asarray(buffer, dtype=np.float32)
        segments, _info = self._model.transcribe(
            full, language=self._language, word_timestamps=True
        )
        yield Hypothesis(tokens=tuple(_segments_to_tokens(segments)), is_final=True)

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        """Return the streaming hypothesis iterator for ``audio``."""
        return self._stream(audio)


def _segments_to_tokens(segments: object) -> list[Token]:
    """Flatten faster-whisper word timestamps into word-level tokens."""
    tokens: list[Token] = []
    for seg in segments:  # type: ignore[attr-defined]
        for word in getattr(seg, "words", None) or []:
            tokens.append(
                Token(
                    text=word.word.strip(),
                    start_ms=int(word.start * 1000),
                    end_ms=int(word.end * 1000),
                )
            )
    return tokens
