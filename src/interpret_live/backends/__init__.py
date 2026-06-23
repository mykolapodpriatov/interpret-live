"""Backend protocols for the two paths: pipeline (STT/MT/TTS) and unified S2S.

The pipeline path composes three async-streaming protocols — :class:`STT`,
:class:`MT`, :class:`TTS` — that the orchestrator wires together. The unified
S2S path is a single :class:`S2S` protocol (audio-in → translated-audio-out)
for cloud realtime providers.

Concrete backends live alongside this package:

* ``fake`` — deterministic, offline test doubles (always importable).
* ``whisper`` / ``nllb`` / ``piper`` — offline adapters behind optional extras.
* ``realtime`` / ``gemini`` / ``elevenlabs`` — cloud adapters behind extras.

All heavy adapters are import-guarded via
:func:`interpret_live.backends.guard.require`, so a missing extra raises a clear
"install interpret-live[...]" message rather than a raw ``ImportError``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from ..types import AudioFrame, Hypothesis, Segment, TtsChunk

__all__ = ["MT", "S2S", "STT", "TTS"]


@runtime_checkable
class STT(Protocol):
    """Streaming speech-to-text: audio frames → partial/final hypotheses."""

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        """Consume audio frames and yield a stream of partial hypotheses.

        The last hypothesis of an utterance has ``is_final=True``. Implementations
        must use the injected clock (never :func:`asyncio.sleep`) for any pacing.
        """
        ...


@runtime_checkable
class MT(Protocol):
    """Machine translation of a single closed source segment."""

    async def translate(
        self,
        segment: Segment,
        context: tuple[str, ...] = (),
    ) -> str:
        """Translate ``segment`` (a CLOSED unit) given rolling source ``context``.

        ``context`` is the left-context word strings (whole tokens); MT must only
        ever be called on closed segments — never a partial clause.
        """
        ...


@runtime_checkable
class TTS(Protocol):
    """Streaming text-to-speech: target text → a stream of audio chunks."""

    def synthesize(
        self,
        text: str,
        *,
        segment_index: int,
        utterance_id: str,
    ) -> AsyncIterator[TtsChunk]:
        """Yield :class:`TtsChunk` audio for ``text``; interruptible via cancel."""
        ...


@runtime_checkable
class S2S(Protocol):
    """Unified speech-to-speech: source audio → translated target audio.

    Models the cloud-realtime path (OpenAI Realtime / Gemini Live) that does
    STT+MT+TTS internally. The harness does not see ASR partials on this path, so
    audio-stage stabilization is the provider's responsibility (documented).
    """

    def stream(
        self,
        audio: AsyncIterator[AudioFrame],
        *,
        utterance_id: str,
    ) -> AsyncIterator[TtsChunk]:
        """Yield translated target-audio chunks for the incoming ``audio``."""
        ...

    async def interrupt(self) -> None:
        """Send the provider's cancel/interrupt to halt in-flight synthesis."""
        ...
