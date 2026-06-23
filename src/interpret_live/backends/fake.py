"""Deterministic, offline test doubles for every backend protocol.

These fakes are scripted (no network, no models, no real audio) and use only the
injected :class:`~interpret_live.clock.Clock` for pacing — **never**
:func:`asyncio.sleep` — so the whole suite is reproducible and runs in well under
a wall-clock second.

* :class:`FakeSTT` — emits a scripted sequence of partial hypotheses, including
  mid-word revisions, with a per-partial clock delay.
* :class:`FakeMT` — a deterministic, configurable source→target mapping.
* :class:`FakeTTS` — emits fake audio frames whose length is proportional to the
  text, with a configurable per-call latency.
* :class:`FakeS2S` — a scripted audio→audio mapping for the unified path.
* :class:`FakeVAD` — replays a scripted speech/silence pattern.

Fake audio edges (:class:`FakeAudioSource` / :class:`FakeAudioSink`) live in
:mod:`interpret_live.audio_io` so they can back both fakes and real I/O.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping

import numpy as np

from ..clock import Clock
from ..types import AudioFrame, Hypothesis, Segment, TtsChunk
from ..vad import VAD

__all__ = ["FakeMT", "FakeS2S", "FakeSTT", "FakeTTS", "FakeVAD"]


class FakeSTT:
    """Replay scripted partial hypotheses, draining the audio iterator.

    The audio is consumed (so backpressure/fan-out behave realistically) but the
    emitted hypotheses come from a fixed script — one or more *utterances*, each a
    list of partials whose final element should carry ``is_final=True``.

    Args:
        script: Either a single utterance (``list[Hypothesis]``) or a list of
            utterances (``list[list[Hypothesis]]``). A single utterance is
            wrapped automatically.
        clock: Injected clock for inter-partial pacing.
        partial_delay_ms: Logical delay between consecutive partials.
        drain_audio: When ``True`` (default) fully consume one frame per partial
            from the audio iterator so the source/queues advance in lock-step.
        gate: Optional async callback awaited *before each utterance after the
            first*. A test can hold back a "resumed" utterance until a barge-in
            has been handled, faithfully modelling speech that comes *after* the
            interrupt (rather than racing the manual clock).
    """

    def __init__(
        self,
        script: list[Hypothesis] | list[list[Hypothesis]],
        *,
        clock: Clock,
        partial_delay_ms: int = 50,
        drain_audio: bool = True,
        gate: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self._utterances = _normalize_script(script)
        self._clock = clock
        self._partial_delay_ms = partial_delay_ms
        self._drain_audio = drain_audio
        self._gate = gate

    async def _stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        audio_iter = aiter(audio)
        for u_index, utterance in enumerate(self._utterances):
            if u_index > 0 and self._gate is not None:
                await self._gate(u_index)
            for partial in utterance:
                if self._drain_audio:
                    with contextlib.suppress(StopAsyncIteration):
                        await anext(audio_iter)
                await self._clock.sleep(self._partial_delay_ms)
                yield partial

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        """Return the scripted hypothesis stream for ``audio``."""
        return self._stream(audio)


class FakeMT:
    """Deterministic translation via an explicit mapping (word- or phrase-level).

    Unknown source text is passed through with a ``prefix`` marker so tests stay
    legible. A per-call ``latency_ms`` clock delay models translation cost.

    Args:
        mapping: Exact source-text → target-text overrides (segment text).
        word_map: Per-word source→target overrides applied when ``mapping`` has no
            entry for the whole segment.
        prefix: Marker prepended to passthrough output (e.g. ``"~"``).
        clock: Injected clock for latency.
        latency_ms: Logical delay per :meth:`translate` call.
    """

    def __init__(
        self,
        mapping: Mapping[str, str] | None = None,
        *,
        word_map: Mapping[str, str] | None = None,
        prefix: str = "",
        clock: Clock,
        latency_ms: int = 30,
    ) -> None:
        self._mapping = dict(mapping or {})
        self._word_map = dict(word_map or {})
        self._prefix = prefix
        self._clock = clock
        self._latency_ms = latency_ms
        #: Segments seen by this MT, in order — tests assert these are all closed.
        self.seen: list[Segment] = []

    async def translate(self, segment: Segment, context: tuple[str, ...] = ()) -> str:
        """Translate a CLOSED ``segment``; record it for assertions."""
        if not segment.closed:
            raise AssertionError(
                f"FakeMT received a non-closed segment (index={segment.index!r}); "
                "MT must only ever run on closed segments"
            )
        self.seen.append(segment)
        await self._clock.sleep(self._latency_ms)
        if segment.text in self._mapping:
            return self._mapping[segment.text]
        words = [self._word_map.get(w, w) for w in segment.text.split()]
        return self._prefix + " ".join(words)


class FakeTTS:
    """Emit fake PCM whose length is proportional to the synthesized text.

    Each call yields one or more :class:`TtsChunk`. The number of chunks and the
    per-chunk clock delay let tests model streaming synthesis and observe
    first-audio-out timing.

    Args:
        clock: Injected clock for per-chunk pacing.
        sample_rate: Output sample rate.
        ms_per_char: Audio milliseconds generated per text character.
        chunks: How many chunks to split a synthesis into (>= 1).
        chunk_latency_ms: Logical delay before each chunk is yielded.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        sample_rate: int = 16000,
        ms_per_char: int = 20,
        chunks: int = 1,
        chunk_latency_ms: int = 20,
    ) -> None:
        if chunks < 1:
            raise ValueError(f"chunks must be >= 1, got {chunks}")
        self._clock = clock
        self._sample_rate = sample_rate
        self._ms_per_char = ms_per_char
        self._chunks = chunks
        self._chunk_latency_ms = chunk_latency_ms

    async def _synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        total_ms = max(self._ms_per_char, len(text) * self._ms_per_char)
        per_chunk_ms = max(1, total_ms // self._chunks)
        per_chunk_samples = max(1, int(per_chunk_ms * self._sample_rate / 1000))
        for i in range(self._chunks):
            await self._clock.sleep(self._chunk_latency_ms)
            samples = np.full(per_chunk_samples, 0.1, dtype=np.float32)
            yield TtsChunk(
                samples=samples,
                sample_rate=self._sample_rate,
                segment_index=segment_index,
                utterance_id=utterance_id,
                final=(i == self._chunks - 1),
            )

    def synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        """Return the streamed chunk iterator for ``text``."""
        return self._synthesize(text, segment_index=segment_index, utterance_id=utterance_id)


class FakeS2S:
    """Scripted unified speech-to-speech: drains audio, emits scripted chunks.

    Args:
        chunks_per_utterance: How many target chunks to emit per utterance.
        clock: Injected clock for per-chunk pacing.
        sample_rate: Output sample rate.
        chunk_latency_ms: Logical delay before each emitted chunk.
        frames_before_output: Audio frames to consume before the first output
            (models the provider buffering input before speaking).
    """

    def __init__(
        self,
        *,
        chunks_per_utterance: int = 2,
        clock: Clock,
        sample_rate: int = 16000,
        chunk_latency_ms: int = 40,
        frames_before_output: int = 1,
    ) -> None:
        self._chunks = chunks_per_utterance
        self._clock = clock
        self._sample_rate = sample_rate
        self._chunk_latency_ms = chunk_latency_ms
        self._frames_before_output = frames_before_output
        self._interrupted = False
        #: Number of times :meth:`interrupt` was called (asserted in tests).
        self.interrupt_count = 0

    async def _stream(
        self, audio: AsyncIterator[AudioFrame], *, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        audio_iter = aiter(audio)
        for _ in range(self._frames_before_output):
            try:
                await anext(audio_iter)
            except StopAsyncIteration:
                break
        for i in range(self._chunks):
            if self._interrupted:
                return
            await self._clock.sleep(self._chunk_latency_ms)
            samples = np.full(self._sample_rate // 10, 0.1, dtype=np.float32)
            yield TtsChunk(
                samples=samples,
                sample_rate=self._sample_rate,
                segment_index=i,
                utterance_id=utterance_id,
                final=(i == self._chunks - 1),
            )

    def stream(
        self, audio: AsyncIterator[AudioFrame], *, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        """Return the scripted target-audio stream for ``audio``."""
        self._interrupted = False
        return self._stream(audio, utterance_id=utterance_id)

    async def interrupt(self) -> None:
        """Mark the stream interrupted so it stops yielding further chunks."""
        self._interrupted = True
        self.interrupt_count += 1


class FakeVAD(VAD):
    """Replay a scripted speech/silence decision per frame.

    Args:
        pattern: Iterable of booleans (``True`` = speech). Consumed one per
            :meth:`is_speech` call; exhausted entries default to ``False``.
    """

    def __init__(self, pattern: Iterable[bool]) -> None:
        self._pattern = list(pattern)
        self._i = 0

    def is_speech(self, frame: AudioFrame) -> bool:
        """Return the next scripted decision (``False`` once exhausted)."""
        if self._i < len(self._pattern):
            val = self._pattern[self._i]
            self._i += 1
            return val
        return False

    def reset(self) -> None:
        self._i = 0


def _normalize_script(
    script: list[Hypothesis] | list[list[Hypothesis]],
) -> list[list[Hypothesis]]:
    """Wrap a single-utterance script into the list-of-utterances shape."""
    if script and isinstance(script[0], Hypothesis):
        return [list(script)]
    return script  # type: ignore[return-value]
