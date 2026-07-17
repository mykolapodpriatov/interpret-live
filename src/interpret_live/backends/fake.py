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
from dataclasses import dataclass

import numpy as np

from ..clock import Clock
from ..types import (
    AudioFrame,
    Hypothesis,
    S2SAudioChunk,
    S2SContentDone,
    S2SEvent,
    S2SInterruptTarget,
    S2SResponseDone,
    S2SResponseStarted,
    S2SSpeechCommitted,
    S2SSpeechStarted,
    Segment,
    TtsChunk,
)
from ..vad import VAD

__all__ = ["FakeMT", "FakeS2S", "FakeS2STurn", "FakeSTT", "FakeTTS", "FakeVAD"]


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


@dataclass(frozen=True, slots=True)
class FakeS2STurn:
    """One scripted provider turn for :class:`FakeS2S`.

    Attributes:
        frames_before: Source frames consumed before speech-start is emitted.
        chunks: Response audio chunks (each ``chunk_ms`` of audio).
        chunk_ms: Duration of each chunk at the fake's sample rate.
        status: Terminal ``response.done`` status when not interrupted.
        speech_started_at_ms: Explicit source onset; defaults to clock now.
        late_chunks_after_interrupt: Extra deltas emitted for this response
            *after* it was interrupted (they must be discarded downstream).
    """

    frames_before: int = 1
    chunks: int = 2
    chunk_ms: int = 100
    status: str = "completed"
    speech_started_at_ms: int | None = None
    late_chunks_after_interrupt: int = 0


class FakeS2S:
    """Scripted persistent speech-to-speech session (typed S2S events).

    Models a session-long provider connection: multiple input turns and
    responses, all control events and statuses, provider response/item
    metadata, targeted cancellation acknowledgement (expected ``cancelled``
    done), optional late old-response events, and post-interrupt output.

    Args:
        clock: Injected clock for per-chunk pacing.
        turns: Scripted turns; an int builds that many default turns.
        sample_rate: Output sample rate.
        chunk_latency_ms: Logical delay before each emitted chunk.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        turns: int | list[FakeS2STurn] = 1,
        sample_rate: int = 16000,
        chunk_latency_ms: int = 40,
    ) -> None:
        self._clock = clock
        self._turns = (
            [FakeS2STurn() for _ in range(turns)] if isinstance(turns, int) else list(turns)
        )
        self._sample_rate = sample_rate
        self._chunk_latency_ms = chunk_latency_ms
        self._interrupted: set[str] = set()
        #: Every interrupt target received, in order (asserted in tests).
        self.interrupt_targets: list[S2SInterruptTarget] = []

    @property
    def interrupt_count(self) -> int:
        """How many provider-side interrupts were requested."""
        return len(self.interrupt_targets)

    async def _stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[S2SEvent]:
        audio_iter = aiter(audio)
        for index, turn in enumerate(self._turns, start=1):
            item_id = f"item-{index}"
            response_id = f"resp-{index}"
            for _ in range(turn.frames_before):
                try:
                    await anext(audio_iter)
                except StopAsyncIteration:
                    return
            onset = (
                turn.speech_started_at_ms
                if turn.speech_started_at_ms is not None
                else self._clock.now_ms()
            )
            yield S2SSpeechStarted(input_item_id=item_id, source_started_at_ms=onset)
            yield S2SSpeechCommitted(input_item_id=item_id)
            yield S2SResponseStarted(response_id=response_id, input_item_id=item_id)
            samples_per_chunk = max(1, int(turn.chunk_ms * self._sample_rate / 1000))
            cancelled = False
            for i in range(turn.chunks):
                if response_id in self._interrupted:
                    cancelled = True
                    break
                await self._clock.sleep(self._chunk_latency_ms)
                if response_id in self._interrupted:
                    cancelled = True
                    break
                yield S2SAudioChunk(
                    samples=np.full(samples_per_chunk, 0.1, dtype=np.float32),
                    sample_rate=self._sample_rate,
                    response_id=response_id,
                    item_id=f"out-{index}",
                    output_index=0,
                    content_index=0,
                    final=(i == turn.chunks - 1),
                )
            if cancelled:
                # Late deltas from the already-cancelled response (must be
                # discarded downstream), then the interrupt acknowledgement.
                for _ in range(turn.late_chunks_after_interrupt):
                    yield S2SAudioChunk(
                        samples=np.full(samples_per_chunk, 0.1, dtype=np.float32),
                        sample_rate=self._sample_rate,
                        response_id=response_id,
                        item_id=f"out-{index}",
                        output_index=0,
                        content_index=0,
                        final=False,
                    )
                yield S2SResponseDone(response_id=response_id, status="cancelled")
                continue
            yield S2SContentDone(response_id=response_id, item_id=f"out-{index}", content_index=0)
            yield S2SResponseDone(response_id=response_id, status=turn.status)

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[S2SEvent]:
        """Return the scripted persistent event stream for ``audio``."""
        return self._stream(audio)

    async def interrupt(self, target: S2SInterruptTarget) -> None:
        """Record the targeted cancellation and stop that response's output."""
        self.interrupt_targets.append(target)
        self._interrupted.add(target.response_id)


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
