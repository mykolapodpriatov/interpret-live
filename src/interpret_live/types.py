"""Core value types shared across :mod:`interpret_live`.

These types are intentionally small, immutable where practical, and free of any
heavy or optional dependency so that the streaming-orchestration core and the
deterministic fakes can be imported with only the light default install
(stdlib + numpy + pydantic).

The audio edge is modelled by the :class:`AudioSource` / :class:`AudioSink`
protocols. The mic fan-out (a single source feeding both ``STT.stream`` and the
``BargeInDetector``) is provided by :func:`interpret_live.audio_io.tee`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AudioFrame",
    "AudioSink",
    "AudioSource",
    "CommitResult",
    "Hypothesis",
    "MetricEvent",
    "MetricKind",
    "Segment",
    "Token",
    "TtsChunk",
]


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A single fixed-size block of mono PCM audio.

    Attributes:
        samples: ``float32`` samples in the range ``[-1.0, 1.0]`` (one channel).
        sample_rate: Sampling rate in Hz (e.g. ``16000``).
        t_ms: Logical timestamp of the *start* of this frame, in milliseconds,
            measured against the injected :class:`~interpret_live.clock.Clock`.
    """

    samples: NDArray[np.float32]
    sample_rate: int
    t_ms: int

    @property
    def duration_ms(self) -> int:
        """Frame duration in milliseconds, derived from sample count + rate."""
        if self.sample_rate <= 0:
            return 0
        return round(1000 * len(self.samples) / self.sample_rate)


@dataclass(frozen=True, slots=True)
class Token:
    """A single ASR word token with its time span.

    The stabilizer and segmenter operate on *word-level* ``Token`` objects, not
    on the MT model's subword pieces. ``start_ms``/``end_ms`` are the recognised
    span against the injected clock.
    """

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """The ASR's current best partial transcript.

    Attributes:
        tokens: Ordered word tokens of the current partial transcript.
        is_final: ``True`` when the ASR considers this hypothesis final for the
            current utterance. On a final hypothesis the stabilizer
            force-commits the remaining tail and resets its window.
    """

    tokens: tuple[Token, ...]
    is_final: bool = False

    @classmethod
    def of(cls, *texts: str, is_final: bool = False, step_ms: int = 100) -> Hypothesis:
        """Build a hypothesis from bare word strings (test/convenience helper).

        Tokens are laid out back-to-back, each ``step_ms`` long, starting at 0.
        """
        toks = tuple(
            Token(text=t, start_ms=i * step_ms, end_ms=(i + 1) * step_ms)
            for i, t in enumerate(texts)
        )
        return cls(tokens=toks, is_final=is_final)


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Result of feeding one partial hypothesis to the stabilizer.

    Attributes:
        newly_committed: Tokens promoted to the stable prefix *this* step.
        committed_prefix: The full stable prefix after this step.
        tentative_tail: Tokens still tentative (after the committed prefix).
    """

    newly_committed: tuple[Token, ...]
    committed_prefix: tuple[Token, ...]
    tentative_tail: tuple[Token, ...]

    @property
    def text(self) -> str:
        """Whitespace-joined text of the newly committed tokens."""
        return " ".join(t.text for t in self.newly_committed)


@dataclass(frozen=True, slots=True)
class Segment:
    """A committed, translatable unit of source text.

    Attributes:
        text: Whitespace-joined source text of the segment.
        tokens: The committed source tokens that make up the segment.
        token_span: ``(start_index, end_index)`` half-open span into the
            utterance's committed-token stream.
        closed: ``True`` when the segment ended on terminal punctuation or a
            ``max_segment_tokens`` flush. Only *closed* segments are translated.
        index: 0-based ordinal of the segment within its utterance.
    """

    text: str
    tokens: tuple[Token, ...]
    token_span: tuple[int, int]
    closed: bool
    index: int


@dataclass(frozen=True, slots=True)
class TtsChunk:
    """A block of synthesized target-language audio for one (sub)segment.

    Attributes:
        samples: ``float32`` PCM samples of synthesized speech.
        sample_rate: Sampling rate in Hz.
        segment_index: Index of the source :class:`Segment` this audio realises.
        utterance_id: The utterance this chunk belongs to (used to discard
            stale chunks on barge-in).
        final: ``True`` for the last chunk of a segment's synthesis.
    """

    samples: NDArray[np.float32]
    sample_rate: int
    segment_index: int
    utterance_id: str
    final: bool = False


MetricKind = Literal[
    "utterance_start",
    "first_tts_out",
    "interrupt",
    "sink_stopped",
    "commit",
    "post_commit_disagreement",
]


@dataclass(frozen=True, slots=True)
class MetricEvent:
    """A single timestamped event appended to the pipeline's in-memory log.

    ``t_ms`` is read from the injected clock so the whole metric stream is
    deterministic in tests. :mod:`interpret_live.metrics` derives first-audio-out
    latency, commit lag, retraction count and barge-in-stop time from this log.

    Attributes:
        kind: The event kind (see :data:`MetricKind`).
        t_ms: Injected-clock time of the event, in milliseconds.
        utterance_id: The utterance the event belongs to.
        detail: Optional small structured payload (e.g. segment index).
    """

    kind: MetricKind
    t_ms: int
    utterance_id: str
    detail: dict[str, int | str] = field(default_factory=dict)


@runtime_checkable
class AudioSource(Protocol):
    """An async source of :class:`AudioFrame` (the mic, or a test fixture)."""

    def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield audio frames until the source is exhausted."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """An async sink that plays synthesized audio and can be stopped.

    ``stop()`` aborts the currently playing chunk and discards anything queued;
    the moment ``stop()`` returns is the endpoint of the ``barge-in-stop``
    metric.
    """

    async def play(self, chunk: TtsChunk) -> None:
        """Play (or enqueue) one synthesized chunk."""
        ...

    async def stop(self) -> None:
        """Abort the current chunk and discard queued audio immediately."""
        ...
