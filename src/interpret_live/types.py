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
    "PlaybackCursor",
    "PlaybackGeneration",
    "PlaybackHandle",
    "PlaybackProgress",
    "PlaybackReceipt",
    "PlaybackRejectedError",
    "S2SAudioChunk",
    "S2SContentDone",
    "S2SEvent",
    "S2SInterruptTarget",
    "S2SProtocolError",
    "S2SResponseDone",
    "S2SResponseError",
    "S2SResponseStarted",
    "S2SSpeechCommitted",
    "S2SSpeechStarted",
    "Segment",
    "Token",
    "TtsChunk",
]


def _validate_mono_samples(samples: NDArray[np.float32], *, owner: str) -> None:
    """Validate the canonical in-process audio contract for ``samples``.

    Canonical audio is mono, normalized ``float32`` in ``[-1.0, 1.0]``; PCM16 is
    a wire/model-boundary encoding only (see :mod:`interpret_live.audio_codec`).
    """
    if not isinstance(samples, np.ndarray):
        raise TypeError(f"{owner}.samples must be a numpy array, got {type(samples).__name__}")
    if samples.dtype != np.float32:
        raise ValueError(f"{owner}.samples must be float32, got {samples.dtype}")
    if samples.ndim != 1:
        raise ValueError(f"{owner}.samples must be one-dimensional (mono), got {samples.ndim}D")
    if samples.size:
        if not np.isfinite(samples).all():
            raise ValueError(f"{owner}.samples must be finite (no NaN/inf)")
        peak = float(np.abs(samples).max())
        if peak > 1.0:
            raise ValueError(
                f"{owner}.samples must be normalized to [-1.0, 1.0]; peak was {peak:.6f} "
                "(clip or rescale at the boundary that produced them)"
            )


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A single fixed-size block of mono PCM audio.

    Attributes:
        samples: ``float32`` samples in the range ``[-1.0, 1.0]`` (one channel).
        sample_rate: Sampling rate in Hz (e.g. ``16000``); carried by every
            frame because the canonical type is rate-annotated, not implied.
        t_ms: Logical timestamp of the *start* of this frame, in milliseconds,
            measured against the injected :class:`~interpret_live.clock.Clock`.
    """

    samples: NDArray[np.float32]
    sample_rate: int
    t_ms: int

    def __post_init__(self) -> None:
        _validate_mono_samples(self.samples, owner="AudioFrame")
        if self.sample_rate <= 0:
            raise ValueError(f"AudioFrame.sample_rate must be > 0, got {self.sample_rate}")

    @property
    def duration_ms(self) -> int:
        """Frame duration in milliseconds, derived from sample count + rate."""
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
        source_turn_id: The STT adapter's upstream turn identifier. Optional
            for legacy/fake construction, but every live STT turn populates it
            (identical on every partial/final of the same detected utterance)
            so the pipeline can discard exactly the interrupted turn's stale
            output and never a fresh turn's.
        speech_started_at_ms: Immutable source speech onset (the first
            VAD-positive input frame's timestamp, injected-clock domain),
            repeated on every partial/final of the turn. Drives the
            ``utterance_start`` metric so first-audio latency starts at actual
            speech onset, not first-decode arrival.
    """

    tokens: tuple[Token, ...]
    is_final: bool = False
    source_turn_id: str | None = None
    speech_started_at_ms: int | None = None

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

    def __post_init__(self) -> None:
        _validate_mono_samples(self.samples, owner="TtsChunk")
        if self.sample_rate <= 0:
            raise ValueError(f"TtsChunk.sample_rate must be > 0, got {self.sample_rate}")

    @property
    def duration_ms(self) -> int:
        """Chunk duration in milliseconds, derived from sample count + rate."""
        return round(1000 * len(self.samples) / self.sample_rate)


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


class PlaybackRejectedError(RuntimeError):
    """A ``schedule()`` was rejected because its generation was stopped.

    Raised to callers blocked on (or validating for) sink capacity whose
    :class:`PlaybackGeneration` has been invalidated by ``stop()``; the chunk
    was **not** enqueued and must be discarded by the caller.
    """


@dataclass(frozen=True, slots=True)
class PlaybackGeneration:
    """An opaque, monotonically ordered token owning a run of sink playback.

    The sink issues one generation per pipeline utterance / provider response
    (:meth:`AudioSink.new_generation`). Only one generation may own the sink at
    a time; ``stop(generation)`` invalidates exactly that generation.
    """

    seq: int


@dataclass(frozen=True, slots=True)
class PlaybackProgress:
    """An immutable snapshot of one scheduled chunk's presentation state.

    Returned by handle notifications (:class:`PlaybackHandle`) and by
    ``stop()`` snapshots. "Presented" counts only audio whose presentation
    (DAC) time has passed in the injected clock domain — never samples that
    are merely queued or sitting in a device buffer.

    Attributes:
        generation_seq: The owning :class:`PlaybackGeneration`'s sequence.
        utterance_id: The utterance/response the chunk belongs to.
        segment_index: The source segment index carried by the chunk.
        chunk_seq: Scheduling ordinal of the chunk within its generation.
        source_rate: Sample rate of the chunk as scheduled (source content).
        source_samples_total: Total source samples in the chunk.
        source_samples_presented: Source-equivalent samples audible so far.
        device_rate: Output device rate the chunk was presented at.
        device_frames_presented: Device frames audible so far.
        first_audible_t_ms: Injected-clock time the first sample became
            audible, or ``None`` if presentation never started.
        interrupted: ``True`` when the chunk was cut short by ``stop()``.
        completed: ``True`` when presentation finished (fully or interrupted).
    """

    generation_seq: int
    utterance_id: str
    segment_index: int
    chunk_seq: int
    source_rate: int
    source_samples_total: int
    source_samples_presented: int
    device_rate: int
    device_frames_presented: int
    first_audible_t_ms: int | None
    interrupted: bool
    completed: bool


#: A receipt is a resolved progress snapshot (started/completed notification).
PlaybackReceipt = PlaybackProgress


@runtime_checkable
class PlaybackHandle(Protocol):
    """The sink's per-chunk handle: notifications plus progress snapshots."""

    @property
    def chunk(self) -> TtsChunk:
        """The scheduled chunk."""
        ...

    @property
    def generation(self) -> PlaybackGeneration:
        """The generation that owns this chunk."""
        ...

    async def started(self) -> PlaybackProgress:
        """Wait until the first sample is audible (or the chunk is stopped)."""
        ...

    async def completed(self) -> PlaybackProgress:
        """Wait until presentation finished (fully played or interrupted)."""
        ...

    def progress(self) -> PlaybackProgress:
        """Return the current immutable presentation snapshot."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """An async playback sink with generation-scoped scheduling and stop.

    The playback contract (architecture decision: bounded, gapless, killable):

    * ``schedule(generation, chunk)`` waits only for bounded sink capacity —
      never for audible completion — so the next chunk can be buffered before
      the current one ends (gapless lookahead). It validates the generation
      before waiting and again under the sink lock immediately before enqueue,
      raising :class:`PlaybackRejectedError` if the generation was stopped.
    * Only one generation may own the sink at a time; a schedule for the next
      generation waits until the previous generation drains or is stopped.
    * ``stop(generation)`` invalidates that generation first (under the same
      lock), atomically snapshots every affected handle's presented position,
      aborts queued output through an independent control path, resolves
      completions as interrupted, and wakes blocked schedules with a typed
      rejection. The moment ``stop()`` returns is the ``barge-in-stop`` metric
      endpoint.
    * ``drain()`` awaits presentation of everything scheduled (normal EOF).
    * ``aclose()`` is idempotent and releases the underlying device/tasks.
    """

    def new_generation(self) -> PlaybackGeneration:
        """Issue the next monotonic playback generation token."""
        ...

    async def schedule(self, generation: PlaybackGeneration, chunk: TtsChunk) -> PlaybackHandle:
        """Enqueue ``chunk`` under ``generation``; returns once buffered."""
        ...

    async def drain(self) -> None:
        """Wait until every scheduled chunk has finished presenting."""
        ...

    async def stop(self, generation: PlaybackGeneration) -> tuple[PlaybackProgress, ...]:
        """Stop ``generation`` immediately; return frozen progress snapshots."""
        ...

    async def aclose(self) -> None:
        """Release the sink's device/tasks (idempotent)."""
        ...


# ----- Persistent S2S provider protocol types (plan Task 6) -------------------


class S2SProtocolError(RuntimeError):
    """A provider event referenced an unknown response/input item."""


class S2SResponseError(RuntimeError):
    """A response ended with a failed/unexpected status.

    Attributes:
        response_id: The provider response that failed.
        status: The terminal status (``failed``, ``incomplete``, or an
            unexpected ``cancelled``).
        reason: Optional provider-supplied detail.
    """

    def __init__(self, response_id: str, status: str, reason: str | None = None) -> None:
        self.response_id = response_id
        self.status = status
        self.reason = reason
        detail = f" ({reason})" if reason else ""
        super().__init__(f"provider response {response_id!r} ended {status}{detail}")


@dataclass(frozen=True, slots=True)
class S2SSpeechStarted:
    """Provider detected input speech start (a required control event).

    ``source_started_at_ms`` is already translated into the injected
    application :class:`~interpret_live.clock.Clock` domain by the adapter.
    """

    input_item_id: str
    source_started_at_ms: int


@dataclass(frozen=True, slots=True)
class S2SSpeechCommitted:
    """Provider committed the input turn's audio buffer."""

    input_item_id: str


@dataclass(frozen=True, slots=True)
class S2SResponseStarted:
    """Provider began generating a response for a committed input turn."""

    response_id: str
    input_item_id: str | None = None


@dataclass(frozen=True, slots=True)
class S2SAudioChunk:
    """One block of provider response audio with full provenance.

    Attributes:
        samples: Canonical mono float32 samples.
        sample_rate: Rate of ``samples`` in Hz.
        response_id: Owning provider response.
        item_id: Provider output item id.
        output_index: Output index within the response.
        content_index: Content index within the output item.
        final: ``True`` when this is the content stream's last audio block
            (content-final status — NOT response completion).
    """

    samples: NDArray[np.float32]
    sample_rate: int
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    final: bool = False

    def __post_init__(self) -> None:
        _validate_mono_samples(self.samples, owner="S2SAudioChunk")
        if self.sample_rate <= 0:
            raise ValueError(f"S2SAudioChunk.sample_rate must be > 0, got {self.sample_rate}")


@dataclass(frozen=True, slots=True)
class S2SContentDone:
    """The response's audio content stream finished (response may continue)."""

    response_id: str
    item_id: str
    content_index: int


@dataclass(frozen=True, slots=True)
class S2SResponseDone:
    """The response reached a terminal status.

    Only ``status="completed"`` is a natural completion. An expected
    ``cancelled`` for an abandoned response is an interrupt acknowledgement;
    ``failed`` / unexpected ``cancelled`` / ``incomplete`` surface as typed
    errors and never emit a natural final or roll a fresh turn.
    """

    response_id: str
    status: str
    reason: str | None = None


#: Everything a persistent S2S provider stream may yield.
S2SEvent = (
    S2SSpeechStarted
    | S2SSpeechCommitted
    | S2SResponseStarted
    | S2SAudioChunk
    | S2SContentDone
    | S2SResponseDone
)


@dataclass(frozen=True, slots=True)
class PlaybackCursor:
    """How much of one response's audio was actually heard.

    ``audio_end_ms`` counts only source-rate audio whose presentation time has
    passed — never queued or device-buffered samples — always scoped to one
    response's item/content stream.
    """

    response_id: str
    item_id: str
    content_index: int
    audio_end_ms: int


@dataclass(frozen=True, slots=True)
class S2SInterruptTarget:
    """The exact response a barge-in must cancel (and truncate, if heard).

    ``cursor`` is ``None`` when the response was cancelled before any of its
    audio became audible.
    """

    response_id: str
    cursor: PlaybackCursor | None = None
