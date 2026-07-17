"""Audio edges: mic fan-out (``tee``), deterministic fakes, and real I/O.

The fakes (:class:`FakeAudioSource`, :class:`FakeAudioSink`) and the
:func:`tee` / :class:`Broadcaster` fan-out are pure-stdlib + numpy and always
importable. The real microphone/speaker streams (:class:`MicSource`,
:class:`SpeakerSink`) require the ``audio`` extra (sounddevice) and are
import-guarded so a missing dependency yields a clear install hint.

**Playback contract:** both sinks implement the generation-scoped
:class:`~interpret_live.types.AudioSink` protocol — ``schedule()`` waits only
for bounded capacity (gapless lookahead), ``stop(generation)`` invalidates
under the sink lock before releasing capacity and returns frozen
:class:`~interpret_live.types.PlaybackProgress` snapshots, and ``drain()``
awaits final presentation. :class:`FakeAudioSink` models presentation time on
the injected clock so the deterministic suite exercises the same semantics the
real speaker has.

**Mic fan-out:** a single :class:`~interpret_live.types.AudioSource` async
iterator cannot be read twice, yet both ``STT.stream`` and the
``BargeInDetector`` need the frames. :func:`tee` drives the source once and fans
each frame to *N* bounded subscriber queues, so the pipeline subscribes STT and
barge-in independently with real backpressure.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .audio_codec import StreamingResampler, chunk_duration_ms
from .clock import Clock, RealClock
from .types import (
    AudioFrame,
    AudioSource,
    PlaybackGeneration,
    PlaybackProgress,
    PlaybackRejectedError,
    TtsChunk,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "AudioStreamError",
    "Broadcaster",
    "DeviceInfo",
    "FakeAudioSink",
    "FakeAudioSource",
    "MicSource",
    "SpeakerSink",
    "list_devices",
    "tee",
    "validate_input_device",
    "validate_output_device",
]


class AudioStreamError(RuntimeError):
    """A typed real-audio failure: device validation, callback status, underrun."""


class Broadcaster:
    """Fan one :class:`AudioSource` out to *N* bounded subscriber queues.

    Use :meth:`subscribe` (before :meth:`run`) to obtain each consumer's frame
    iterator, then run :meth:`run` as a task to pump the source.

    **Real-time-mic semantics — drop, don't block:** a live microphone cannot be
    paused, so when a subscriber's bounded queue is full the *oldest* frame in
    that queue is dropped to make room for the newest. This keeps the fan-out
    free of head-of-line blocking: a slow or finished consumer (e.g. an STT stage
    that has stopped reading at end-of-utterance) never stalls the other
    subscribers (e.g. the barge-in detector). Each subscriber is still bounded.
    The end-of-stream sentinel is always delivered (it is never dropped).
    """

    _SENTINEL = object()

    def __init__(self, source: AudioSource, *, maxsize: int = 8) -> None:
        self._source = source
        self._maxsize = maxsize
        self._queues: list[asyncio.Queue[AudioFrame | object]] = []
        self._started = False
        #: Per-subscriber count of dropped (overflowed) frames, for diagnostics.
        self.dropped: list[int] = []

    def subscribe(self) -> AsyncIterator[AudioFrame]:
        """Register a new subscriber and return its frame iterator.

        Must be called before :meth:`run` starts pumping.
        """
        if self._started:
            raise RuntimeError("cannot subscribe after the broadcaster has started")
        queue: asyncio.Queue[AudioFrame | object] = asyncio.Queue(maxsize=self._maxsize)
        self._queues.append(queue)
        self.dropped.append(0)
        return self._consume(queue)

    async def _consume(
        self, queue: asyncio.Queue[AudioFrame | object]
    ) -> AsyncIterator[AudioFrame]:
        while True:
            item = await queue.get()
            if item is self._SENTINEL:
                return
            assert isinstance(item, AudioFrame)
            yield item

    def _offer(self, index: int, frame: AudioFrame) -> None:
        """Put ``frame`` on subscriber ``index``'s queue, dropping oldest if full.

        A drop is counted per-subscriber in :attr:`dropped` and logged at WARNING
        level. A dropped frame can desync a VAD's debounce (a missed silence/
        speech transition), so this is surfaced rather than silent; size the
        subscriber queue (``maxsize``) so normal operation never drops.
        """
        queue = self._queues[index]
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()  # drop the oldest frame
                self.dropped[index] += 1
                _logger.warning(
                    "Broadcaster subscriber %d dropped a frame (total dropped: %d); "
                    "a dropped frame can desync VAD debounce — increase the queue maxsize",
                    index,
                    self.dropped[index],
                )
        queue.put_nowait(frame)

    async def run(self) -> None:
        """Pump the source, fanning each frame to all subscriber queues.

        Frames are offered non-blockingly (drop-oldest on overflow); the
        end-of-stream sentinel is delivered with a blocking put so no consumer
        misses termination.
        """
        self._started = True
        async for frame in self._source.frames():
            for i in range(len(self._queues)):
                self._offer(i, frame)
            # Yield so consumers can drain between frames (keeps queues shallow
            # and lets the manual-clock harness make progress deterministically).
            await asyncio.sleep(0)
        # Deliver the sentinel, making room if a (still-active) consumer's queue
        # is full. A consumer that already returned simply never reads it.
        for queue in self._queues:
            while queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(self._SENTINEL)


def tee(
    source: AudioSource, n: int, *, maxsize: int = 8
) -> tuple[Broadcaster, list[AsyncIterator[AudioFrame]]]:
    """Create a :class:`Broadcaster` over ``source`` with ``n`` subscribers.

    Returns ``(broadcaster, subscribers)``. The caller schedules
    ``broadcaster.run()`` as a task and consumes each subscriber iterator
    (e.g. one for STT, one for the barge-in detector).
    """
    if n < 1:
        raise ValueError(f"tee needs n >= 1 subscribers, got {n}")
    bc = Broadcaster(source, maxsize=maxsize)
    subs = [bc.subscribe() for _ in range(n)]
    return bc, subs


class FakeAudioSource(AudioSource):
    """An :class:`AudioSource` that yields a fixed list of frames.

    Args:
        frames: The frames to emit, in order.
        clock: Injected clock for inter-frame pacing.
        frame_delay_ms: Logical delay before each frame (models real-time mic).
    """

    def __init__(
        self,
        frames: list[AudioFrame],
        *,
        clock: Clock,
        frame_delay_ms: int = 0,
    ) -> None:
        self._buffer = frames
        self._clock = clock
        self._frame_delay_ms = frame_delay_ms

    @classmethod
    def silence(
        cls,
        *,
        clock: Clock,
        count: int,
        frame_ms: int = 20,
        sample_rate: int = 16000,
        amplitude: float = 0.0,
        frame_delay_ms: int = 0,
    ) -> FakeAudioSource:
        """Build a source of ``count`` constant-amplitude frames."""
        n = max(1, int(frame_ms * sample_rate / 1000))
        frames = [
            AudioFrame(
                samples=np.full(n, amplitude, dtype=np.float32),
                sample_rate=sample_rate,
                t_ms=i * frame_ms,
            )
            for i in range(count)
        ]
        return cls(frames, clock=clock, frame_delay_ms=frame_delay_ms)

    async def _iter_frames(self) -> AsyncIterator[AudioFrame]:
        for frame in self._buffer:
            if self._frame_delay_ms:
                await self._clock.sleep(self._frame_delay_ms)
            yield frame

    def frames(self) -> AsyncIterator[AudioFrame]:
        """Return the async frame iterator."""
        return self._iter_frames()


class _Handle:
    """Shared :class:`~interpret_live.types.PlaybackHandle` implementation.

    Presentation state is mutated only by its owning sink (single event loop);
    every externally visible value is exposed through immutable
    :class:`PlaybackProgress` snapshots.
    """

    __slots__ = (
        "_chunk",
        "_completed",
        "_completed_evt",
        "_dev_end",
        "_dev_start",
        "_device_presented",
        "_device_rate",
        "_first_audible_t_ms",
        "_generation",
        "_interrupted",
        "_present_task",
        "_seq",
        "_source_presented",
        "_started_at_ms",
        "_started_evt",
    )

    def __init__(
        self,
        chunk: TtsChunk,
        generation: PlaybackGeneration,
        seq: int,
        device_rate: int,
    ) -> None:
        self._chunk = chunk
        self._generation = generation
        self._seq = seq
        self._device_rate = device_rate
        self._source_presented = 0
        self._device_presented = 0
        self._first_audible_t_ms: int | None = None
        self._started_at_ms: int | None = None
        self._interrupted = False
        self._completed = False
        self._started_evt = asyncio.Event()
        self._completed_evt = asyncio.Event()
        self._present_task: asyncio.Task[None] | None = None
        # Absolute device-frame span (used by the real sink's ledger).
        self._dev_start = 0
        self._dev_end = 0

    @property
    def chunk(self) -> TtsChunk:
        """The scheduled chunk."""
        return self._chunk

    @property
    def generation(self) -> PlaybackGeneration:
        """The generation that owns this chunk."""
        return self._generation

    def progress(self) -> PlaybackProgress:
        """Return the current immutable presentation snapshot."""
        return PlaybackProgress(
            generation_seq=self._generation.seq,
            utterance_id=self._chunk.utterance_id,
            segment_index=self._chunk.segment_index,
            chunk_seq=self._seq,
            source_rate=self._chunk.sample_rate,
            source_samples_total=len(self._chunk.samples),
            source_samples_presented=self._source_presented,
            device_rate=self._device_rate,
            device_frames_presented=self._device_presented,
            first_audible_t_ms=self._first_audible_t_ms,
            interrupted=self._interrupted,
            completed=self._completed,
        )

    async def started(self) -> PlaybackProgress:
        """Wait until the first sample is audible (or the chunk is stopped)."""
        await self._started_evt.wait()
        return self.progress()

    async def completed(self) -> PlaybackProgress:
        """Wait until presentation finished (fully played or interrupted)."""
        await self._completed_evt.wait()
        return self.progress()

    def _resolve(self, *, interrupted: bool) -> None:
        """Resolve both notifications (used by stop/close paths)."""
        self._interrupted = self._interrupted or interrupted
        self._completed = True
        self._started_evt.set()
        self._completed_evt.set()


class _GenerationLedger:
    """Generation bookkeeping shared by both sinks (single-loop state).

    Owns the monotonic generation counter, the invalidated set, and the
    outstanding-handle list that enforces single-generation sink ownership and
    bounded capacity. All access happens under the owning sink's condition.
    """

    __slots__ = ("capacity", "gen_seq", "invalidated", "outstanding", "seq_by_gen")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.gen_seq = 0
        self.invalidated: set[int] = set()
        self.outstanding: list[_Handle] = []
        self.seq_by_gen: dict[int, int] = {}

    def new_generation(self) -> PlaybackGeneration:
        self.gen_seq += 1
        return PlaybackGeneration(seq=self.gen_seq)

    def next_chunk_seq(self, generation: PlaybackGeneration) -> int:
        seq = self.seq_by_gen.get(generation.seq, 0)
        self.seq_by_gen[generation.seq] = seq + 1
        return seq

    def schedulable(self, generation: PlaybackGeneration) -> bool:
        """Predicate for schedule(): wake on invalidation OR eligible capacity."""
        if generation.seq in self.invalidated:
            return True  # wake the waiter so it can raise PlaybackRejectedError
        if len(self.outstanding) >= self.capacity:
            return False
        # Single-generation ownership: a newer generation may not enter while
        # any other generation still has outstanding audio.
        return all(h._generation.seq == generation.seq for h in self.outstanding)

    def check(self, generation: PlaybackGeneration) -> None:
        if generation.seq in self.invalidated:
            raise PlaybackRejectedError(
                f"playback generation {generation.seq} was stopped; chunk rejected"
            )

    def affected(self, generation: PlaybackGeneration) -> list[_Handle]:
        return [h for h in self.outstanding if h._generation.seq == generation.seq]


class FakeAudioSink:
    """Deterministic recording sink implementing the full playback contract.

    Presentation is modelled on the injected clock: chunks play sequentially,
    each occupying its real duration, so handle ``started``/``completed``
    notifications, partial-stop snapshots, capacity blocking, and generation
    ownership behave exactly like the real speaker — without hardware.

    Recorder attributes (``played``, ``stop_count``, ``stopped_at_ms``) are
    kept for test/bench assertions; ``played`` records chunks in presentation
    order at the moment they become audible.

    Args:
        clock: Injected clock used for presentation timing; without one the
            sink presents instantaneously (duration 0) which keeps legacy
            zero-clock tests meaningful.
        capacity: Maximum outstanding (scheduled, unfinished) chunks.
    """

    def __init__(self, *, clock: Clock | None = None, capacity: int = 8) -> None:
        self._clock = clock
        self._ledger = _GenerationLedger(capacity)
        self._cond = asyncio.Condition()
        self._last_handle: _Handle | None = None
        self._closed = False
        #: Every chunk that became audible, in presentation order.
        self.played: list[TtsChunk] = []
        #: Number of times :meth:`stop` was called.
        self.stop_count = 0
        #: Clock time (ms) of the first :meth:`stop`, or ``None``.
        self.stopped_at_ms: int | None = None

    # -- recorders kept for tests/bench ---------------------------------------

    @property
    def played_utterance_ids(self) -> list[str]:
        """The utterance id of each played chunk, in order."""
        return [c.utterance_id for c in self.played]

    def concatenated(self) -> NDArray[np.float32]:
        """All played samples concatenated (for whole-output assertions)."""
        if not self.played:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([c.samples for c in self.played])

    # -- AudioSink protocol ----------------------------------------------------

    def new_generation(self) -> PlaybackGeneration:
        """Issue the next monotonic playback generation token."""
        return self._ledger.new_generation()

    async def schedule(self, generation: PlaybackGeneration, chunk: TtsChunk) -> _Handle:
        """Enqueue ``chunk`` under ``generation``; returns once buffered."""
        async with self._cond:
            self._ledger.check(generation)  # validate before waiting
            await self._cond.wait_for(lambda: self._ledger.schedulable(generation))
            self._ledger.check(generation)  # re-validate under the lock
            handle = _Handle(
                chunk,
                generation,
                self._ledger.next_chunk_seq(generation),
                chunk.sample_rate,
            )
            prev = self._last_handle
            self._last_handle = handle
            self._ledger.outstanding.append(handle)
            handle._present_task = asyncio.create_task(
                self._present(handle, prev),
                name=f"fake-sink-present-g{generation.seq}c{handle._seq}",
            )
            return handle

    async def _present(self, handle: _Handle, prev: _Handle | None) -> None:
        """Present one chunk after its predecessor finishes (device timeline)."""
        if prev is not None:
            await prev._completed_evt.wait()
        if handle._completed_evt.is_set():
            return  # stopped while queued
        now = self._now()
        handle._first_audible_t_ms = now
        handle._started_at_ms = now
        self.played.append(handle._chunk)
        handle._started_evt.set()
        duration = chunk_duration_ms(len(handle._chunk.samples), handle._chunk.sample_rate)
        if self._clock is not None and duration > 0:
            await self._clock.sleep(round(duration))
        if handle._interrupted:
            return  # stop() already snapshotted a partial position
        handle._source_presented = len(handle._chunk.samples)
        handle._device_presented = len(handle._chunk.samples)
        handle._completed = True
        handle._completed_evt.set()
        async with self._cond:
            if handle in self._ledger.outstanding:
                self._ledger.outstanding.remove(handle)
            self._cond.notify_all()

    async def drain(self) -> None:
        """Wait until every scheduled chunk has finished presenting."""
        async with self._cond:
            await self._cond.wait_for(lambda: not self._ledger.outstanding)

    async def stop(self, generation: PlaybackGeneration) -> tuple[PlaybackProgress, ...]:
        """Stop ``generation`` immediately; return frozen progress snapshots."""
        tasks: list[asyncio.Task[None]] = []
        snapshots: list[PlaybackProgress] = []
        async with self._cond:
            self._ledger.invalidated.add(generation.seq)
            now = self._now()
            for handle in self._ledger.affected(generation):
                if handle._completed:
                    # Fully presented on the same tick the stop arrived: keep it
                    # a natural completion, just release it from the ledger.
                    snapshots.append(handle.progress())
                    self._ledger.outstanding.remove(handle)
                    continue
                handle._interrupted = True
                if handle._started_at_ms is not None:
                    elapsed = max(0, now - handle._started_at_ms)
                    total = len(handle._chunk.samples)
                    presented = min(total, round(handle._chunk.sample_rate * elapsed / 1000.0))
                    handle._source_presented = presented
                    handle._device_presented = presented
                handle._resolve(interrupted=True)
                snapshots.append(handle.progress())
                if handle._present_task is not None and not handle._present_task.done():
                    tasks.append(handle._present_task)
                self._ledger.outstanding.remove(handle)
            self.stop_count += 1
            if self.stopped_at_ms is None and self._clock is not None:
                self.stopped_at_ms = now
            self._cond.notify_all()
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return tuple(snapshots)

    async def aclose(self) -> None:
        """Stop every outstanding generation and release tasks (idempotent)."""
        if self._closed:
            return
        self._closed = True
        async with self._cond:
            gen_seqs = {h._generation.seq for h in self._ledger.outstanding}
        for seq in gen_seqs:
            await self.stop(PlaybackGeneration(seq=seq))

    def _now(self) -> int:
        return self._clock.now_ms() if self._clock is not None else 0


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One audio device as reported by PortAudio, for enumeration/validation."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool
    is_default_output: bool


def _require_sd() -> ModuleType:
    from .backends.guard import require

    return require("sounddevice", backend="audio", extra="audio")


def list_devices() -> list[DeviceInfo]:
    """Enumerate audio devices with directions, channels, and default rates."""
    sd = _require_sd()
    defaults = getattr(sd, "default", None)
    default_in, default_out = -1, -1
    if defaults is not None:
        pair = getattr(defaults, "device", (-1, -1))
        with contextlib.suppress(TypeError, IndexError):
            default_in, default_out = int(pair[0]), int(pair[1])
    infos: list[DeviceInfo] = []
    for i, dev in enumerate(sd.query_devices()):
        infos.append(
            DeviceInfo(
                index=i,
                name=str(dev.get("name", "")),
                max_input_channels=int(dev.get("max_input_channels", 0)),
                max_output_channels=int(dev.get("max_output_channels", 0)),
                default_samplerate=float(dev.get("default_samplerate", 0.0)),
                is_default_input=(i == default_in),
                is_default_output=(i == default_out),
            )
        )
    return infos


def validate_input_device(device: int | None, sample_rate: int) -> None:
    """Fail fast (typed) if ``device`` cannot capture mono at ``sample_rate``."""
    sd = _require_sd()
    try:
        sd.check_input_settings(device=device, channels=1, samplerate=sample_rate, dtype="float32")
    except Exception as exc:
        raise AudioStreamError(
            f"input device {device!r} cannot capture mono float32 at {sample_rate} Hz: {exc}"
        ) from exc


def validate_output_device(device: int | None, sample_rate: int) -> None:
    """Fail fast (typed) if ``device`` cannot play mono at ``sample_rate``."""
    sd = _require_sd()
    try:
        sd.check_output_settings(device=device, channels=1, samplerate=sample_rate, dtype="float32")
    except Exception as exc:
        raise AudioStreamError(
            f"output device {device!r} cannot play mono float32 at {sample_rate} Hz: {exc}"
        ) from exc


class MicSource(AudioSource):
    """Real microphone source backed by sounddevice (``audio`` extra).

    * The PortAudio callback never raises: on a full frame queue the oldest
      frame is dropped (counted in :attr:`dropped`, logged at WARNING).
    * Non-empty callback status surfaces as :class:`AudioStreamError` on the
      async consumer.
    * PortAudio ADC timestamps are calibrated once against the injected
      monotonic clock at stream start; every frame's ``t_ms`` lives in that
      shared clock domain (falling back to sample-count arithmetic when the
      host API reports no ADC time).
    * Cancelling the consuming iterator closes the input stream.

    Args:
        sample_rate: Capture rate in Hz (validated against the device).
        frame_ms: Frame duration in milliseconds.
        device: Input device index (``None`` = default).
        clock: Injected clock (defaults to a :class:`RealClock`).
        queue_frames: Bounded frame-queue size before drop-oldest applies.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        device: int | None = None,
        clock: Clock | None = None,
        queue_frames: int = 32,
    ) -> None:
        self._sd = _require_sd()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        if frame_ms <= 0:
            raise ValueError(f"frame_ms must be > 0, got {frame_ms}")
        if queue_frames < 1:
            raise ValueError(f"queue_frames must be >= 1, got {queue_frames}")
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._device = device
        self._clock = clock
        self._queue_frames = queue_frames
        #: Count of frames dropped because the consumer fell behind.
        self.dropped = 0

    async def _frames(self) -> AsyncIterator[AudioFrame]:
        validate_input_device(self._device, self._sample_rate)
        loop = asyncio.get_running_loop()
        clock = self._clock or RealClock()
        queue: asyncio.Queue[tuple[NDArray[np.float32], float, str | None]] = asyncio.Queue(
            maxsize=self._queue_frames
        )

        def _push(item: tuple[NDArray[np.float32], float, str | None]) -> None:
            # Runs on the event loop thread: queue mutation is race-free here.
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                    self.dropped += 1
                    _logger.warning(
                        "MicSource dropped a frame (total dropped: %d); "
                        "the consumer is not keeping up with the capture rate",
                        self.dropped,
                    )
            queue.put_nowait(item)

        def _callback(
            indata: NDArray[np.float32],
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            # PortAudio thread: never raise, never block.
            try:
                adc = float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0)
                note = str(status) if status else None
                loop.call_soon_threadsafe(_push, (indata[:, 0].copy(), adc, note))
            except RuntimeError:  # loop already closed during teardown
                pass

        blocksize = int(self._frame_ms * self._sample_rate / 1000)
        stream = self._sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=blocksize,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=_callback,
        )
        samples_seen = 0
        try:
            stream.start()
            # Calibrate the PortAudio stream clock to the injected clock once.
            base_stream_time = float(getattr(stream, "time", 0.0) or 0.0)
            base_clock_ms = clock.now_ms()
            while True:
                block, adc, note = await queue.get()
                if note is not None:
                    raise AudioStreamError(f"input stream reported status: {note}")
                if adc > 0.0 and base_stream_time > 0.0:
                    t_ms = round(base_clock_ms + (adc - base_stream_time) * 1000.0)
                else:
                    t_ms = round(base_clock_ms + 1000.0 * samples_seen / self._sample_rate)
                samples_seen += len(block)
                samples = np.clip(
                    np.asarray(block, dtype=np.float32), -1.0, 1.0
                )  # drivers may overshoot slightly; the frame contract is strict
                yield AudioFrame(samples=samples, sample_rate=self._sample_rate, t_ms=t_ms)
        finally:
            with contextlib.suppress(Exception):
                stream.abort()
            with contextlib.suppress(Exception):
                stream.close()

    def frames(self) -> AsyncIterator[AudioFrame]:
        """Return the live microphone frame iterator."""
        return self._frames()


class SpeakerSink:
    """Real speaker sink: one persistent callback-mode output stream.

    * Owns a single ``sounddevice.OutputStream`` for its whole life — never a
      global ``sd.play()`` or blocking ``write()`` per chunk.
    * Every chunk is resampled to the configured device rate by one stateful
      resampler (flushed at a final chunk, stop, or input-rate change).
    * A bounded ring buffer feeds the non-blocking PortAudio callback; a
      starved callback zero-fills (never replays stale memory) and counts an
      underrun.
    * DAC time is calibrated to the injected clock at stream start; presented
      progress counts only frames whose DAC time has passed, mapped back to
      source-content samples per chunk.
    * ``stop(generation)`` invalidates under the sink lock, snapshots presented
      positions, clears this sink's queued output through the stream's abort
      path, and wakes blocked schedules with :class:`PlaybackRejectedError`.

    Args:
        device: Output device index (``None`` = default).
        device_rate: Output rate in Hz (``None`` = the device's default rate).
        clock: Injected clock (defaults to a :class:`RealClock` at start).
        capacity: Maximum outstanding (scheduled, unfinished) chunks.
        ring_ms: Ring-buffer length in milliseconds of device audio.
    """

    def __init__(
        self,
        *,
        device: int | None = None,
        device_rate: int | None = None,
        clock: Clock | None = None,
        capacity: int = 8,
        ring_ms: int = 1000,
    ) -> None:
        self._sd = _require_sd()
        if device_rate is not None and device_rate <= 0:
            raise ValueError(f"device_rate must be > 0, got {device_rate}")
        if ring_ms <= 0:
            raise ValueError(f"ring_ms must be > 0, got {ring_ms}")
        self._device = device
        self._configured_rate = device_rate
        self._clock: Clock | None = clock
        self._ring_ms = ring_ms
        self._ledger = _GenerationLedger(capacity)
        self._cond = asyncio.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None
        self._rate = 0
        self._ring: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._write_abs = 0  # written by the loop side only
        self._read_abs = 0  # written by the callback thread only
        self._blocks: deque[tuple[int, int, int, float]] = deque(maxlen=256)
        self._resampler: StreamingResampler | None = None
        self._base_stream_time = 0.0
        self._base_clock_ms = 0
        self._max_presented = 0
        self._max_dev_end = 0
        self._closed = False
        self._wake_tasks: set[asyncio.Task[None]] = set()
        #: Number of starved callbacks (audio expected but the ring was short).
        self.underruns = 0
        #: Last non-empty callback status message, if any (typed surfacing).
        self.last_status: str | None = None

    # -- AudioSink protocol ----------------------------------------------------

    def new_generation(self) -> PlaybackGeneration:
        """Issue the next monotonic playback generation token."""
        return self._ledger.new_generation()

    async def schedule(self, generation: PlaybackGeneration, chunk: TtsChunk) -> _Handle:
        """Resample, buffer, and ledger one chunk; returns once buffered."""
        async with self._cond:
            self._ledger.check(generation)  # validate before waiting
            await self._cond.wait_for(lambda: self._ledger.schedulable(generation))
            self._ledger.check(generation)  # re-validate under the lock
            self._ensure_started()
            data = self._convert(chunk)
            handle = _Handle(
                chunk,
                generation,
                self._ledger.next_chunk_seq(generation),
                self._rate,
            )
            handle._dev_start = self._write_abs
            handle._dev_end = handle._dev_start + len(data)
            self._max_dev_end = max(self._max_dev_end, handle._dev_end)
            self._ledger.outstanding.append(handle)
            offset = 0
            while offset < len(data):
                if generation.seq in self._ledger.invalidated:
                    # stop() snapshotted this handle mid-write; nothing further
                    # may enqueue after the snapshot.
                    raise PlaybackRejectedError(
                        f"playback generation {generation.seq} was stopped mid-schedule"
                    )
                space = len(self._ring) - (self._write_abs - self._read_abs)
                n = min(space, len(data) - offset)
                if n > 0:
                    self._ring_write(data[offset : offset + n])
                    offset += n
                else:
                    await self._cond.wait()  # progress or stop wakes us
            return handle

    async def drain(self) -> None:
        """Wait until every scheduled chunk has finished presenting."""
        async with self._cond:
            await self._cond.wait_for(lambda: not self._ledger.outstanding)

    async def stop(self, generation: PlaybackGeneration) -> tuple[PlaybackProgress, ...]:
        """Stop ``generation`` now via the stream's abort path; snapshot progress."""
        async with self._cond:
            self._ledger.invalidated.add(generation.seq)
            presented_abs = self._presented_abs()
            snapshots: list[PlaybackProgress] = []
            for handle in self._ledger.affected(generation):
                self._apply_presented(handle, presented_abs)
                handle._resolve(interrupted=True)
                snapshots.append(handle.progress())
                self._ledger.outstanding.remove(handle)
            # Independent abort path: drop queued ring content and the device
            # buffer without waiting behind playback work; only this sink's
            # stream is touched.
            self._write_abs = self._read_abs
            self._max_dev_end = self._read_abs
            if self._stream is not None:
                with contextlib.suppress(Exception):
                    self._stream.abort()
                with contextlib.suppress(Exception):
                    self._stream.start()
                self._recalibrate()
            if self._resampler is not None:
                self._resampler.reset()
                self._resampler = None
            self._cond.notify_all()
            return tuple(snapshots)

    async def aclose(self) -> None:
        """Close the output stream and resolve leftovers (idempotent)."""
        if self._closed:
            return
        self._closed = True
        async with self._cond:
            for handle in list(self._ledger.outstanding):
                self._apply_presented(handle, self._presented_abs())
                handle._resolve(interrupted=True)
            self._ledger.outstanding.clear()
            self._max_dev_end = self._read_abs
            if self._stream is not None:
                with contextlib.suppress(Exception):
                    self._stream.abort()
                with contextlib.suppress(Exception):
                    self._stream.close()
                self._stream = None
            self._cond.notify_all()

    # -- internals ---------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._stream is not None:
            return
        rate = self._configured_rate
        if rate is None:
            dev = self._sd.query_devices(self._device, "output")
            rate = int(float(dev.get("default_samplerate", 0.0)) or 48000)
        validate_output_device(self._device, rate)
        self._rate = rate
        self._ring = np.zeros(max(1, int(self._ring_ms * rate / 1000)), dtype=np.float32)
        self._loop = asyncio.get_running_loop()
        if self._clock is None:
            self._clock = RealClock()
        self._stream = self._sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        self._recalibrate()

    def _recalibrate(self) -> None:
        """(Re)anchor the PortAudio stream clock to the injected clock."""
        assert self._clock is not None
        self._base_stream_time = float(getattr(self._stream, "time", 0.0) or 0.0)
        self._base_clock_ms = self._clock.now_ms()
        self._blocks.clear()
        self._max_presented = self._read_abs

    def _convert(self, chunk: TtsChunk) -> NDArray[np.float32]:
        """Resample one chunk to the device rate via the stateful resampler."""
        if self._resampler is None or self._resampler.in_rate != chunk.sample_rate:
            if self._resampler is not None:
                # Input-rate change: flush the old stream's tail and attribute
                # it to the previous chunk's ledger span.
                tail = self._resampler.flush()
                if len(tail) and self._ledger.outstanding:
                    prev = self._ledger.outstanding[-1]
                    self._ring_write_blocking_unsafe(tail)
                    prev._dev_end += len(tail)
                    self._max_dev_end = max(self._max_dev_end, prev._dev_end)
            self._resampler = StreamingResampler(chunk.sample_rate, self._rate)
        data = self._resampler.process(chunk.samples)
        if chunk.final:
            tail = self._resampler.flush()
            if len(tail):
                data = np.concatenate([data, tail])
            self._resampler.reset()
            self._resampler = None
        return np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)

    def _ring_write_blocking_unsafe(self, data: NDArray[np.float32]) -> None:
        """Write a small tail assuming ring space (rate-change flush only)."""
        space = len(self._ring) - (self._write_abs - self._read_abs)
        self._ring_write(data[: max(0, space)])

    def _ring_write(self, data: NDArray[np.float32]) -> None:
        n = len(data)
        if not n:
            return
        start = self._write_abs % len(self._ring)
        first = min(n, len(self._ring) - start)
        self._ring[start : start + first] = data[:first]
        if first < n:
            self._ring[: n - first] = data[first:]
        self._write_abs += n

    def _callback(
        self,
        outdata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        # PortAudio thread: never raise, never block, never replay stale memory.
        try:
            avail = self._write_abs - self._read_abs
            n = min(frames, avail)
            read_before = self._read_abs
            if n > 0:
                start = read_before % len(self._ring)
                first = min(n, len(self._ring) - start)
                outdata[:first, 0] = self._ring[start : start + first]
                if first < n:
                    outdata[first:n, 0] = self._ring[: n - first]
            if n < frames:
                outdata[n:, 0] = 0.0
                # An underrun is a starved callback while scheduled audio still
                # exists beyond what the ring could satisfy — a real zero-fill
                # gap, not the normal silence after everything drained.
                if self._max_dev_end > read_before + n:
                    self.underruns += 1
            dac = float(getattr(time_info, "outputBufferDacTime", 0.0) or 0.0)
            self._blocks.append((read_before, frames, n, dac))
            self._read_abs = read_before + n
            if status:
                self.last_status = str(status)
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._on_progress)
        except Exception:  # pragma: no cover - last-resort callback guard
            outdata.fill(0.0)

    def _stream_time_now(self) -> float:
        assert self._clock is not None
        return self._base_stream_time + (self._clock.now_ms() - self._base_clock_ms) / 1000.0

    def _presented_abs(self) -> int:
        """Device frames whose DAC presentation time has already passed."""
        presented = self._max_presented
        if self._blocks:
            t_stream = self._stream_time_now()
            found = False
            for read_before, _requested, consumed, dac in reversed(self._blocks):
                if dac <= 0.0:
                    continue
                if t_stream >= dac:
                    within = int((t_stream - dac) * self._rate)
                    presented = max(presented, read_before + min(consumed, within))
                    found = True
                    break
            if not found and all(dac <= 0.0 for *_x, dac in self._blocks):
                # Host API reports no DAC times: treat consumed as presented.
                presented = max(presented, self._read_abs)
        presented = min(presented, self._read_abs)
        self._max_presented = presented
        return presented

    def _first_audible_ms(self, dev_frame: int) -> int | None:
        """Map an absolute device frame to its calibrated clock DAC time."""
        assert self._clock is not None
        for read_before, _requested, consumed, dac in self._blocks:
            if dac <= 0.0:
                continue
            if read_before <= dev_frame < read_before + consumed:
                dac_at = dac + (dev_frame - read_before) / self._rate
                return round(self._base_clock_ms + (dac_at - self._base_stream_time) * 1000.0)
        return self._clock.now_ms()

    def _apply_presented(self, handle: _Handle, presented_abs: int) -> None:
        """Update a handle's presented counters from the device-frame ledger."""
        span = handle._dev_end - handle._dev_start
        dev_presented = max(0, min(span, presented_abs - handle._dev_start))
        handle._device_presented = dev_presented
        total = len(handle._chunk.samples)
        if span > 0:
            handle._source_presented = min(total, round(total * dev_presented / span))
        else:
            handle._source_presented = 0
        if dev_presented > 0 and handle._first_audible_t_ms is None:
            handle._first_audible_t_ms = self._first_audible_ms(handle._dev_start)

    def _on_progress(self) -> None:
        """Loop-side: resolve started/completed handles from callback progress."""
        presented_abs = self._presented_abs()
        for handle in list(self._ledger.outstanding):
            self._apply_presented(handle, presented_abs)
            if handle._device_presented > 0 and not handle._started_evt.is_set():
                handle._started_evt.set()
            if presented_abs >= handle._dev_end and not handle._completed:
                handle._completed = True
                handle._completed_evt.set()
                self._ledger.outstanding.remove(handle)
        # Wake capacity/drain waiters; ring space may have been freed.
        self._notify_waiters()

    def _notify_waiters(self) -> None:
        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        if self._loop is not None:
            task = self._loop.create_task(_notify())
            self._wake_tasks.add(task)
            task.add_done_callback(self._wake_tasks.discard)
