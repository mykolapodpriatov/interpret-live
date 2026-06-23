"""Audio edges: mic fan-out (``tee``), deterministic fakes, and real I/O.

The fakes (:class:`FakeAudioSource`, :class:`FakeAudioSink`) and the
:func:`tee` / :class:`Broadcaster` fan-out are pure-stdlib + numpy and always
importable. The real microphone/speaker streams (:class:`MicSource`,
:class:`SpeakerSink`) require the ``audio`` extra (sounddevice) and are
import-guarded so a missing dependency yields a clear install hint.

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
from collections.abc import AsyncIterator

import numpy as np
from numpy.typing import NDArray

from .clock import Clock
from .types import AudioFrame, AudioSink, AudioSource, TtsChunk

_logger = logging.getLogger(__name__)

__all__ = [
    "Broadcaster",
    "FakeAudioSink",
    "FakeAudioSource",
    "MicSource",
    "SpeakerSink",
    "tee",
]


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


class FakeAudioSink(AudioSink):
    """Record played chunks and support deterministic :meth:`stop`.

    Tracks every chunk handed to :meth:`play` and the clock time of the first
    :meth:`stop` call, which is the ``barge-in-stop`` metric endpoint.

    Args:
        clock: Injected clock used to timestamp :meth:`stop`.
    """

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock
        #: Every chunk passed to :meth:`play`, in order.
        self.played: list[TtsChunk] = []
        #: Number of times :meth:`stop` was called.
        self.stop_count = 0
        #: Clock time (ms) of the first :meth:`stop`, or ``None``.
        self.stopped_at_ms: int | None = None
        self._stopped = False

    async def play(self, chunk: TtsChunk) -> None:
        """Record ``chunk`` as played (no-op audio)."""
        if self._stopped:
            # After a stop, a fresh utterance's chunks resume normally.
            self._stopped = False
        self.played.append(chunk)

    async def stop(self) -> None:
        """Abort the current chunk and discard queued audio (records the time)."""
        self._stopped = True
        self.stop_count += 1
        if self.stopped_at_ms is None and self._clock is not None:
            self.stopped_at_ms = self._clock.now_ms()

    @property
    def played_utterance_ids(self) -> list[str]:
        """The utterance id of each played chunk, in order."""
        return [c.utterance_id for c in self.played]

    def concatenated(self) -> NDArray[np.float32]:
        """All played samples concatenated (for whole-output assertions)."""
        if not self.played:
            return np.empty(0, dtype=np.float32)
        return np.concatenate([c.samples for c in self.played])


class MicSource(AudioSource):  # pragma: no cover - requires the 'audio' extra
    """Real microphone source backed by sounddevice (``audio`` extra)."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        device: int | None = None,
    ) -> None:
        from .backends.guard import require

        self._sd = require("sounddevice", backend="audio", extra="audio")
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._device = device

    async def _frames(self) -> AsyncIterator[AudioFrame]:
        blocksize = int(self._frame_ms * self._sample_rate / 1000)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[NDArray[np.float32]] = asyncio.Queue(maxsize=16)

        def _callback(
            indata: NDArray[np.float32],
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

        t_ms = 0
        with self._sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=blocksize,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=_callback,
        ):
            while True:
                block = await queue.get()
                yield AudioFrame(samples=block, sample_rate=self._sample_rate, t_ms=t_ms)
                t_ms += self._frame_ms

    def frames(self) -> AsyncIterator[AudioFrame]:
        """Return the live microphone frame iterator."""
        return self._frames()


class SpeakerSink(AudioSink):  # pragma: no cover - requires the 'audio' extra
    """Real speaker sink backed by sounddevice (``audio`` extra)."""

    def __init__(self, *, sample_rate: int = 16000, device: int | None = None) -> None:
        from .backends.guard import require

        self._sd = require("sounddevice", backend="audio", extra="audio")
        self._sample_rate = sample_rate
        self._device = device

    async def play(self, chunk: TtsChunk) -> None:
        """Play one chunk, blocking until it finishes (off the event loop)."""
        await asyncio.get_event_loop().run_in_executor(None, self._play_blocking, chunk)

    def _play_blocking(self, chunk: TtsChunk) -> None:
        self._sd.play(chunk.samples, samplerate=chunk.sample_rate, device=self._device)
        self._sd.wait()

    async def stop(self) -> None:
        """Stop playback immediately."""
        await asyncio.get_event_loop().run_in_executor(None, self._sd.stop)
