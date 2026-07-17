"""The unified speech-to-speech path (cloud realtime) behind one Session.

On this path a single :class:`~interpret_live.backends.S2S` provider does
STT+MT+TTS internally; the harness does **not** see ASR partials, so the
LocalAgreement audio-stage stabilizer is honestly **bypassed** (it is the
provider's responsibility — documented in the capability matrix). The harness
still detects a barge-in onset on the *source* mic and sends the provider's
interrupt/cancel, and it still records latency / barge-in-stop metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import AsyncIterator

from .backends import S2S
from .clock import Clock
from .config import PipelineConfig
from .metrics import MetricsLog
from .types import (
    AudioFrame,
    AudioSink,
    MetricEvent,
    PlaybackGeneration,
    PlaybackHandle,
    PlaybackRejectedError,
    TtsChunk,
)
from .vad import BargeInDetector

__all__ = ["S2SPipeline"]


class S2SPipeline:
    """Drive source audio through an :class:`S2S` provider to the output sink.

    Args:
        s2s: The unified speech-to-speech provider.
        sink: Output audio sink.
        clock: Injected clock.
        config: Tunable config (queue bounds / VAD knobs reused).
        barge_in: Optional barge-in detector watching the source mic.
        metrics: Optional shared metrics log.
    """

    def __init__(
        self,
        *,
        s2s: S2S,
        sink: AudioSink,
        clock: Clock,
        config: PipelineConfig | None = None,
        barge_in: BargeInDetector | None = None,
        metrics: MetricsLog | None = None,
    ) -> None:
        self._s2s = s2s
        self._sink = sink
        self._clock = clock
        self._config = config or PipelineConfig()
        self._barge_in = barge_in
        self._metrics = metrics or MetricsLog()
        self._utterance_count = 0
        self._utterance_id = self._new_utterance_id()
        self._interrupt = asyncio.Event()
        self._first_seen: set[str] = set()
        self._utt_generations: dict[str, PlaybackGeneration] = {}
        self._playback_watchers: dict[int, set[asyncio.Task[None]]] = {}

    @property
    def metrics(self) -> MetricsLog:
        """The metrics log this pipeline appends to."""
        return self._metrics

    @property
    def utterance_id(self) -> str:
        """The current utterance id."""
        return self._utterance_id

    def _new_utterance_id(self) -> str:
        self._utterance_count += 1
        return f"s2s-{self._utterance_count}"

    def _emit(self, kind: str, uid: str | None = None) -> None:
        self._metrics.append(
            MetricEvent(
                kind=kind,  # type: ignore[arg-type]
                t_ms=self._clock.now_ms(),
                utterance_id=uid or self._utterance_id,
                detail={},
            )
        )

    async def run(self, source_frames: AsyncIterator[AudioFrame]) -> None:
        """Run the S2S provider over ``source_frames`` (no barge-in wired)."""
        self._emit("utterance_start")
        await self._pump(source_frames)

    async def run_with_barge_in(
        self,
        s2s_frames: AsyncIterator[AudioFrame],
        barge_frames: AsyncIterator[AudioFrame],
    ) -> None:
        """Run with a fanned-out stream feeding the barge-in detector."""
        if self._barge_in is None:
            raise RuntimeError("run_with_barge_in requires a barge_in detector")
        self._emit("utterance_start")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._pump(s2s_frames), name="s2s-pump")
            tg.create_task(self._barge_in_stage(barge_frames), name="s2s-barge-in")

    async def _pump(self, source_frames: AsyncIterator[AudioFrame]) -> None:
        """Stream provider chunks to the sink, racing against the interrupt."""
        stream = self._s2s.stream(source_frames, utterance_id=self._utterance_id)
        stream_iter = aiter(stream)
        while True:
            nxt = asyncio.create_task(_anext_or_none(stream_iter), name="s2s-next")
            interrupt_wait = asyncio.create_task(self._interrupt.wait(), name="s2s-int")
            done, _pending = await asyncio.wait(
                {nxt, interrupt_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if interrupt_wait in done and not nxt.done():
                nxt.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await nxt
                await self._handle_barge_in()
                return
            interrupt_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await interrupt_wait
            chunk = nxt.result()
            if chunk is None:
                # Normal EOF: present everything scheduled, settle watchers.
                await self._sink.drain()
                for gen_seq in list(self._playback_watchers):
                    await self._settle_watchers(gen_seq)
                return
            await self._play(chunk)

    async def _play(self, chunk: TtsChunk) -> None:
        """Schedule one provider chunk under its utterance's generation."""
        generation = self._generation_for(chunk.utterance_id)
        try:
            handle = await self._sink.schedule(generation, chunk)
        except PlaybackRejectedError:
            return  # the generation was stopped while we were blocked
        if chunk.utterance_id not in self._first_seen:
            self._first_seen.add(chunk.utterance_id)
            self._spawn_started_watcher(handle, chunk.utterance_id)

    def _generation_for(self, utterance_id: str) -> PlaybackGeneration:
        gen = self._utt_generations.get(utterance_id)
        if gen is None:
            gen = self._sink.new_generation()
            self._utt_generations[utterance_id] = gen
        return gen

    def _spawn_started_watcher(self, handle: PlaybackHandle, utterance_id: str) -> None:
        """Emit ``first_tts_out`` from the handle's started receipt (async)."""

        async def _watch() -> None:
            receipt = await handle.started()
            if receipt.first_audible_t_ms is None:
                return  # stopped before ever becoming audible
            self._metrics.append(
                MetricEvent(
                    kind="first_tts_out",
                    t_ms=receipt.first_audible_t_ms,
                    utterance_id=utterance_id,
                    detail={},
                )
            )

        task = asyncio.create_task(_watch(), name=f"s2s-first-tts-watch-{utterance_id}")
        self._playback_watchers.setdefault(handle.generation.seq, set()).add(task)
        task.add_done_callback(
            functools.partial(_discard_watcher, self._playback_watchers, handle.generation.seq)
        )

    async def _settle_watchers(self, gen_seq: int) -> None:
        """Cancel and await the notification tasks of one generation."""
        for task in self._playback_watchers.pop(gen_seq, set()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _handle_barge_in(self) -> None:
        """Send the provider interrupt, stop playback generations, record metrics."""
        self._emit("interrupt")
        await self._s2s.interrupt()
        # Ensure the current utterance owns a generation even if no chunk was
        # scheduled yet, so a late chunk cannot be scheduled under it.
        self._generation_for(self._utterance_id)
        for uid, gen in list(self._utt_generations.items()):
            await self._sink.stop(gen)
            await self._settle_watchers(gen.seq)
            del self._utt_generations[uid]
        self._emit("sink_stopped")

    async def _barge_in_stage(self, barge_frames: AsyncIterator[AudioFrame]) -> None:
        assert self._barge_in is not None

        async def _on_onset(_frame: AudioFrame) -> None:
            self._interrupt.set()

        await self._barge_in.watch(barge_frames, _on_onset)


def _discard_watcher(
    watchers: dict[int, set[asyncio.Task[None]]], seq: int, task: asyncio.Task[None]
) -> None:
    """Done-callback: drop a finished notification task from its registry."""
    watchers.get(seq, set()).discard(task)


async def _anext_or_none(it: AsyncIterator[TtsChunk]) -> TtsChunk | None:
    """Return the next chunk, or ``None`` at end-of-stream."""
    try:
        return await anext(it)
    except StopAsyncIteration:
        return None
