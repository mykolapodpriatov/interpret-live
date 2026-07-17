"""The unified speech-to-speech path (cloud realtime) behind one Session.

On this path a single :class:`~interpret_live.backends.S2S` provider does
STT+MT+TTS internally over one session-long connection; the harness does
**not** see ASR partials, so the LocalAgreement audio-stage stabilizer is
honestly **bypassed** (documented in the capability matrix).

Turn/response ownership (plan Task 6):

* The provider's *speech started* control event creates each local utterance
  and stamps ``utterance_start`` at the provider-mapped source onset. Local
  :class:`~interpret_live.vad.EnergyVAD` is authoritative only for the
  *immediate* barge-in reaction; its segmentation never has to match the
  provider VAD's, and a local onset never rolls a speculative turn.
* Explicit provider input-item/response-ID -> local-utterance maps attribute
  audio, metrics, and interrupts even when responses outlive their turn.
* On local barge-in the pipeline atomically abandons exactly the snapshotted
  response, stops its playback generation through the sink's independent
  abort path, builds a :class:`~interpret_live.types.PlaybackCursor` from
  completed receipts plus the stop snapshot (never queued/device-buffered
  samples), sends exactly one provider cancel/truncate, and continues the
  same provider session so later speech produces audio.
* Only ``response.done(status=completed)`` is a natural completion. An
  expected ``cancelled`` for an abandoned response is an interrupt
  acknowledgement; ``failed``/unexpected ``cancelled``/``incomplete`` surface
  as typed :class:`~interpret_live.types.S2SResponseError`.
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
    PlaybackCursor,
    PlaybackGeneration,
    PlaybackHandle,
    PlaybackProgress,
    PlaybackRejectedError,
    S2SAudioChunk,
    S2SContentDone,
    S2SEvent,
    S2SInterruptTarget,
    S2SProtocolError,
    S2SResponseDone,
    S2SResponseError,
    S2SResponseStarted,
    S2SSpeechCommitted,
    S2SSpeechStarted,
    TtsChunk,
)
from .vad import BargeInDetector

__all__ = ["S2SPipeline"]

_NATURAL = "completed"


class _ResponseState:
    """Playback bookkeeping for one provider response."""

    __slots__ = (
        "generation",
        "handles",
        "heard_ms_by_content",
        "input_item_id",
        "last_content",
        "response_id",
        "utterance_id",
    )

    def __init__(
        self,
        response_id: str,
        utterance_id: str,
        input_item_id: str | None,
        generation: PlaybackGeneration,
    ) -> None:
        self.response_id = response_id
        self.utterance_id = utterance_id
        self.input_item_id = input_item_id
        self.generation = generation
        # Live playback handles with their provenance: (handle, item, content).
        self.handles: list[tuple[PlaybackHandle, str, int]] = []
        # Heard (presented) source-rate milliseconds per (item_id, content_index).
        self.heard_ms_by_content: dict[tuple[str, int], float] = {}
        # The most recent (item_id, content_index) that carried audio.
        self.last_content: tuple[str, int] | None = None

    def absorb_receipt(self, item_id: str, content_index: int, receipt: PlaybackProgress) -> None:
        if receipt.source_rate > 0:
            heard = 1000.0 * receipt.source_samples_presented / receipt.source_rate
            key = (item_id, content_index)
            self.heard_ms_by_content[key] = self.heard_ms_by_content.get(key, 0.0) + heard

    def reap_completed(self) -> None:
        """Fold finished handles' receipts into the heard ledger."""
        remaining: list[tuple[PlaybackHandle, str, int]] = []
        for handle, item_id, content_index in self.handles:
            progress = handle.progress()
            if progress.completed:
                self.absorb_receipt(item_id, content_index, progress)
            else:
                remaining.append((handle, item_id, content_index))
        self.handles = remaining

    def live_handles(self) -> bool:
        return any(not handle.progress().completed for handle, _i, _c in self.handles)

    def cursor(self) -> PlaybackCursor | None:
        """The heard-audio cursor, or ``None`` if nothing was ever audible."""
        if self.last_content is None:
            return None
        item_id, content_index = self.last_content
        heard = self.heard_ms_by_content.get((item_id, content_index), 0.0)
        if heard <= 0.0:
            return None
        return PlaybackCursor(
            response_id=self.response_id,
            item_id=item_id,
            content_index=content_index,
            audio_end_ms=round(heard),
        )


class S2SPipeline:
    """Drive source audio through a persistent :class:`S2S` provider session.

    Args:
        s2s: The unified speech-to-speech provider.
        sink: Output audio sink (generation-scoped playback contract).
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
        self._interrupt = asyncio.Event()
        self._first_seen: set[str] = set()
        # Provider ownership maps.
        self._utt_by_input: dict[str, str] = {}
        self._responses: dict[str, _ResponseState] = {}
        self._abandoned: set[str] = set()
        self._committed_inputs: set[str] = set()
        self._current_response: str | None = None
        self._latest_utterance: str | None = None
        self._response_order: list[str] = []
        # One pipeline state lock protects response/playback ownership.
        self._state_lock = asyncio.Lock()
        self._playback_watchers: dict[int, set[asyncio.Task[None]]] = {}

    @property
    def metrics(self) -> MetricsLog:
        """The metrics log this pipeline appends to."""
        return self._metrics

    def _new_utterance_id(self) -> str:
        self._utterance_count += 1
        return f"s2s-{self._utterance_count}"

    def _emit(self, kind: str, uid: str, t_ms: int | None = None) -> None:
        self._metrics.append(
            MetricEvent(
                kind=kind,  # type: ignore[arg-type]
                t_ms=self._clock.now_ms() if t_ms is None else t_ms,
                utterance_id=uid,
                detail={},
            )
        )

    async def run(self, source_frames: AsyncIterator[AudioFrame]) -> None:
        """Run the persistent provider session over ``source_frames``."""
        await self._pump(source_frames)

    async def run_with_barge_in(
        self,
        s2s_frames: AsyncIterator[AudioFrame],
        barge_frames: AsyncIterator[AudioFrame],
    ) -> None:
        """Run with a fanned-out stream feeding the barge-in detector."""
        if self._barge_in is None:
            raise RuntimeError("run_with_barge_in requires a barge_in detector")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._pump(s2s_frames), name="s2s-pump")
            tg.create_task(self._barge_in_stage(barge_frames), name="s2s-barge-in")

    # ----- receive/schedule split ---------------------------------------------

    async def _pump(self, source_frames: AsyncIterator[AudioFrame]) -> None:
        """Receiver fills a bounded event queue; the scheduler consumes it."""
        event_q: asyncio.Queue[S2SEvent | None | BaseException] = asyncio.Queue(
            maxsize=self._config.queue_maxsize
        )

        async def _receive() -> None:
            try:
                async for event in self._s2s.stream(source_frames):
                    await event_q.put(event)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                await event_q.put(exc)
                return
            await event_q.put(None)

        receiver = asyncio.create_task(_receive(), name="s2s-receiver")
        try:
            await self._schedule_events(event_q)
        finally:
            if not receiver.done():
                receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
            for gen_seq in list(self._playback_watchers):
                await self._settle_watchers(gen_seq)

    async def _schedule_events(
        self, event_q: asyncio.Queue[S2SEvent | None | BaseException]
    ) -> None:
        while True:
            get_task = asyncio.create_task(event_q.get(), name="s2s-event-get")
            interrupt_wait = asyncio.create_task(self._interrupt.wait(), name="s2s-int")
            done, _pending = await asyncio.wait(
                {get_task, interrupt_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if interrupt_wait in done and get_task not in done:
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
                await self._handle_barge_in()
                continue
            interrupt_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await interrupt_wait
            event = get_task.result()
            if event is None:
                await self._drain_observing_interrupts()
                return
            if isinstance(event, BaseException):
                raise event
            await self._dispatch_event(event)

    async def _drain_observing_interrupts(self) -> None:
        """Drain scheduled audio while barge-in stays observable.

        The provider stream may end while audio is still presenting (trailing
        playback); a barge-in during that window must still stop the sink and
        cancel/truncate the finished response.
        """
        while True:
            drain = asyncio.create_task(self._sink.drain(), name="s2s-drain")
            interrupt_wait = asyncio.create_task(self._interrupt.wait(), name="s2s-int-drain")
            done, _pending = await asyncio.wait(
                {drain, interrupt_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if interrupt_wait in done and drain not in done:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain
                await self._handle_barge_in()
                continue
            interrupt_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await interrupt_wait
            await drain  # surface unexpected drain errors
            return

    async def _dispatch_event(self, event: S2SEvent) -> None:
        if isinstance(event, S2SSpeechStarted):
            self._on_speech_started(event)
        elif isinstance(event, S2SSpeechCommitted):
            self._on_speech_committed(event)
        elif isinstance(event, S2SResponseStarted):
            self._on_response_started(event)
        elif isinstance(event, S2SAudioChunk):
            await self._on_audio(event)
        elif isinstance(event, S2SContentDone):
            self._on_content_done(event)
        elif isinstance(event, S2SResponseDone):
            self._on_response_done(event)

    # ----- provider lifecycle events -------------------------------------------

    def _on_speech_started(self, event: S2SSpeechStarted) -> None:
        if event.input_item_id in self._utt_by_input:
            return  # duplicate provider event: idempotent
        uid = self._new_utterance_id()
        self._utt_by_input[event.input_item_id] = uid
        self._latest_utterance = uid
        # The provider's mapped source onset creates the local utterance —
        # exactly once, so local/provider VAD disagreement cannot double-roll.
        self._emit("utterance_start", uid, t_ms=event.source_started_at_ms)

    def _on_speech_committed(self, event: S2SSpeechCommitted) -> None:
        if event.input_item_id not in self._utt_by_input:
            raise S2SProtocolError(
                f"speech committed for unknown input item {event.input_item_id!r}"
            )
        self._committed_inputs.add(event.input_item_id)  # duplicates: idempotent

    def _on_response_started(self, event: S2SResponseStarted) -> None:
        if event.response_id in self._responses or event.response_id in self._abandoned:
            return  # duplicate provider event: idempotent
        if event.input_item_id is not None:
            uid = self._utt_by_input.get(event.input_item_id)
            if uid is None:
                raise S2SProtocolError(
                    f"response {event.response_id!r} references unknown input "
                    f"item {event.input_item_id!r}"
                )
        else:
            uid = self._latest_utterance
            if uid is None:
                raise S2SProtocolError(
                    f"response {event.response_id!r} started before any input turn"
                )
        state = _ResponseState(
            event.response_id, uid, event.input_item_id, self._sink.new_generation()
        )
        self._responses[event.response_id] = state
        self._response_order.append(event.response_id)
        self._current_response = event.response_id

    async def _on_audio(self, event: S2SAudioChunk) -> None:
        if event.response_id in self._abandoned:
            return  # late delta for a cancelled response: discarded
        state = self._responses.get(event.response_id)
        if state is None:
            raise S2SProtocolError(f"audio for unknown response {event.response_id!r}")
        chunk = TtsChunk(
            samples=event.samples,
            sample_rate=event.sample_rate,
            segment_index=event.content_index,
            utterance_id=state.utterance_id,
            final=event.final,
        )
        # Schedule raced against the interrupt: a barge-in must stay observable
        # while we are blocked on sink capacity, not only between events.
        schedule = asyncio.create_task(
            self._sink.schedule(state.generation, chunk), name="s2s-schedule"
        )
        interrupt_wait = asyncio.create_task(self._interrupt.wait(), name="s2s-int-sched")
        done, _pending = await asyncio.wait(
            {schedule, interrupt_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if interrupt_wait in done and schedule not in done:
            await self._handle_barge_in()
            with contextlib.suppress(PlaybackRejectedError, asyncio.CancelledError):
                await schedule
            return
        interrupt_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await interrupt_wait
        try:
            handle = schedule.result()
        except PlaybackRejectedError:
            return  # the generation was stopped while we were blocked
        state.reap_completed()  # bounded outstanding handles, in-order receipts
        state.handles.append((handle, event.item_id, event.content_index))
        state.last_content = (event.item_id, event.content_index)
        if state.utterance_id not in self._first_seen:
            self._first_seen.add(state.utterance_id)
            self._spawn_started_watcher(handle, state.utterance_id)

    def _on_content_done(self, event: S2SContentDone) -> None:
        # Content-audio done closes only that content stream; the response (and
        # its queued/playing audio) lives on until response.done.
        if event.response_id in self._abandoned:
            return
        if event.response_id not in self._responses:
            raise S2SProtocolError(f"content done for unknown response {event.response_id!r}")

    def _on_response_done(self, event: S2SResponseDone) -> None:
        if event.response_id in self._abandoned:
            # The interrupt acknowledgement (expected `cancelled`) — or a
            # completed/failed race that lost to our cancel. Never a fresh
            # local turn, never a second roll.
            self._abandoned.discard(event.response_id)
            self._responses.pop(event.response_id, None)
            return
        state = self._responses.get(event.response_id)
        if state is None:
            raise S2SProtocolError(f"done for unknown response {event.response_id!r}")
        if self._current_response == event.response_id:
            self._current_response = None
        if event.status == _NATURAL:
            # Natural completion closes generation state without losing the
            # queued/playing chunks' ownership: handles keep presenting and the
            # state stays until its audio is reaped or the session ends.
            return
        raise S2SResponseError(event.response_id, event.status, event.reason)

    # ----- barge-in ----------------------------------------------------------------

    async def _handle_barge_in(self) -> None:
        """Cancel exactly the snapshotted response; keep the session alive."""
        async with self._state_lock:
            target = self._interrupt_target_locked()
            if target is None:
                # Ordinary speech with no in-progress response and no queued or
                # playing output: a new provider turn, not an interrupt.
                self._interrupt = asyncio.Event()
                return
            state = self._responses[target]
            # Atomic (no awaits yet): mark abandoned and snapshot identity so a
            # newer response starting during cleanup can never be cancelled.
            self._abandoned.add(target)
            if self._current_response == target:
                self._current_response = None
        self._emit("interrupt", state.utterance_id)
        # Stop playback through the sink's independent abort path: the
        # generation is invalidated before capacity is released, and the
        # snapshots freeze each handle at its DAC-passed position.
        await self._sink.stop(state.generation)
        # Fold every handle's final position (completed receipts plus the
        # atomic stop snapshot) into the per-item/content heard ledger.
        for handle, item_id, content_index in state.handles:
            state.absorb_receipt(item_id, content_index, handle.progress())
        state.handles = []
        await self._settle_watchers(state.generation.seq)
        cursor = state.cursor()
        # Exactly one provider-side cancellation/truncation for the snapshotted
        # response — even if it finished or a newer response started meanwhile.
        await self._s2s.interrupt(S2SInterruptTarget(response_id=target, cursor=cursor))
        self._emit("sink_stopped", state.utterance_id)
        # Re-arm; do NOT roll a speculative local turn: the provider's next
        # speech-start event creates the fresh utterance exactly once.
        self._interrupt = asyncio.Event()

    def _interrupt_target_locked(self) -> str | None:
        """Pick the response a barge-in should cancel (or ``None`` to gate)."""
        if self._current_response is not None:
            return self._current_response
        # No in-progress response: the interrupt is destructive only when
        # audio is still queued or playing (e.g. trailing playback after a
        # natural completion).
        for response_id in reversed(self._response_order):
            state = self._responses.get(response_id)
            if state is None or response_id in self._abandoned:
                continue
            state.reap_completed()
            if state.live_handles():
                return response_id
        return None

    # ----- notification plumbing -----------------------------------------------

    def _spawn_started_watcher(self, handle: PlaybackHandle, utterance_id: str) -> None:
        """Emit ``first_tts_out`` from the handle's started receipt (async)."""

        async def _watch() -> None:
            receipt = await handle.started()
            if receipt.first_audible_t_ms is None:
                return  # stopped before ever becoming audible
            self._emit("first_tts_out", utterance_id, t_ms=receipt.first_audible_t_ms)

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
