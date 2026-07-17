"""The interruptible streaming orchestrator: STT → stabilize → segment → MT → TTS.

This is the pipeline path of the plan. It consumes a source mic, runs streaming
STT, stabilizes the partials with LocalAgreement so only stable text flows
downstream, segments the committed stream into closed translatable units,
translates each closed segment as soon as it closes (simultaneity), synthesizes
target audio, and plays it to a sink — all concurrently, with bounded queues, on
the injected :class:`~interpret_live.clock.Clock`.

Barge-in (structured cancellation lifecycle, exact per the plan):

1. A :class:`~interpret_live.vad.BargeInDetector` watching the *source* mic
   raises an interrupt.
2. For the **current utterance** only, the pipeline cancels the in-flight MT/TTS
   task group, **discards** queued (not-yet-played) ``TtsChunk``s, and calls
   ``sink.stop()`` to abort the currently-playing chunk + flush. The moment
   ``stop()`` returns is the ``barge-in-stop`` metric endpoint.
3. STT keeps running; the new speech becomes a **new utterance** with a fresh
   stabilizer whose offsets begin after the last committed token — already-
   emitted segments are kept and never re-translated.

The per-utterance MT/TTS work runs in its own task group so barge-in cancels it
and awaits completion, leaking no tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import AsyncIterator
from typing import TypeVar

from .backends import MT, STT, TTS
from .clock import Clock
from .config import PipelineConfig
from .metrics import MetricsLog
from .segment import Segmenter
from .stabilize import LocalAgreementStabilizer
from .types import (
    AudioFrame,
    AudioSink,
    Hypothesis,
    MetricEvent,
    PlaybackGeneration,
    PlaybackHandle,
    PlaybackRejectedError,
    Segment,
    TtsChunk,
)
from .vad import BargeInDetector

__all__ = ["Pipeline"]

_T = TypeVar("_T")


class _QueuedSegment:
    """A closed segment plus the utterance id that owns it.

    The owning id is stamped at segmentation time so metric attribution
    (``first_tts_out``) and barge-in discarding stay correct even after the STT
    stage has rolled on to a later utterance while this segment is still in
    flight downstream.
    """

    __slots__ = ("context", "segment", "utterance_id")

    def __init__(self, segment: Segment, utterance_id: str, context: tuple[str, ...]) -> None:
        self.segment = segment
        self.utterance_id = utterance_id
        # Rolling source context captured now, because the segmenter is reset on
        # utterance roll and would otherwise have lost this segment's history by
        # the time it is translated.
        self.context = context


class _Interrupt:
    """An async, re-armable interrupt flag for the current utterance."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def fire(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event = asyncio.Event()

    @property
    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


class Pipeline:
    """Run one direction end-to-end: source audio → translated target audio.

    Args:
        stt: Streaming STT backend.
        mt: Machine-translation backend (called only on closed segments).
        tts: Streaming TTS backend.
        sink: Output audio sink (``play`` + ``stop``).
        clock: Injected clock.
        config: Tunable :class:`PipelineConfig`.
        barge_in: Optional barge-in detector; when provided, the pipeline
            subscribes it to the source and acts on onsets.
        metrics: Optional shared :class:`MetricsLog`; one is created if omitted.
    """

    def __init__(
        self,
        *,
        stt: STT,
        mt: MT,
        tts: TTS,
        sink: AudioSink,
        clock: Clock,
        config: PipelineConfig | None = None,
        barge_in: BargeInDetector | None = None,
        metrics: MetricsLog | None = None,
    ) -> None:
        self._stt = stt
        self._mt = mt
        self._tts = tts
        self._sink = sink
        self._clock = clock
        self._config = config or PipelineConfig()
        self._barge_in = barge_in
        self._metrics = metrics or MetricsLog()

        self._stabilizer = LocalAgreementStabilizer(n=self._config.agreement_n)
        self._segmenter = Segmenter(
            max_segment_tokens=self._config.max_segment_tokens,
            context_tokens=self._config.context_tokens,
        )
        self._interrupt = _Interrupt()
        self._utterance_count = 0
        self._utterance_id = self._new_utterance_id()
        self._first_tts_seen_for: set[str] = set()
        self._started_utterances: set[str] = set()
        # Barge-in roll requests. The supervisor (which detects barge-in) asks
        # the STT stage to roll the utterance; only the STT stage mutates the
        # stabilizer/segmenter, keeping that state single-owner and race-free.
        self._roll_requests = 0
        self._roll_done = 0
        # Utterance ids the supervisor has barged in on. A segment that was stuck
        # on a full-queue ``put`` can unblock *after* the barge-in drains the
        # queue and enqueue a stale segment; the supervisor discards any segment
        # whose owning utterance is here so that post-barge-in stale segment is
        # never translated. This is keyed on the segment's stamped owner (not the
        # live ``_utterance_id``, which the STT stage may not have rolled yet),
        # so the discard never races the STT stage's roll.
        self._abandoned_utterances: set[str] = set()
        # Owner of the most recent segment the supervisor pulled, so an idle
        # barge-in (no in-flight segment) can still mark the right utterance.
        self._last_seg_utterance_id: str | None = None
        # Playback generations: one sink generation per utterance, plus the
        # per-generation notification tasks (first-audible watchers) so a stop
        # can cancel and await exactly the affected tasks before a fresh
        # generation is issued.
        self._utt_generations: dict[str, PlaybackGeneration] = {}
        self._playback_watchers: dict[int, set[asyncio.Task[None]]] = {}
        # Upstream (STT-adapter) turn tracking: the source turn currently being
        # processed and the turns abandoned by a barge-in. A stale hypothesis is
        # discarded only when its upstream turn ID matches an interrupted turn;
        # the first hypothesis/final of the next detected turn always processes.
        self._current_source_turn: str | None = None
        self._abandoned_source_turns: set[str] = set()
        # Playback handles still presenting (pruned opportunistically); used to
        # gate barge-in on actually having target audio queued or playing.
        self._live_handles: list[PlaybackHandle] = []

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
        return f"utt-{self._utterance_count}"

    def _emit(self, kind: str, detail: dict[str, int | str] | None = None) -> None:
        self._metrics.append(
            MetricEvent(
                kind=kind,  # type: ignore[arg-type]
                t_ms=self._clock.now_ms(),
                utterance_id=self._utterance_id,
                detail=detail or {},
            )
        )

    def _ensure_utterance_started(self, hyp: Hypothesis) -> None:
        """Record ``utterance_start`` at the actual source speech onset.

        A live STT hypothesis carries ``speech_started_at_ms`` (the first
        VAD-positive frame's timestamp), so first-audio latency starts at real
        speech onset — not at first-decode arrival. Legacy fakes without the
        field keep the current-clock behavior.
        """
        if self._utterance_id not in self._started_utterances:
            self._started_utterances.add(self._utterance_id)
            t_ms = (
                hyp.speech_started_at_ms
                if hyp.speech_started_at_ms is not None
                else self._clock.now_ms()
            )
            self._metrics.append(
                MetricEvent(
                    kind="utterance_start",
                    t_ms=t_ms,
                    utterance_id=self._utterance_id,
                    detail={},
                )
            )

    async def run(self, source_frames: AsyncIterator[AudioFrame]) -> None:
        """Drive the full pipeline over ``source_frames`` until exhausted.

        ``source_frames`` is typically a fanned-out subscriber from
        :func:`interpret_live.audio_io.tee`; a second subscriber feeds the
        barge-in detector when configured. When no barge-in detector is wired,
        the source is consumed directly by STT.
        """
        # Segment queue (stabilize/segment stage → MT/TTS stage).
        seg_queue: asyncio.Queue[_QueuedSegment | None] = asyncio.Queue(
            maxsize=self._config.queue_maxsize
        )
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._stt_stage(source_frames, seg_queue), name="stt-stage")
            tg.create_task(self._mt_tts_supervisor(seg_queue), name="mt-tts-supervisor")

    async def run_with_barge_in(
        self,
        stt_frames: AsyncIterator[AudioFrame],
        barge_frames: AsyncIterator[AudioFrame],
    ) -> None:
        """Run with a separate fanned-out stream feeding the barge-in detector."""
        if self._barge_in is None:
            raise RuntimeError("run_with_barge_in requires a barge_in detector")
        seg_queue: asyncio.Queue[_QueuedSegment | None] = asyncio.Queue(
            maxsize=self._config.queue_maxsize
        )
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._stt_stage(stt_frames, seg_queue), name="stt-stage")
            tg.create_task(self._mt_tts_supervisor(seg_queue), name="mt-tts-supervisor")
            tg.create_task(self._barge_in_stage(barge_frames), name="barge-in-stage")

    # ----- Stage 1: STT → stabilize → segment ---------------------------------

    async def _stt_stage(
        self,
        source_frames: AsyncIterator[AudioFrame],
        seg_queue: asyncio.Queue[_QueuedSegment | None],
    ) -> None:
        """Consume STT hypotheses, stabilize, segment, enqueue closed segments."""
        try:
            async for hyp in self._stt.stream(source_frames):
                # Honour any barge-in roll requested by the supervisor before
                # processing the next (post-interrupt) hypothesis. Only this stage
                # mutates the stabilizer/segmenter, so the state stays race-free.
                rolled = self._apply_pending_rolls()
                if self._is_stale_hypothesis(hyp, rolled):
                    # Output that still belongs to a barged-in turn. Committing
                    # it into the fresh utterance would re-segment — and thus
                    # re-translate — text emitted before the barge-in. The roll
                    # is turn-aware: only the interrupted upstream turn's output
                    # is dropped; the next detected turn always processes, even
                    # when its very first hypothesis is already final.
                    continue
                if hyp.source_turn_id is not None:
                    self._current_source_turn = hyp.source_turn_id
                self._ensure_utterance_started(hyp)
                disagree_before = self._stabilizer.post_commit_disagreement
                result = self._stabilizer.commit(hyp)
                if self._stabilizer.post_commit_disagreement > disagree_before:
                    # A later hypothesis contradicted already-committed tokens.
                    self._emit("post_commit_disagreement")
                if result.newly_committed:
                    self._emit("commit", {"tokens": len(result.newly_committed)})
                    closed = self._segmenter.feed(result.newly_committed)
                    for seg in closed:
                        await seg_queue.put(self._enqueue_segment(seg))
                if hyp.is_final:
                    tail = self._segmenter.flush()
                    if tail is not None:
                        await seg_queue.put(self._enqueue_segment(tail))
                    # Utterance finished naturally: start a fresh utterance id so
                    # the next speech is tracked independently.
                    self._roll_utterance()
        finally:
            await seg_queue.put(None)  # signal MT/TTS supervisor to stop

    def _enqueue_segment(self, seg: Segment) -> _QueuedSegment:
        """Stamp ``seg`` with its owning utterance id + captured rolling context."""
        context = tuple(t.text for t in self._segmenter.context_for(seg))
        return _QueuedSegment(seg, self._utterance_id, context)

    def _apply_pending_rolls(self) -> bool:
        """Apply any barge-in roll requests issued by the supervisor.

        Returns ``True`` if at least one roll was applied in this call, so the
        caller can drop a stale ``is_final`` from the just-interrupted utterance.
        Each applied roll also marks the source turn being processed at the
        interrupt as abandoned, enabling turn-aware stale discarding.
        """
        applied = False
        while self._roll_done < self._roll_requests:
            if self._current_source_turn is not None:
                self._abandoned_source_turns.add(self._current_source_turn)
            self._roll_utterance()
            self._roll_done += 1
            applied = True
        return applied

    def _is_stale_hypothesis(self, hyp: Hypothesis, rolled: bool) -> bool:
        """Is ``hyp`` leftover output from a barged-in source turn?

        With upstream turn IDs the check is exact: only output whose
        ``source_turn_id`` matches an interrupted turn is stale. Legacy
        hypotheses without IDs keep the previous heuristic (drop the final
        that arrives immediately after a roll).
        """
        if hyp.source_turn_id is not None:
            return hyp.source_turn_id in self._abandoned_source_turns
        return rolled and hyp.is_final

    def _roll_utterance(self) -> None:
        """End the current utterance and reset stabilizer/segmenter offsets.

        Called only from the STT stage (on a natural ``is_final`` or to apply a
        barge-in roll request), so the stabilizer/segmenter have a single owner.
        """
        self._stabilizer.reset()
        self._segmenter.reset()
        self._utterance_id = self._new_utterance_id()

    # ----- Stage 2: MT → TTS supervisor (interruptible per utterance) ---------

    async def _mt_tts_supervisor(self, seg_queue: asyncio.Queue[_QueuedSegment | None]) -> None:
        """Pull closed segments and synthesize them, honoring barge-in.

        Each segment's MT+TTS work runs as a task raced against the interrupt so a
        barge-in can cancel exactly the in-flight work and await its completion
        (leak-free). Queued, not-yet-played segments for the interrupted utterance
        are discarded on barge-in.
        """
        # tts_queue carries synthesized chunks to the playback task.
        tts_queue: asyncio.Queue[TtsChunk | None] = asyncio.Queue(
            maxsize=self._config.queue_maxsize
        )
        play_task = asyncio.create_task(self._playback(tts_queue), name="playback")
        try:
            while True:
                qseg = await self._next_segment(seg_queue, tts_queue)
                if qseg is None:
                    break
                self._last_seg_utterance_id = qseg.utterance_id
                if qseg.utterance_id in self._abandoned_utterances:
                    # A stale segment from an utterance that was barged in on (it
                    # unblocked from a full-queue ``put`` after the drain). Drop it
                    # so it is never translated under the resumed conversation.
                    continue
                await self._synthesize_segment(qseg, tts_queue, seg_queue)
        finally:
            # Use a non-blocking enqueue so shutdown can never hang on a full
            # tts_queue (e.g. playback stalled on a slow sink at teardown).
            _enqueue_tts_sentinel(tts_queue)
            await play_task

    async def _next_segment(
        self,
        seg_queue: asyncio.Queue[_QueuedSegment | None],
        tts_queue: asyncio.Queue[TtsChunk | None],
    ) -> _QueuedSegment | None:
        """Wait for the next segment, but handle a barge-in that arrives while idle.

        If an interrupt fires while the supervisor is between segments, it is
        destructive only when there is target work to interrupt — queued
        segments/chunks or audio still presenting (e.g. trailing playback).
        Ordinary speech after silence with no active target work is a new
        turn, not an interrupt: the flag is consumed without a roll or stop.
        """
        while True:
            get_task = asyncio.create_task(seg_queue.get(), name="seg-get")
            interrupt_wait = asyncio.create_task(self._interrupt.wait(), name="interrupt-wait-idle")
            done, _pending = await asyncio.wait(
                {get_task, interrupt_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if get_task in done:
                interrupt_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await interrupt_wait
                return get_task.result()
            get_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await get_task
            if self._target_work_active(seg_queue, tts_queue):
                await self._handle_barge_in(tts_queue, seg_queue, self._last_seg_utterance_id)
            else:
                # Nothing to interrupt: a fresh utterance must not abandon
                # itself. Consume the onset and keep waiting.
                self._interrupt.clear()

    def _target_work_active(
        self,
        seg_queue: asyncio.Queue[_QueuedSegment | None],
        tts_queue: asyncio.Queue[TtsChunk | None],
    ) -> bool:
        """Is any target-language work queued, in flight, or still audible?"""
        if seg_queue.qsize() or tts_queue.qsize():
            return True
        self._live_handles = [h for h in self._live_handles if not h.progress().completed]
        return bool(self._live_handles)

    async def _synthesize_segment(
        self,
        qseg: _QueuedSegment,
        tts_queue: asyncio.Queue[TtsChunk | None],
        seg_queue: asyncio.Queue[_QueuedSegment | None],
    ) -> None:
        """Translate + synthesize one segment, abortable by barge-in."""
        interrupt_wait = asyncio.create_task(self._interrupt.wait(), name="interrupt-wait")
        work = asyncio.create_task(
            self._translate_and_synthesize(qseg, tts_queue),
            name=f"seg-{qseg.segment.index}",
        )
        done, _pending = await asyncio.wait(
            {work, interrupt_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if interrupt_wait in done and not work.done():
            # Barge-in won the race: cancel in-flight work and await it.
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
            await self._handle_barge_in(tts_queue, seg_queue, qseg.utterance_id)
        else:
            # Work finished. It may have completed on the *same* tick the
            # interrupt fired, in which case the interrupt event is still set with
            # nothing in flight to cancel; consume it here so it does not trigger
            # a spurious barge-in (extra roll + discarding this just-finished
            # segment's audio) on the next segment. This segment's audio is kept.
            interrupt_fired = interrupt_wait in done
            interrupt_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await interrupt_wait
            # Surface any error from the work task.
            await work
            if interrupt_fired and self._interrupt.is_set:
                self._interrupt.clear()

    async def _translate_and_synthesize(
        self,
        qseg: _QueuedSegment,
        tts_queue: asyncio.Queue[TtsChunk | None],
    ) -> None:
        seg = qseg.segment
        target_text = await self._mt.translate(seg, qseg.context)
        async for chunk in self._tts.synthesize(
            target_text,
            segment_index=seg.index,
            utterance_id=qseg.utterance_id,
        ):
            await tts_queue.put(chunk)

    async def _handle_barge_in(
        self,
        tts_queue: asyncio.Queue[TtsChunk | None],
        seg_queue: asyncio.Queue[_QueuedSegment | None],
        interrupted_utterance_id: str | None,
    ) -> None:
        """Discard queued chunks/segments, stop the sink, request a fresh utterance.

        Runs on the supervisor task. It performs the *audio-side* teardown and
        metric emission, then asks the STT stage (the sole owner of the
        stabilizer/segmenter) to roll to a new utterance — so already-emitted
        segments are kept and never re-translated, and the new speech is
        stabilized fresh from the next committed token.

        ``interrupted_utterance_id`` is the owner of the work being interrupted;
        it is marked abandoned so a segment that unblocks from a full-queue
        ``put`` *after* this drain is discarded rather than translated.
        """
        self._emit("interrupt")
        if interrupted_utterance_id is not None:
            self._abandoned_utterances.add(interrupted_utterance_id)
            # Ensure the interrupted utterance owns a generation even when no
            # chunk reached the sink yet: invalidating it now guarantees a late
            # chunk can never be scheduled under the interrupted turn.
            self._generation_for(interrupted_utterance_id, create=True)
        # 1) Discard queued, not-yet-played chunks for the interrupted utterance.
        _drain(tts_queue)
        # 2) Discard queued, not-yet-synthesized segments for this utterance.
        _drain(seg_queue)
        # 3) Stop every live playback generation (all pre-interrupt audio):
        #    the sink invalidates each generation under its lock first, then we
        #    cancel and await that generation's pending notification tasks so
        #    nothing stale survives into the next generation. The final stop()
        #    returning is the barge-in-stop metric endpoint.
        for uid, gen in list(self._utt_generations.items()):
            await self._sink.stop(gen)
            await self._settle_watchers(gen.seq)
            del self._utt_generations[uid]
        self._emit("sink_stopped")
        # 4) Ask the STT stage to start a NEW utterance with a fresh stabilizer.
        self._roll_requests += 1
        self._interrupt.clear()

    def _generation_for(
        self, utterance_id: str, *, create: bool = True
    ) -> PlaybackGeneration | None:
        """Return (creating if needed) the sink generation owning ``utterance_id``."""
        gen = self._utt_generations.get(utterance_id)
        if gen is None and create:
            gen = self._sink.new_generation()
            self._utt_generations[utterance_id] = gen
        return gen

    async def _settle_watchers(self, gen_seq: int) -> None:
        """Cancel and await the notification tasks of one stopped generation."""
        for task in self._playback_watchers.pop(gen_seq, set()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ----- Stage 3: playback ---------------------------------------------------

    async def _playback(self, tts_queue: asyncio.Queue[TtsChunk | None]) -> None:
        """Schedule synthesized chunks with bounded lookahead; drain at EOF.

        ``schedule()`` waits only for bounded sink capacity, so the next chunk
        is buffered before the current one finishes (gapless playback). The
        ``first_tts_out`` metric is emitted from the first handle's *started*
        receipt — first audible target audio, not first chunk received.
        """
        while True:
            chunk = await tts_queue.get()
            if chunk is None:
                break
            if chunk.utterance_id in self._abandoned_utterances:
                continue  # stale chunk from a barged-in utterance
            generation = self._generation_for(chunk.utterance_id)
            assert generation is not None
            try:
                handle = await self._sink.schedule(generation, chunk)
            except PlaybackRejectedError:
                continue  # the generation was stopped while we were blocked
            self._live_handles = [h for h in self._live_handles if not h.progress().completed]
            self._live_handles.append(handle)
            if chunk.utterance_id not in self._first_tts_seen_for:
                self._first_tts_seen_for.add(chunk.utterance_id)
                self._spawn_started_watcher(handle, chunk.utterance_id)
        # Normal EOF: present everything scheduled, then settle notifications.
        await self._sink.drain()
        for gen_seq in list(self._playback_watchers):
            await self._settle_watchers(gen_seq)

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
                    detail={"segment": handle.chunk.segment_index},
                )
            )

        task = asyncio.create_task(_watch(), name=f"first-tts-watch-{utterance_id}")
        self._playback_watchers.setdefault(handle.generation.seq, set()).add(task)
        task.add_done_callback(
            functools.partial(_discard_watcher, self._playback_watchers, handle.generation.seq)
        )

    # ----- Barge-in detection --------------------------------------------------

    async def _barge_in_stage(self, barge_frames: AsyncIterator[AudioFrame]) -> None:
        """Watch the fanned-out source and fire the interrupt on a debounced onset."""
        assert self._barge_in is not None

        async def _on_onset(_frame: AudioFrame) -> None:
            self._interrupt.fire()

        await self._barge_in.watch(barge_frames, _on_onset)


def _enqueue_tts_sentinel(tts_queue: asyncio.Queue[TtsChunk | None]) -> None:
    """Enqueue the playback stop-sentinel without ever blocking.

    At shutdown the queue may be full (e.g. playback stalled on a slow sink); a
    blocking ``await put(None)`` could then hang forever. Drop the oldest queued
    chunk to make room if needed, so the ``None`` sentinel that stops the
    playback task is always delivered and teardown cannot deadlock.
    """
    while tts_queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            tts_queue.get_nowait()
    tts_queue.put_nowait(None)


def _discard_watcher(
    watchers: dict[int, set[asyncio.Task[None]]], seq: int, task: asyncio.Task[None]
) -> None:
    """Done-callback: drop a finished notification task from its registry."""
    watchers.get(seq, set()).discard(task)


def _drain(queue: asyncio.Queue[_T | None]) -> int:
    """Remove all non-``None`` items from ``queue``; preserve ``None`` sentinels.

    Returns the number of payload items discarded. ``None`` sentinels (the
    end-of-stream markers) are re-enqueued so a barge-in that races a natural
    end-of-stream does not strand a downstream consumer.
    """
    discarded = 0
    sentinels = 0
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item is None:
            sentinels += 1
        else:
            discarded += 1
    for _ in range(sentinels):
        queue.put_nowait(None)
    return discarded
