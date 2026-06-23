"""End-to-end pipeline tests (asyncio, manual clock, drain-then-advance).

Proves the marquee behaviours deterministically:

* **Simultaneity** — first target audio is emitted *before* the source
  utterance's final hypothesis.
* **Audio-stage stability** — a mid-word ASR revision causes ZERO TTS
  retraction (the wrong guess never reaches MT/TTS).
* **Barge-in** — cancels in-flight MT/TTS promptly, discards queued chunks,
  calls ``sink.stop()``, leaks no task, and starts a NEW utterance whose
  stabilizer offsets begin after the last committed token (already-emitted
  segments are not re-translated).
* **Backpressure** — bounded queues do not deadlock under the drain harness.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import numpy as np

from helpers import frame, hyp, make_tokens
from interpret_live.audio_io import FakeAudioSink, FakeAudioSource, tee
from interpret_live.backends.fake import FakeMT, FakeSTT, FakeTTS
from interpret_live.clock import ManualClock, drain_then_advance
from interpret_live.config import PipelineConfig
from interpret_live.pipeline import Pipeline, _enqueue_tts_sentinel, _QueuedSegment
from interpret_live.types import AudioFrame, Hypothesis, MetricEvent, Segment, TtsChunk
from interpret_live.vad import BargeInDetector, EnergyVAD


class _GatedSTT:
    """A scripted STT that parks before a chosen partial until an event is set.

    Used to reproduce the is-final/barge-in race deterministically: the test can
    fire a barge-in while the interrupted utterance's ``is_final`` hypothesis is
    held just before being emitted, so the roll is applied in the *same* STT
    iteration that then processes the stale ``is_final``.
    """

    def __init__(
        self,
        partials: list[Hypothesis],
        *,
        clock: ManualClock,
        gate: asyncio.Event,
        gate_before_index: int,
        partial_delay_ms: int = 40,
    ) -> None:
        self._partials = partials
        self._clock = clock
        self._gate = gate
        self._gate_before_index = gate_before_index
        self._partial_delay_ms = partial_delay_ms

    async def _stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        audio_iter = aiter(audio)
        for i, partial in enumerate(self._partials):
            with contextlib.suppress(StopAsyncIteration):
                await anext(audio_iter)
            if i == self._gate_before_index:
                await self._gate.wait()
            await self._clock.sleep(self._partial_delay_ms)
            yield partial

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        return self._stream(audio)


def _closed_segment(text: str, index: int) -> Segment:
    """Build a CLOSED single-token segment for direct-queue supervisor tests."""
    toks = make_tokens([text])
    return Segment(
        text=text, tokens=toks, token_span=(index, index + len(toks)), closed=True, index=index
    )


def _src(
    clock: ManualClock, count: int, *, amplitude: float = 0.05, delay_ms: int = 20
) -> FakeAudioSource:
    frames = [frame(amplitude, t_ms=i * 20, n=320) for i in range(count)]
    return FakeAudioSource(frames, clock=clock, frame_delay_ms=delay_ms)


def _events_of(pipe: Pipeline, kind: str) -> list[MetricEvent]:
    return [e for e in pipe.metrics.events if e.kind == kind]


# ----- Simultaneity: first audio before final hypothesis ----------------------


async def test_first_audio_out_before_final_hypothesis() -> None:
    clock = ManualClock()
    # An utterance whose first sentence closes well before the final hypothesis.
    script = [
        [
            hyp("hello"),
            hyp("hello", "there."),  # sentence closes here -> segment -> audio
            hyp("hello", "there.", "friend"),
            hyp("hello", "there.", "friend."),
            hyp("hello", "there.", "friend.", is_final=True),  # final much later
        ]
    ]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=50)
    mt = FakeMT({"hello there.": "hola.", "friend.": "amigo."}, clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=10)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 8).frames()))
    await drain_then_advance(clock)
    await task

    first_audio = _events_of(pipe, "first_tts_out")
    finals = [e for e in pipe.metrics.events if e.kind == "commit"]
    assert first_audio, "expected target audio to be produced"
    # The first audio-out time must precede the final-hypothesis time. The final
    # hypothesis is the last STT partial; its commit/segment happens last. We
    # assert first audio happened strictly before the last commit (which carries
    # the final tail) — i.e. we are speaking before the utterance ends.
    assert first_audio[0].t_ms < finals[-1].t_ms
    assert sink.played, "sink should have received audio"


# ----- Audio-stage stability: a revision causes no retraction -----------------


async def test_midword_revision_causes_zero_tts_retraction() -> None:
    clock = ManualClock()
    # ASR first guesses "wether" then revises to "weather"; LocalAgreement (n=2)
    # never commits the wrong guess, so MT/TTS only ever see "weather".
    script = [
        [
            hyp("the"),
            hyp("the", "wether"),  # wrong guess
            hyp("the", "weather"),  # revised
            hyp("the", "weather", "today."),
            hyp("the", "weather", "today.", is_final=True),
        ]
    ]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    mt = FakeMT({"the weather today.": "el clima de hoy."}, clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=2, chunk_latency_ms=10)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 8).frames()))
    await drain_then_advance(clock)
    await task

    # MT never received the wrong word: every translated segment is the corrected
    # sentence, and the retraction count is zero.
    assert all("wether" not in seg.text for seg in mt.seen)
    assert any(seg.text == "the weather today." for seg in mt.seen)
    assert pipe.metrics.retraction_count == 0
    # The sink only ever received forward audio (no stop/retraction).
    assert sink.stop_count == 0


async def test_final_tail_without_terminal_punctuation_is_flushed_and_spoken() -> None:
    clock = ManualClock()
    # The utterance ends WITHOUT terminal punctuation; the final hypothesis must
    # still force-commit and the trailing clause must be flushed to MT/TTS.
    script = [
        [
            hyp("no"),
            hyp("no", "period"),
            hyp("no", "period", "here"),
            hyp("no", "period", "here", is_final=True),
        ]
    ]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    mt = FakeMT({"no period here": "sin punto aquí"}, clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=10)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 8).frames()))
    await drain_then_advance(clock)
    await task

    # The unterminated clause was force-closed at end-of-utterance and translated.
    assert any(seg.text == "no period here" for seg in mt.seen)
    assert sink.played, "the flushed final tail should produce audio"


async def test_post_commit_disagreement_metric_emitted_in_pipeline() -> None:
    clock = ManualClock()
    # n=2: commit "i scream", then a revision contradicts the committed "scream"
    # -> the stabilizer increments post_commit_disagreement and the pipeline
    # emits the metric event (the committed prefix never retracts).
    script = [
        [
            hyp("i", "scream"),
            hyp("i", "scream"),  # commits "i", "scream"
            hyp("i", "cream", "cone."),  # contradicts committed "scream"
            hyp("i", "cream", "cone.", is_final=True),
        ]
    ]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    mt = FakeMT({}, prefix="~", clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=10)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 8).frames()))
    await drain_then_advance(clock)
    await task

    disagreements = _events_of(pipe, "post_commit_disagreement")
    assert disagreements, "a contradicting revision must emit post_commit_disagreement"
    # And, crucially, no audio retraction happened despite the disagreement.
    assert pipe.metrics.retraction_count == 0


# ----- Barge-in: cancel + discard + stop + new utterance, no leak -------------


async def test_barge_in_cancels_discards_stops_and_starts_new_utterance() -> None:
    clock = ManualClock()
    # Utterance 1 is a long, slowly-translated sentence so MT/TTS is in flight
    # when the user barges in. Utterance 2 (the resumed speech) is gated so it is
    # only emitted AFTER the barge-in has been handled — faithfully modelling
    # speech that comes after the interrupt.
    barge_handled = asyncio.Event()

    async def gate(_u: int) -> None:
        await barge_handled.wait()

    utt1 = [
        hyp("the"),
        hyp("the", "long"),
        hyp("the", "long", "sentence."),
        hyp("the", "long", "sentence.", is_final=True),
    ]
    utt2 = [hyp("resumed."), hyp("resumed.", is_final=True)]
    stt = FakeSTT([utt1, utt2], clock=clock, partial_delay_ms=40, gate=gate)
    mt = FakeMT(
        {"the long sentence.": "la oración larga.", "resumed.": "reanudado."},
        clock=clock,
        latency_ms=120,  # slow MT so work is in flight at barge-in
    )
    tts = FakeTTS(clock=clock, chunks=3, chunk_latency_ms=60)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 12).frames()))
    # Advance until the segment "the long sentence." has reached MT (work in
    # flight), then fire a barge-in.
    for _ in range(80):
        await asyncio.sleep(0)
        if mt.seen:
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    interrupted_id = pipe.utterance_id
    pipe._interrupt.fire()
    # Let the barge-in be handled, then release the resumed utterance.
    for _ in range(40):
        await asyncio.sleep(0)
        if _events_of(pipe, "sink_stopped"):
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    barge_handled.set()
    await drain_then_advance(clock)
    await task

    # An interrupt + sink stop were recorded; the sink was stopped at least once.
    assert _events_of(pipe, "interrupt"), "barge-in must record an interrupt"
    assert _events_of(pipe, "sink_stopped"), "barge-in must stop the sink"
    assert sink.stop_count >= 1

    # No leaked tasks: only the main test task remains.
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == [], f"leaked tasks after barge-in: {leaked}"

    # The resumed utterance was translated under a FRESH utterance id (the roll
    # happened), independently of the interrupted one.
    assert any(seg.text == "resumed." for seg in mt.seen)
    resumed_starts = [
        e
        for e in pipe.metrics.events
        if e.kind == "utterance_start" and e.utterance_id != interrupted_id
    ]
    assert resumed_starts, "resumed speech must start a new utterance after barge-in"


async def test_barge_in_does_not_retranslate_already_emitted_segments() -> None:
    clock = ManualClock()
    # "alpha." closes and is emitted; then a barge-in fires while "beta." is the
    # in-flight work. The resumed utterance is gated until after the barge-in.
    barge_handled = asyncio.Event()

    async def gate(_u: int) -> None:
        await barge_handled.wait()

    utt1 = [
        hyp("alpha."),
        hyp("alpha.", "beta."),
        hyp("alpha.", "beta.", is_final=True),
    ]
    utt2 = [hyp("gamma."), hyp("gamma.", is_final=True)]
    stt = FakeSTT([utt1, utt2], clock=clock, partial_delay_ms=40, gate=gate)
    mt = FakeMT(
        {"alpha.": "a.", "beta.": "b.", "gamma.": "g."},
        clock=clock,
        latency_ms=80,
    )
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=30)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 10).frames()))
    # Let "alpha." reach MT, then barge in.
    for _ in range(80):
        await asyncio.sleep(0)
        if any(seg.text == "alpha." for seg in mt.seen):
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    pipe._interrupt.fire()
    for _ in range(40):
        await asyncio.sleep(0)
        if _events_of(pipe, "sink_stopped"):
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    barge_handled.set()
    await drain_then_advance(clock)
    await task

    # "alpha." was translated exactly once and never re-translated after barge-in.
    alpha_count = sum(1 for seg in mt.seen if seg.text == "alpha.")
    assert alpha_count == 1


# ----- Backpressure: bounded queues, no deadlock ------------------------------


async def test_bounded_queues_do_not_deadlock_under_backpressure() -> None:
    clock = ManualClock()
    # Many short sentences with a tiny queue bound forces backpressure; the
    # drain-then-advance harness must still drive it to completion.
    partials = []
    text_so_far: list[str] = []
    mapping = {}
    for i in range(8):
        word = f"s{i}."
        text_so_far.append(word)
        partials.append(hyp(*text_so_far))
        partials.append(hyp(*text_so_far))  # repeat so LocalAgreement commits
        mapping[word] = f"t{i}."
    partials.append(hyp(*text_so_far, is_final=True))
    stt = FakeSTT([partials], clock=clock, partial_delay_ms=20)
    mt = FakeMT(mapping, clock=clock, latency_ms=40)  # slower than STT -> backlog
    tts = FakeTTS(clock=clock, chunks=2, chunk_latency_ms=30)
    sink = FakeAudioSink(clock=clock)
    cfg = PipelineConfig(queue_maxsize=2)  # tight bound
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=cfg)

    task = asyncio.ensure_future(pipe.run(_src(clock, 20).frames()))
    await drain_then_advance(clock)
    await task  # completes => no deadlock

    # All 8 sentences were translated and produced audio.
    assert len({seg.text for seg in mt.seen}) == 8
    assert len(sink.played) == 8 * 2  # 8 segments x 2 chunks


# ----- Barge-in via the real detector + mic fan-out ---------------------------


async def test_barge_in_through_detector_and_tee() -> None:
    clock = ManualClock()
    # The source opens with a couple of silence frames (arming the detector),
    # then carries loud frames (speech) so the EnergyVAD/BargeInDetector fire an
    # onset on the fanned-out copy while STT also consumes it. The leading silence
    # models the gap before the speaker resumes (a real barge-in), which the
    # detector now requires before firing.
    silence = [
        AudioFrame(samples=np.zeros(320, dtype=np.float32), sample_rate=16000, t_ms=i * 20)
        for i in range(2)
    ]
    loud = [
        AudioFrame(
            samples=np.full(320, 0.5, dtype=np.float32), sample_rate=16000, t_ms=(2 + i) * 20
        )
        for i in range(12)
    ]
    source = FakeAudioSource(silence + loud, clock=clock, frame_delay_ms=20)
    script = [
        [
            hyp("keep."),
            hyp("keep.", "talking."),
            hyp("keep.", "talking.", is_final=True),
        ]
    ]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    mt = FakeMT({"keep.": "sigue.", "talking.": "hablando."}, clock=clock, latency_ms=40)
    tts = FakeTTS(clock=clock, chunks=2, chunk_latency_ms=40)
    sink = FakeAudioSink(clock=clock)
    detector = BargeInDetector(EnergyVAD(threshold=0.02, hangover_ms=0), onset_ms=40, clock=clock)
    pipe = Pipeline(
        stt=stt,
        mt=mt,
        tts=tts,
        sink=sink,
        clock=clock,
        config=PipelineConfig(queue_maxsize=4),
        barge_in=detector,
    )

    broadcaster, (stt_sub, barge_sub) = tee(source, 2, maxsize=4)

    async def run() -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(broadcaster.run(), name="bc")
            tg.create_task(pipe.run_with_barge_in(stt_sub, barge_sub), name="pipe")

    task = asyncio.ensure_future(run())
    await drain_then_advance(clock)
    await task

    # The sustained loud source triggers at least one barge-in onset.
    assert _events_of(pipe, "interrupt"), "loud source should trigger a barge-in"
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


# ----- Regression: is-final/barge-in double-roll re-translation (HIGH #1) ------


async def test_barge_in_coinciding_with_stale_is_final_does_not_retranslate() -> None:
    """A barge-in roll applied in the same STT step as the interrupted utterance's
    ``is_final`` must NOT force-commit that stale tail (which would re-segment and
    re-translate everything already emitted before the barge-in).

    Determinism: the interrupted utterance's ``is_final`` hypothesis is held just
    before being emitted (``_GatedSTT``). The test drives until "alpha." has been
    translated + the sink stopped by the barge-in (so the roll is pending), then
    releases the gate so the *next* STT iteration both applies the roll AND sees
    the stale ``is_final``. MT must see each source segment at most once.
    """
    clock = ManualClock()
    gate = asyncio.Event()
    # "alpha." commits (n=2) and is emitted; "beta." is still tentative; the final
    # hypothesis (index 2) is gated so the barge-in roll lands on the same step.
    partials = [
        hyp("alpha."),
        hyp("alpha.", "beta."),  # commits "alpha." -> segment -> MT
        hyp("alpha.", "beta.", is_final=True),  # gated stale final
    ]
    stt = _GatedSTT(partials, clock=clock, gate=gate, gate_before_index=2, partial_delay_ms=40)
    mt = FakeMT({"alpha.": "a.", "beta.": "b."}, clock=clock, latency_ms=60)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=20)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 8).frames()))
    # Drive until "alpha." has reached MT (segment emitted), then barge in.
    for _ in range(80):
        await asyncio.sleep(0)
        if any(seg.text == "alpha." for seg in mt.seen):
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    pipe._interrupt.fire()
    # Let the barge-in be fully handled (roll requested) before releasing the
    # gated final hypothesis, so the roll and the stale final coincide.
    for _ in range(40):
        await asyncio.sleep(0)
        if _events_of(pipe, "sink_stopped"):
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    assert _events_of(pipe, "sink_stopped"), "barge-in must have stopped the sink first"
    gate.set()
    await drain_then_advance(clock)
    await task

    # The crux: "alpha." was translated exactly once. Without the fix, the stale
    # is_final re-commits from index 0 under the new utterance and "alpha." (and
    # "beta.") are re-segmented and re-translated.
    assert sum(1 for seg in mt.seen if seg.text == "alpha.") == 1
    # No source segment is ever translated more than once.
    counts: dict[str, int] = {}
    for seg in mt.seen:
        counts[seg.text] = counts.get(seg.text, 0) + 1
    assert all(c == 1 for c in counts.values()), f"a segment was re-translated: {counts}"


# ----- Regression: post-drain in-flight segment leak (MEDIUM #2) ---------------


async def test_segment_stuck_on_full_queue_is_discarded_after_barge_in() -> None:
    """A segment that unblocks from a full-queue ``put`` AFTER a barge-in's drain
    (so it carries the interrupted utterance's id) must be discarded, not
    translated, while the resumed utterance's segment is translated.

    Determinism: the supervisor is driven directly over a hand-built segment
    queue. "A." is put first and reaches the (slow) MT; the test fires a barge-in
    while it is in flight, which abandons "A."'s utterance and drains the queue.
    The stale "B." (stamped with the abandoned utterance) and the resumed "C."
    are enqueued only AFTER the drain — exactly modelling the producer that
    unblocked late — and the supervisor must drop "B." but translate "C.".
    """
    clock = ManualClock()
    mt = FakeMT({"A.": "a.", "B.": "b.", "C.": "c."}, clock=clock, latency_ms=80)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=20)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=FakeSTT([], clock=clock), mt=mt, tts=tts, sink=sink, clock=clock)

    seg_queue: asyncio.Queue[_QueuedSegment | None] = asyncio.Queue(maxsize=8)
    interrupted_id = pipe.utterance_id  # "utt-1"
    await seg_queue.put(_QueuedSegment(_closed_segment("A.", 0), interrupted_id, ()))
    sup = asyncio.ensure_future(pipe._mt_tts_supervisor(seg_queue))

    # Drive until "A." is in flight at MT, then fire the barge-in.
    for _ in range(80):
        await asyncio.sleep(0)
        if any(seg.text == "A." for seg in mt.seen):
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    pipe._interrupt.fire()
    for _ in range(40):
        await asyncio.sleep(0)
        if _events_of(pipe, "sink_stopped"):
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    assert _events_of(pipe, "sink_stopped"), "barge-in must have run (queue drained)"

    # Now enqueue the late-unblocking stale segment (interrupted utterance id) and
    # a genuine resumed-utterance segment, then the stop sentinel.
    resumed_id = "utt-99"
    await seg_queue.put(_QueuedSegment(_closed_segment("B.", 1), interrupted_id, ()))
    await seg_queue.put(_QueuedSegment(_closed_segment("C.", 0), resumed_id, ()))
    await seg_queue.put(None)
    await drain_then_advance(clock)
    await sup

    # "B." (stale) must never be translated; "C." (resumed) must be.
    assert all(seg.text != "B." for seg in mt.seen), "stale post-barge-in segment was translated"
    assert any(seg.text == "C." for seg in mt.seen), "resumed segment must be translated"
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


# ----- Regression: simultaneous work-done + interrupt race (MEDIUM #3) ---------


class _InterruptOnLastChunkTTS:
    """A TTS that fires the pipeline interrupt while emitting its final chunk.

    This makes ``work`` (translate+synthesize) complete on the *same* tick the
    interrupt becomes set, deterministically reproducing the both-in-``done`` race
    in ``_synthesize_segment`` without juggling the manual clock.
    """

    def __init__(self, pipe: Pipeline, *, clock: ManualClock) -> None:
        self._pipe = pipe
        self._clock = clock

    async def _synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        await self._clock.sleep(10)
        # Fire the interrupt as the final chunk is produced: by the time the work
        # task completes, the interrupt event is already set (both land in `done`).
        self._pipe._interrupt.fire()
        yield TtsChunk(
            samples=np.full(8, 0.1, dtype=np.float32),
            sample_rate=16000,
            segment_index=segment_index,
            utterance_id=utterance_id,
            final=True,
        )

    def synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        return self._synthesize(text, segment_index=segment_index, utterance_id=utterance_id)


async def test_simultaneous_work_done_and_interrupt_keeps_audio_no_spurious_roll() -> None:
    """When synthesis completes on the same tick an interrupt fires, the finished
    segment's audio is kept and the stale interrupt does NOT spuriously barge-in
    the next segment (no extra roll, the just-finished audio is not discarded).
    """
    clock = ManualClock()
    mt = FakeMT({"first.": "uno.", "second.": "dos."}, clock=clock, latency_ms=10)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(
        stt=FakeSTT([], clock=clock),
        mt=mt,
        tts=FakeTTS(clock=clock),  # replaced below
        sink=sink,
        clock=clock,
    )
    pipe._tts = _InterruptOnLastChunkTTS(pipe, clock=clock)  # type: ignore[assignment]

    seg_queue: asyncio.Queue[_QueuedSegment | None] = asyncio.Queue(maxsize=8)
    # segA finishes synthesis exactly as the interrupt fires (both-done race);
    # segB follows and must be synthesized normally (no spurious barge-in).
    await seg_queue.put(_QueuedSegment(_closed_segment("first.", 0), "utt-1", ()))
    await seg_queue.put(_QueuedSegment(_closed_segment("second.", 1), "utt-1", ()))
    await seg_queue.put(None)
    sup = asyncio.ensure_future(pipe._mt_tts_supervisor(seg_queue))
    await drain_then_advance(clock)
    await sup

    # The just-finished segment's audio is kept (segA's chunk reached the sink).
    assert any(c.segment_index == 0 for c in sink.played), "finished segment's audio must be kept"
    # No spurious barge-in: the sink was never stopped and segB was translated
    # (it was not cancelled/discarded by a stale interrupt).
    assert sink.stop_count == 0, "a spurious barge-in stopped the sink"
    assert _events_of(pipe, "interrupt") == [], "a spurious barge-in was triggered"
    assert pipe._roll_requests == 0, "a spurious roll was requested"
    assert any(seg.text == "second." for seg in mt.seen), "the next segment must be translated"
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


# ----- Regression: shutdown sentinel never blocks on a full tts_queue (LOW #4) -


def test_enqueue_tts_sentinel_never_blocks_on_full_queue() -> None:
    """The shutdown stop-sentinel is enqueued without blocking even when the
    tts_queue is full (the original ``await put(None)`` could hang forever if
    playback had stalled on a slow sink at teardown).
    """
    q: asyncio.Queue[TtsChunk | None] = asyncio.Queue(maxsize=2)
    chunk = TtsChunk(
        samples=np.full(4, 0.1, dtype=np.float32),
        sample_rate=16000,
        segment_index=0,
        utterance_id="utt-1",
    )
    q.put_nowait(chunk)
    q.put_nowait(chunk)
    assert q.full()  # a blocking put here would hang forever if never drained

    _enqueue_tts_sentinel(q)  # must return immediately, dropping oldest to fit

    # The stop-sentinel is now reachable so the playback task will terminate.
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert items[-1] is None, "stop-sentinel must be enqueued"
    assert None in items
