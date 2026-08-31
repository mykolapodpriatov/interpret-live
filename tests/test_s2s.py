"""Persistent S2S protocol tests via FakeS2S behind the same Session surface.

Covers the session-long provider connection: provider-driven local turn
creation (onset timestamps from provider speech-start events), response/item
ownership maps, response-ID-scoped barge-in with heard-audio cursors,
post-interrupt recovery on the same connection, terminal status semantics,
and clean teardown.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from helpers import frame
from interpret_live.audio_io import FakeAudioSink, FakeAudioSource
from interpret_live.backends.fake import FakeS2S, FakeS2STurn
from interpret_live.clock import ManualClock, drain_then_advance
from interpret_live.config import PipelineConfig
from interpret_live.s2s import S2SPipeline
from interpret_live.session import S2SBackend, Session
from interpret_live.types import AudioFrame, MetricEvent, S2SResponseError


def _src(clock: ManualClock, count: int) -> FakeAudioSource:
    frames = [frame(0.5, t_ms=i * 20, n=320) for i in range(count)]
    return FakeAudioSource(frames, clock=clock, frame_delay_ms=20)


def _events_of(pipe: S2SPipeline, kind: str) -> list[MetricEvent]:
    return [e for e in pipe.metrics.events if e.kind == kind]


async def _run_until(pipe_task: asyncio.Task, clock: ManualClock, cond) -> None:  # type: ignore[no-untyped-def]
    """Drive the manual clock gently until ``cond()`` holds.

    Several bare yields run between clock advances so tasks keep pace with
    logical time and the condition is observed at the earliest moment.
    """
    for _ in range(400):
        for _ in range(6):
            await asyncio.sleep(0)
            if cond():
                return
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    raise AssertionError("condition never became true")


async def test_s2s_streams_translated_audio_with_provider_onset_metrics() -> None:
    clock = ManualClock()
    s2s = FakeS2S(
        clock=clock,
        turns=[FakeS2STurn(chunks=3, speech_started_at_ms=20)],
        chunk_latency_ms=30,
    )
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 6).frames()))
    await drain_then_advance(clock)
    await task

    assert len(sink.played) == 3
    report = pipe.metrics.report()
    assert report.utterances
    u = report.utterances[0]
    # utterance_start came from the provider speech-start event's mapped source
    # onset (20 ms), so first-audio latency is anchored at real speech onset.
    starts = _events_of(pipe, "utterance_start")
    assert starts[0].t_ms == 20
    assert u.first_audio_out_ms is not None and u.first_audio_out_ms > 0


async def test_two_turns_produce_independent_nonzero_first_audio_latencies() -> None:
    clock = ManualClock()
    s2s = FakeS2S(
        clock=clock,
        turns=[
            FakeS2STurn(chunks=2, speech_started_at_ms=0),
            FakeS2STurn(chunks=2, speech_started_at_ms=200, frames_before=2),
        ],
        chunk_latency_ms=30,
    )
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 12).frames()))
    await drain_then_advance(clock)
    await task

    report = pipe.metrics.report()
    latencies = [u.first_audio_out_ms for u in report.utterances]
    assert len(latencies) == 2
    assert all(latency is not None and latency > 0 for latency in latencies)
    starts = _events_of(pipe, "utterance_start")
    assert [e.t_ms for e in starts] == [0, 200]


async def test_barge_in_sends_exactly_one_response_scoped_interrupt() -> None:
    clock = ManualClock()
    s2s = FakeS2S(
        clock=clock,
        turns=[FakeS2STurn(chunks=6), FakeS2STurn(chunks=2, frames_before=2)],
        chunk_latency_ms=40,
    )
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 16).frames()))
    await _run_until(task, clock, lambda: len(sink.played) >= 1)
    pipe._interrupt.set()
    await drain_then_advance(clock)
    await task

    # Exactly one provider interrupt, scoped to the first response.
    assert [t.response_id for t in s2s.interrupt_targets] == ["resp-1"]
    assert sink.stop_count >= 1
    # The same provider session continued: the second turn produced audio
    # under a NEW local utterance id without reconnecting.
    utts = {c.utterance_id for c in sink.played}
    assert "s2s-2" in utts, f"post-interrupt speech must produce audio: {utts}"
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


async def test_cursor_reports_only_presented_audio() -> None:
    clock = ManualClock()
    # One 400 ms chunk; we interrupt 150 ms into its presentation.
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=1, chunk_ms=400)], chunk_latency_ms=20)
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 10).frames()))
    await _run_until(task, clock, lambda: len(sink.played) >= 1)
    started_at = clock.now_ms()
    # Present 150 ms of the chunk, then barge in.
    clock.advance(started_at + 150)
    await asyncio.sleep(0)
    pipe._interrupt.set()
    await drain_then_advance(clock)
    await task

    assert len(s2s.interrupt_targets) == 1
    cursor = s2s.interrupt_targets[0].cursor
    assert cursor is not None
    assert cursor.response_id == "resp-1"
    # Only the DAC-passed 150 ms counts; the queued remainder is excluded.
    assert cursor.audio_end_ms == 150


async def test_interrupt_before_any_audio_sends_cancel_without_cursor() -> None:
    clock = ManualClock()
    # Long latency before the first chunk: interrupt lands mid-generation.
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=2)], chunk_latency_ms=500)
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 8).frames()))
    # Wait until the response exists but no audio has been produced yet.
    await _run_until(task, clock, lambda: pipe._current_response is not None)
    assert not sink.played
    pipe._interrupt.set()
    await drain_then_advance(clock)
    await task

    assert len(s2s.interrupt_targets) == 1
    assert s2s.interrupt_targets[0].response_id == "resp-1"
    assert s2s.interrupt_targets[0].cursor is None  # nothing was ever audible


async def test_speech_with_nothing_to_interrupt_sends_no_provider_cancel() -> None:
    clock = ManualClock()
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=1, frames_before=4)])
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 10).frames()))
    # Fire the interrupt before ANY provider turn/response exists.
    pipe._interrupt.set()
    await drain_then_advance(clock)
    await task

    assert s2s.interrupt_targets == []
    assert sink.stop_count == 0
    # The session ran normally afterwards.
    assert sink.played


async def test_late_events_from_cancelled_response_are_ignored() -> None:
    clock = ManualClock()
    s2s = FakeS2S(
        clock=clock,
        turns=[
            FakeS2STurn(chunks=6, late_chunks_after_interrupt=2),
            FakeS2STurn(chunks=1, frames_before=2),
        ],
        chunk_latency_ms=40,
    )
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 14).frames()))
    await _run_until(task, clock, lambda: len(sink.played) >= 1)
    played_before = len(sink.played)
    pipe._interrupt.set()
    await drain_then_advance(clock)
    await task

    # Late deltas from resp-1 never reached the sink under the old utterance.
    resp1_chunks = [c for c in sink.played if c.utterance_id == "s2s-1"]
    assert len(resp1_chunks) <= played_before
    # The cancelled response's done event did not create a second fresh turn:
    # exactly two utterance_starts (one per provider speech-start).
    assert len(_events_of(pipe, "utterance_start")) == 2
    # And the new turn still played.
    assert any(c.utterance_id == "s2s-2" for c in sink.played)


async def test_response_done_completed_keeps_trailing_playback_ownership() -> None:
    clock = ManualClock()
    # A big chunk: the provider finishes the response long before the audio
    # finishes presenting; then a barge-in during trailing playback must still
    # cancel/truncate that exact finished response.
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=1, chunk_ms=600)], chunk_latency_ms=20)
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 8).frames()))
    await _run_until(task, clock, lambda: len(sink.played) >= 1)
    t0 = clock.now_ms()
    clock.advance(t0 + 100)  # 100 ms of the 600 ms chunk presented
    await asyncio.sleep(0)
    pipe._interrupt.set()
    await drain_then_advance(clock)
    await task

    assert [t.response_id for t in s2s.interrupt_targets] == ["resp-1"]
    cursor = s2s.interrupt_targets[0].cursor
    assert cursor is not None and 0 < cursor.audio_end_ms < 600


async def test_failed_response_status_surfaces_typed_error() -> None:
    clock = ManualClock()
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=1, status="failed")])
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 6).frames()))
    await drain_then_advance(clock)
    with pytest.raises(S2SResponseError, match=r"resp-1.*failed"):
        await task


async def test_incomplete_response_status_surfaces_typed_error() -> None:
    clock = ManualClock()
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=1, status="incomplete")])
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 6).frames()))
    await drain_then_advance(clock)
    with pytest.raises(S2SResponseError, match="incomplete"):
        await task


async def test_unexpected_cancelled_status_surfaces_typed_error() -> None:
    clock = ManualClock()
    # The provider claims cancellation we never asked for.
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=1, status="cancelled")])
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 6).frames()))
    await drain_then_advance(clock)
    with pytest.raises(S2SResponseError, match="cancelled"):
        await task


async def test_s2s_session_barge_in_via_tee_sends_provider_interrupt() -> None:
    clock = ManualClock()
    silence = [
        AudioFrame(samples=np.zeros(320, dtype=np.float32), sample_rate=16000, t_ms=i * 20)
        for i in range(2)
    ]
    loud = [
        AudioFrame(
            samples=np.full(320, 0.5, dtype=np.float32), sample_rate=16000, t_ms=(2 + i) * 20
        )
        for i in range(16)
    ]
    source = FakeAudioSource(silence + loud, clock=clock, frame_delay_ms=20)
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=8, chunk_ms=200)], chunk_latency_ms=40)
    sink = FakeAudioSink(clock=clock)
    backend = S2SBackend(s2s=s2s)
    cfg = PipelineConfig(queue_maxsize=4, barge_in_onset_ms=40, vad_threshold=0.02)
    session = Session.create(
        backend=backend,
        source=source,
        sink=sink,
        clock=clock,
        config=cfg,
        enable_barge_in=True,
    )

    task = asyncio.ensure_future(session.run())
    await drain_then_advance(clock)
    await task

    assert s2s.interrupt_count >= 1
    assert sink.stop_count >= 1
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


async def test_s2s_session_dispatch_bypasses_stabilizer() -> None:
    clock = ManualClock()
    s2s = FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=2)], chunk_latency_ms=20)
    sink = FakeAudioSink(clock=clock)
    source = _src(clock, 6)
    backend = S2SBackend(s2s=s2s)
    assert backend.capabilities.stabilizer is False

    session = Session.create(
        backend=backend, source=source, sink=sink, clock=clock, config=PipelineConfig()
    )
    task = asyncio.ensure_future(session.run())
    await drain_then_advance(clock)
    await task
    assert len(sink.played) == 2
