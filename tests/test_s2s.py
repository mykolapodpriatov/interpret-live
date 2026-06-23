"""Unified S2S path tests via FakeS2S behind the same Session surface.

Covers the cloud path: source audio → translated audio, the stabilizer honestly
bypassed (it never runs on this path), and barge-in sending the provider's
interrupt + stopping the sink.
"""

from __future__ import annotations

import asyncio

import numpy as np

from helpers import frame
from interpret_live.audio_io import FakeAudioSink, FakeAudioSource
from interpret_live.backends.fake import FakeS2S
from interpret_live.clock import ManualClock, drain_then_advance
from interpret_live.config import PipelineConfig
from interpret_live.s2s import S2SPipeline
from interpret_live.session import S2SBackend, Session
from interpret_live.types import AudioFrame


def _src(clock: ManualClock, count: int) -> FakeAudioSource:
    frames = [frame(0.5, t_ms=i * 20, n=320) for i in range(count)]
    return FakeAudioSource(frames, clock=clock, frame_delay_ms=20)


async def test_s2s_streams_translated_audio_to_sink() -> None:
    clock = ManualClock()
    s2s = FakeS2S(chunks_per_utterance=3, clock=clock, chunk_latency_ms=30)
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 6).frames()))
    await drain_then_advance(clock)
    await task

    assert len(sink.played) == 3
    # first-audio-out latency is recorded on the S2S path too.
    report = pipe.metrics.report()
    assert report.utterances
    assert report.utterances[0].first_audio_out_ms is not None


async def test_s2s_barge_in_interrupts_provider_and_stops_sink() -> None:
    clock = ManualClock()
    s2s = FakeS2S(chunks_per_utterance=6, clock=clock, chunk_latency_ms=40)
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    task = asyncio.ensure_future(pipe.run(_src(clock, 12).frames()))
    # Let a couple of chunks play, then interrupt.
    for _ in range(40):
        await asyncio.sleep(0)
        if len(sink.played) >= 1:
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    pipe._interrupt.set()
    await drain_then_advance(clock)
    await task

    assert s2s.interrupt_count >= 1, "provider interrupt must be sent on barge-in"
    assert sink.stop_count >= 1, "sink must be stopped on barge-in"
    # No leaked tasks.
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


async def test_s2s_session_barge_in_via_tee_sends_provider_interrupt() -> None:
    clock = ManualClock()
    # Source opens with silence (arming the detector), then loud frames so the
    # EnergyVAD/BargeInDetector fire an onset on the fanned-out copy while the S2S
    # provider streams; the session must send the provider's interrupt and stop
    # the sink. The leading silence models the gap before the speaker resumes.
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
    s2s = FakeS2S(chunks_per_utterance=8, clock=clock, chunk_latency_ms=40)
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
    s2s = FakeS2S(chunks_per_utterance=2, clock=clock, chunk_latency_ms=20)
    sink = FakeAudioSink(clock=clock)
    source = _src(clock, 6)
    backend = S2SBackend(s2s=s2s)
    # The S2S backend declares the stabilizer is NOT active (cloud-internal).
    assert backend.capabilities.stabilizer is False

    session = Session.create(
        backend=backend, source=source, sink=sink, clock=clock, config=PipelineConfig()
    )
    task = asyncio.ensure_future(session.run())
    await drain_then_advance(clock)
    await task
    assert len(sink.played) == 2
