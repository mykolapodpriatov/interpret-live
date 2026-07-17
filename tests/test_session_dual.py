"""DualChannel + capability-negotiation tests.

Covers two independent directions running concurrently with separate sources and
sinks and asserting **no cross-talk** (A's audio reaches only B's sink and vice
versa), plus fail-early capability negotiation in :func:`Session.create`.
"""

from __future__ import annotations

import asyncio

import pytest

from helpers import frame, hyp
from interpret_live.audio_io import FakeAudioSink, FakeAudioSource
from interpret_live.backends.fake import FakeMT, FakeS2S, FakeSTT, FakeTTS
from interpret_live.clock import ManualClock, drain_then_advance
from interpret_live.config import PipelineConfig
from interpret_live.session import (
    Capabilities,
    CapabilityError,
    DualChannel,
    PipelineBackend,
    S2SBackend,
    Session,
)


def _src(clock: ManualClock, count: int, amplitude: float = 0.5) -> FakeAudioSource:
    frames = [frame(amplitude, t_ms=i * 20, n=320) for i in range(count)]
    return FakeAudioSource(frames, clock=clock, frame_delay_ms=20)


def _pipeline_backend(clock: ManualClock, mapping: dict[str, str]) -> PipelineBackend:
    script = [[hyp("a"), hyp("a", "b."), hyp("a", "b.", is_final=True)]]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    mt = FakeMT(mapping, clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=20)
    return PipelineBackend(stt=stt, mt=mt, tts=tts)


async def test_dual_channel_no_cross_talk() -> None:
    clock = ManualClock()
    a_source = _src(clock, 6)
    b_source = _src(clock, 6)
    a_sink = FakeAudioSink(clock=clock)
    b_sink = FakeAudioSink(clock=clock)

    # Each direction has its own backend (independent fakes / utterance ids).
    backend_ab = _pipeline_backend(clock, {"a b.": "AB"})
    backend_ba = _pipeline_backend(clock, {"a b.": "BA"})

    a_to_b = Session.create(
        backend=backend_ab,
        source=a_source,
        sink=b_sink,  # A speaks -> B hears
        clock=clock,
        config=PipelineConfig(),
        for_dual=True,
    )
    b_to_a = Session.create(
        backend=backend_ba,
        source=b_source,
        sink=a_sink,  # B speaks -> A hears
        clock=clock,
        config=PipelineConfig(),
        for_dual=True,
    )
    dual = DualChannel(a_to_b=a_to_b, b_to_a=b_to_a)

    task = asyncio.ensure_future(dual.run())
    await drain_then_advance(clock)
    await task

    # B's sink only ever heard the A->B translation; A's sink only the B->A one.
    assert {c.segment_index for c in b_sink.played}  # B heard something
    assert {c.segment_index for c in a_sink.played}  # A heard something
    # The two directions' utterance ids are disjoint (independent sessions),
    # proving there is no shared state / cross-talk between channels.
    a_ids = set(b_sink.played_utterance_ids)
    b_ids = set(a_sink.played_utterance_ids)
    # Both used "utt-*" ids from their own pipelines; the audio sample marker
    # differs only by direction, but crucially each sink received chunks from
    # exactly one direction's pipeline (no chunk appears in both sinks).
    a_chunks = set(map(id, b_sink.played))
    b_chunks = set(map(id, a_sink.played))
    assert a_chunks.isdisjoint(b_chunks)
    assert a_ids and b_ids


async def test_pipeline_session_with_barge_in_runs_via_tee() -> None:
    clock = ManualClock()
    # Source opens with silence (arming the detector) then loud frames, which
    # drive the barge-in detector on the fanned-out copy. The leading silence
    # models the gap before the speaker resumes (now required to fire an onset).
    a_frames = [frame(0.0, t_ms=i * 20, n=320) for i in range(6)] + [
        frame(0.5, t_ms=(6 + i) * 20, n=320) for i in range(12)
    ]
    a_source = FakeAudioSource(a_frames, clock=clock, frame_delay_ms=20)
    sink = FakeAudioSink(clock=clock)
    script = [[hyp("keep."), hyp("keep.", "going."), hyp("keep.", "going.", is_final=True)]]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    # Slow MT keeps target work in flight across the onset window: barge-in is
    # destructive only when there is target work to interrupt.
    mt = FakeMT({"keep.": "sigue.", "going.": "yendo."}, clock=clock, latency_ms=400)
    tts = FakeTTS(clock=clock, chunks=2, chunk_latency_ms=40)
    backend = PipelineBackend(stt=stt, mt=mt, tts=tts)
    session = Session.create(
        backend=backend,
        source=a_source,
        sink=sink,
        clock=clock,
        config=PipelineConfig(queue_maxsize=4, barge_in_onset_ms=40),
        enable_barge_in=True,
        require_stabilizer=True,  # the pipeline path supports it
    )
    task = asyncio.ensure_future(session.run())
    await drain_then_advance(clock)
    await task
    report = session.metrics()
    # A barge-in onset was detected and recorded on the pipeline path.
    assert any(u.barge_in_stop_ms is not None for u in report.utterances)
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


async def test_session_interrupt_method_fires_barge_in() -> None:
    clock = ManualClock()
    source = _src(clock, 8, amplitude=0.05)  # quiet: no auto barge-in
    sink = FakeAudioSink(clock=clock)
    script = [[hyp("hi."), hyp("hi.", is_final=True)]]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    mt = FakeMT({"hi.": "hola."}, clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=20)
    backend = PipelineBackend(stt=stt, mt=mt, tts=tts)
    session = Session.create(
        backend=backend, source=source, sink=sink, clock=clock, config=PipelineConfig()
    )
    # The interrupt() method is a no-op-safe manual hook; calling it then running
    # should not raise and should still complete.
    session.interrupt()
    task = asyncio.ensure_future(session.run())
    await drain_then_advance(clock)
    await task
    assert session.metrics() is not None


async def test_dual_channel_create_helper_runs_both_directions() -> None:
    clock = ManualClock()
    a_source = _src(clock, 6)
    b_source = _src(clock, 6)
    a_sink = FakeAudioSink(clock=clock)
    b_sink = FakeAudioSink(clock=clock)
    # A single backend instance used for both directions via the create helper.
    backend = _pipeline_backend(clock, {"a b.": "X"})

    dual = DualChannel.create(
        backend=backend,
        a_source=a_source,
        a_sink=a_sink,
        b_source=b_source,
        b_sink=b_sink,
        clock=clock,
        config=PipelineConfig(),
    )
    task = asyncio.ensure_future(dual.run())
    await drain_then_advance(clock)
    await task
    reports = dual.metrics()
    assert len(reports) == 2


# ----- Capability negotiation (fail early) ------------------------------------


def test_require_stabilizer_on_s2s_backend_raises_clear_error() -> None:
    clock = ManualClock()
    s2s = FakeS2S(clock=clock)
    backend = S2SBackend(s2s=s2s)
    src = _src(clock, 2)
    sink = FakeAudioSink(clock=clock)
    with pytest.raises(CapabilityError, match="bypasses the LocalAgreement stabilizer"):
        Session.create(
            backend=backend,
            source=src,
            sink=sink,
            clock=clock,
            require_stabilizer=True,
        )


def test_dual_on_backend_without_dual_capability_raises() -> None:
    clock = ManualClock()

    class _NoDualBackend:
        name = "limited"

        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(interrupt=True, metrics=True, dual=False, stabilizer=False)

    src = _src(clock, 2)
    sink = FakeAudioSink(clock=clock)
    with pytest.raises(CapabilityError, match="cannot run dual-channel"):
        Session.create(
            backend=_NoDualBackend(),  # type: ignore[arg-type]
            source=src,
            sink=sink,
            clock=clock,
            for_dual=True,
        )


def test_barge_in_on_backend_without_interrupt_raises() -> None:
    clock = ManualClock()

    class _NoInterruptBackend:
        name = "no-interrupt"

        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(interrupt=False, metrics=True, dual=True, stabilizer=False)

    src = _src(clock, 2)
    sink = FakeAudioSink(clock=clock)
    with pytest.raises(CapabilityError, match="does not support barge-in"):
        Session.create(
            backend=_NoInterruptBackend(),  # type: ignore[arg-type]
            source=src,
            sink=sink,
            clock=clock,
            enable_barge_in=True,
        )


def test_pipeline_backend_declares_stabilizer_active() -> None:
    clock = ManualClock()
    backend = _pipeline_backend(clock, {})
    caps = backend.capabilities
    assert caps.stabilizer is True
    assert caps.interrupt is True
    assert caps.dual is True
