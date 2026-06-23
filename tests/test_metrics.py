"""Metrics tests: first-audio-out, commit lag, retraction count, barge-in stop.

All metrics are derived deterministically from the in-memory event log, so the
tests build a log directly and assert the derivations, plus cross-check against
a real pipeline run.
"""

from __future__ import annotations

import asyncio

from helpers import frame, hyp
from interpret_live.audio_io import FakeAudioSink, FakeAudioSource
from interpret_live.backends.fake import FakeMT, FakeSTT, FakeTTS
from interpret_live.clock import ManualClock, drain_then_advance
from interpret_live.config import PipelineConfig
from interpret_live.metrics import MetricsLog
from interpret_live.pipeline import Pipeline
from interpret_live.types import MetricEvent


def _ev(kind: str, t_ms: int, uid: str) -> MetricEvent:
    return MetricEvent(kind=kind, t_ms=t_ms, utterance_id=uid)  # type: ignore[arg-type]


def test_first_audio_out_latency_derivation() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 100, "u1"))
    log.append(_ev("first_tts_out", 450, "u1"))
    m = log.for_utterance("u1")
    assert m.first_audio_out_ms == 350


def test_commit_lag_derivation() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 100, "u1"))
    log.append(_ev("commit", 220, "u1"))
    log.append(_ev("commit", 300, "u1"))  # only the FIRST commit defines lag
    m = log.for_utterance("u1")
    assert m.commit_lag_ms == 120


def test_barge_in_stop_time_derivation() -> None:
    log = MetricsLog()
    log.append(_ev("interrupt", 1000, "u1"))
    log.append(_ev("sink_stopped", 1040, "u1"))
    m = log.for_utterance("u1")
    assert m.barge_in_stop_ms == 40


def test_retraction_count_is_zero_on_stable_path() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    log.append(_ev("first_tts_out", 100, "u1"))
    assert log.retraction_count == 0
    assert log.report().total_retractions == 0


def test_post_commit_disagreement_counted_per_utterance() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    log.append(_ev("post_commit_disagreement", 50, "u1"))
    log.append(_ev("post_commit_disagreement", 80, "u1"))
    m = log.for_utterance("u1")
    assert m.post_commit_disagreement == 2
    assert log.report().total_post_commit_disagreement == 2


def test_missing_events_yield_none() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    m = log.for_utterance("u1")
    assert m.first_audio_out_ms is None
    assert m.commit_lag_ms is None
    assert m.barge_in_stop_ms is None


def test_report_aggregates_worst_case() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    log.append(_ev("first_tts_out", 200, "u1"))
    log.append(_ev("utterance_start", 1000, "u2"))
    log.append(_ev("first_tts_out", 1500, "u2"))
    report = log.report()
    assert report.max_first_audio_out_ms == 500  # u2 is worse than u1 (200)


def test_record_retraction_increments() -> None:
    log = MetricsLog()
    log.record_retraction()
    log.record_retraction(2)
    assert log.retraction_count == 3


def test_report_max_barge_in_stop_across_utterances() -> None:
    log = MetricsLog()
    log.append(_ev("interrupt", 100, "u1"))
    log.append(_ev("sink_stopped", 130, "u1"))  # 30ms
    log.append(_ev("interrupt", 500, "u2"))
    log.append(_ev("sink_stopped", 560, "u2"))  # 60ms (worse)
    report = log.report()
    assert report.max_barge_in_stop_ms == 60


def test_report_max_metrics_none_when_no_events() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    report = log.report()
    assert report.max_first_audio_out_ms is None
    assert report.max_barge_in_stop_ms is None


def test_events_property_returns_all_in_order() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    log.append(_ev("commit", 50, "u1"))
    kinds = [e.kind for e in log.events]
    assert kinds == ["utterance_start", "commit"]


# ----- Cross-check against a real run -----------------------------------------


async def test_metrics_from_real_pipeline_run() -> None:
    clock = ManualClock()
    script = [
        [
            hyp("hello"),
            hyp("hello", "world."),
            hyp("hello", "world.", is_final=True),
        ]
    ]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=50)
    mt = FakeMT({"hello world.": "hola mundo."}, clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=10)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=PipelineConfig())

    frames = [frame(0.05, t_ms=i * 20, n=320) for i in range(6)]
    source = FakeAudioSource(frames, clock=clock, frame_delay_ms=20)
    task = asyncio.ensure_future(pipe.run(source.frames()))
    await drain_then_advance(clock)
    await task

    report = pipe.metrics.report()
    # Exactly one utterance, with positive first-audio-out and zero retractions.
    assert report.total_retractions == 0
    u = report.utterances[0]
    assert u.first_audio_out_ms is not None and u.first_audio_out_ms > 0
    assert u.commit_lag_ms is not None and u.commit_lag_ms > 0
