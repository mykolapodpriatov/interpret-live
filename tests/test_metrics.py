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


def test_commit_to_audio_derivation_isolates_backend_cost() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 100, "u1"))
    log.append(_ev("commit", 220, "u1"))  # first commit anchors the MT+TTS window
    log.append(_ev("first_tts_out", 450, "u1"))
    log.append(_ev("commit", 300, "u1"))  # a later commit must not move the anchor
    m = log.for_utterance("u1")
    assert m.commit_to_audio_ms == 230  # 450 - 220 (backend cost, not stabilizer lag)
    # It is exactly the span between commit lag and first-audio-out.
    assert m.first_audio_out_ms == m.commit_lag_ms + m.commit_to_audio_ms  # type: ignore[operator]


def test_commit_to_audio_is_none_without_both_events() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    log.append(_ev("commit", 50, "u1"))  # committed, but no audio yet
    assert log.for_utterance("u1").commit_to_audio_ms is None


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


def test_utterance_metrics_to_dict_is_explicit_and_json_ready() -> None:
    import json

    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    log.append(_ev("first_tts_out", 200, "u1"))
    log.append(_ev("commit", 120, "u1"))
    d = log.for_utterance("u1").to_dict()
    assert d == {
        "utterance_id": "u1",
        "first_audio_out_ms": 200,
        "commit_lag_ms": 120,
        "commit_to_audio_ms": 80,  # first_tts_out (200) - first_commit (120)
        "barge_in_stop_ms": None,
        "retraction_count": 0,
        "post_commit_disagreement": 0,
    }
    # None survives a JSON round-trip as null, not a dropped key.
    assert json.loads(json.dumps(d))["barge_in_stop_ms"] is None


def test_report_to_dict_serializes_utterances_and_aggregates() -> None:
    import json

    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    log.append(_ev("first_tts_out", 200, "u1"))
    log.append(_ev("post_commit_disagreement", 40, "u1"))
    log.record_retraction(2, utterance_id="u1")
    d = log.report().to_dict()
    assert d["total_retractions"] == 2
    assert d["total_post_commit_disagreement"] == 1
    assert d["max_first_audio_out_ms"] == 200
    assert d["max_barge_in_stop_ms"] is None
    utterances = d["utterances"]
    assert isinstance(utterances, list)
    assert utterances[0]["utterance_id"] == "u1"
    assert utterances[0]["retraction_count"] == 2
    # The whole report is JSON-serializable.
    assert json.loads(json.dumps(d))["total_retractions"] == 2


def test_missing_events_yield_none() -> None:
    log = MetricsLog()
    log.append(_ev("utterance_start", 0, "u1"))
    m = log.for_utterance("u1")
    assert m.first_audio_out_ms is None
    assert m.commit_lag_ms is None
    assert m.commit_to_audio_ms is None
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


# ----- commit-to-audio on the built-in bench fixtures -------------------------


def test_commit_to_audio_ms_on_builtin_fixtures() -> None:
    """The metric surfaces deterministically through ``run_bench`` on both fixtures.

    Values are exact (ManualClock + drain-then-advance) and, by construction,
    equal ``first_audio_out_ms - commit_lag_ms`` for every utterance.
    """
    from interpret_live.bench import get_fixture, run_bench

    default = asyncio.run(run_bench(get_fixture("default-en-2sent"))).report
    assert {u.utterance_id: u.commit_to_audio_ms for u in default.utterances} == {
        "utt-1": 290,
        "utt-2": 370,
    }

    late = asyncio.run(run_bench(get_fixture("late-revision-en"))).report
    assert [u.commit_to_audio_ms for u in late.utterances] == [290]

    for report in (default, late):
        for u in report.utterances:
            assert u.commit_to_audio_ms == u.first_audio_out_ms - u.commit_lag_ms  # type: ignore[operator]


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


# ----- deterministic multi-turn first-audio latency (both paths) ---------------


async def test_offline_multi_turn_latency_from_scripted_onsets_and_presentation() -> None:
    """Scripted source onsets + sink first-presentation timestamps yield exact
    non-zero per-turn first_audio_out_ms including decode/MT/TTS delay."""
    from helpers import make_tokens
    from interpret_live.types import Hypothesis

    clock = ManualClock()

    def _turn_hyp(word: str, turn: str, onset: int, final: bool) -> Hypothesis:
        return Hypothesis(
            tokens=make_tokens([word]),
            is_final=final,
            source_turn_id=turn,
            speech_started_at_ms=onset,
        )

    script = [
        [_turn_hyp("uno.", "t1", 5, False), _turn_hyp("uno.", "t1", 5, True)],
        [_turn_hyp("dos.", "t2", 115, False), _turn_hyp("dos.", "t2", 115, True)],
    ]
    stt = FakeSTT(script, clock=clock, partial_delay_ms=40)
    mt = FakeMT({"uno.": "U", "dos.": "D"}, clock=clock, latency_ms=20)
    tts = FakeTTS(clock=clock, chunks=1, chunk_latency_ms=10, ms_per_char=20)
    sink = FakeAudioSink(clock=clock)
    pipe = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock)

    frames = [frame(0.05, t_ms=i * 20, n=320) for i in range(10)]
    source = FakeAudioSource(frames, clock=clock, frame_delay_ms=20)
    task = asyncio.ensure_future(pipe.run(source.frames()))
    await drain_then_advance(clock)
    await task

    report = pipe.metrics.report()
    latencies = {
        u.utterance_id: u.first_audio_out_ms
        for u in report.utterances
        if u.first_audio_out_ms is not None
    }
    # Deterministic pacing: the fake source paces one frame per 20 ms and the
    # fake STT drains a frame + waits 40 ms per partial, so turn 1's final
    # lands at 120 (commit) -> +20 MT -> +10 TTS => audible at 150, i.e.
    # 145 ms after the scripted onset (5). Turn 2's final lands at 260 =>
    # audible at 270, i.e. 155 ms after its onset (115). Both latencies are
    # anchored at the scripted source onsets, not first-decode arrival.
    assert latencies == {"utt-1": 145, "utt-2": 155}


async def test_s2s_multi_turn_latency_includes_queued_playback_delay() -> None:
    """Cloud path: onsets from provider events; the second turn's first audio
    waits for the first turn's audio to finish presenting (honest latency)."""
    from interpret_live.backends.fake import FakeS2S, FakeS2STurn
    from interpret_live.s2s import S2SPipeline

    clock = ManualClock()
    s2s = FakeS2S(
        clock=clock,
        turns=[
            FakeS2STurn(chunks=1, chunk_ms=100, speech_started_at_ms=0),
            FakeS2STurn(chunks=1, chunk_ms=100, frames_before=2, speech_started_at_ms=60),
        ],
        chunk_latency_ms=30,
    )
    sink = FakeAudioSink(clock=clock)
    pipe = S2SPipeline(s2s=s2s, sink=sink, clock=clock, config=PipelineConfig())

    frames = [frame(0.05, t_ms=i * 20, n=320) for i in range(10)]
    source = FakeAudioSource(frames, clock=clock, frame_delay_ms=20)
    task = asyncio.ensure_future(pipe.run(source.frames()))
    await drain_then_advance(clock)
    await task

    report = pipe.metrics.report()
    latencies = [u.first_audio_out_ms for u in report.utterances]
    assert len(latencies) == 2
    assert all(latency is not None and latency > 0 for latency in latencies)
    # Turn 1: frame at 20 -> chunk at 50, audible at 50 => 50 ms from onset 0.
    # Turn 2: chunk ready at 110 but turn 1's 100 ms of audio presents until
    # 150 -> audible at 150 => 90 ms from onset 60 (includes playback queueing).
    assert latencies == [50, 90]
