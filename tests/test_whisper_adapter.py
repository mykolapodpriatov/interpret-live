"""WhisperSTT adapter tests against a fake in-process decode worker.

No faster-whisper, no spawn: a scripted worker double stands in at the
model-process boundary, so endpoint-driven decode pacing, full-prefix
hypotheses, turn ordering, latest-wins partials, cancellation, and overrun
surfacing are all deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from interpret_live.backends.whisper import SttOverrunError, SttStreamError, WhisperSTT
from interpret_live.types import AudioFrame, Hypothesis

_RATE = 16000


class FakeDecodeWorker:
    """Scripted stand-in for the model-process boundary.

    Decodes deterministically: one token per 100 ms of buffered audio, so a
    longer buffer always extends (never rewrites) the token prefix. ``hold``
    can be armed to keep one request "in flight" while ingestion continues.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.cancel_signals = 0
        self.clear_signals = 0
        self.closed = False
        self.started = False
        self.hold: asyncio.Event | None = None
        self.hold_matcher: Any = None
        self.cancelled_after_hold = False
        self.results_override: Any = None

    async def start(self) -> None:
        self.started = True

    async def request(self, payload: dict[str, Any]) -> tuple[str, Any]:
        self.requests.append(payload)
        if self.hold is not None and (self.hold_matcher is None or self.hold_matcher(payload)):
            await self.hold.wait()
            if self.cancelled_after_hold:
                return ("cancelled", None)
        if self.results_override is not None:
            return self.results_override(payload)
        n_samples = len(payload["pcm"]) // 4
        n_tokens = n_samples // (_RATE // 10)  # one word per 100 ms
        tokens = [(f"w{i}", i * 100, (i + 1) * 100) for i in range(n_tokens)]
        return ("ok", tokens)

    def signal_cancel(self) -> None:
        self.cancel_signals += 1

    def clear_cancel(self) -> None:
        self.clear_signals += 1

    async def aclose(self) -> None:
        self.closed = True


def _adapter(worker: FakeDecodeWorker, **overrides: Any) -> WhisperSTT:
    kwargs: dict[str, Any] = {
        "vad_threshold": 0.02,
        "vad_hangover_ms": 0,
        "pre_roll_ms": 40,
        "partial_interval_ms": 100,
        "end_silence_ms": 100,
        "max_utterance_ms": 30_000,
        "worker": worker,
    }
    kwargs.update(overrides)
    return WhisperSTT(**kwargs)


def _frame(amp: float, t_ms: int, *, rate: int = _RATE, ms: int = 20) -> AudioFrame:
    n = int(ms * rate / 1000)
    return AudioFrame(samples=np.full(n, amp, dtype=np.float32), sample_rate=rate, t_ms=t_ms)


async def _feed(pattern: str, *, rate: int = _RATE) -> AsyncIterator[AudioFrame]:
    for i, ch in enumerate(pattern):
        yield _frame(0.5 if ch == "s" else 0.0, i * 20, rate=rate)
        await asyncio.sleep(0)


async def _collect(stt: WhisperSTT, pattern: str, *, rate: int = _RATE) -> list[Hypothesis]:
    return [h async for h in stt.stream(_feed(pattern, rate=rate))]


async def test_two_utterances_two_finals_with_turn_ids_and_onsets() -> None:
    worker = FakeDecodeWorker()
    stt = _adapter(worker)
    # speech 300ms, silence 200ms, speech 300ms, trailing silence.
    hyps = await _collect(stt, "sssssssssssssss..........sssssssssssssss..........")
    finals = [h for h in hyps if h.is_final]
    assert len(finals) == 2, "one final per detected utterance"
    assert finals[0].source_turn_id == "turn-1"
    assert finals[1].source_turn_id == "turn-2"
    # Onset = first VAD-positive frame timestamp of each utterance.
    assert finals[0].speech_started_at_ms == 0
    assert finals[1].speech_started_at_ms == 500
    # Every hypothesis of a turn repeats the same onset/turn id.
    for h in hyps:
        assert h.source_turn_id in {"turn-1", "turn-2"}
        assert h.speech_started_at_ms in {0, 500}
    await stt.aclose()
    assert worker.closed and worker.cancel_signals >= 1


async def test_partials_are_full_prefix_from_index_zero() -> None:
    worker = FakeDecodeWorker()
    stt = _adapter(worker)
    hyps = await _collect(stt, "s" * 30 + "." * 10)  # 600 ms speech
    assert hyps, "expected at least one hypothesis"
    texts = [[t.text for t in h.tokens] for h in hyps]
    for words in texts:
        # Deterministic fake decode: token i is always "wi" — a full prefix
        # from index zero, never a cropped tail.
        assert words == [f"w{i}" for i in range(len(words))]
    # Prefixes strictly grow to the final.
    lengths = [len(w) for w in texts]
    assert lengths == sorted(lengths)
    assert hyps[-1].is_final
    # Word timestamps stay utterance-relative (start at 0).
    assert hyps[-1].tokens[0].start_ms == 0


async def test_ingestion_continues_during_slow_final_and_order_is_kept() -> None:
    worker = FakeDecodeWorker()
    worker.hold = asyncio.Event()
    worker.hold_matcher = lambda p: p["final"] and p["turn"] == "turn-1"
    stt = _adapter(worker)

    heartbeat = 0

    async def beat() -> None:
        nonlocal heartbeat
        while True:
            heartbeat += 1
            await asyncio.sleep(0)

    beat_task = asyncio.create_task(beat())
    hyps: list[Hypothesis] = []

    async def consume() -> None:
        # Two utterances; the first final decode is held while turn-2's audio
        # keeps flowing (ingestion must continue buffering).
        async for h in stt.stream(_feed("ssssssssss......ssssssssss..........")):
            hyps.append(h)

    consumer = asyncio.create_task(consume())
    for _ in range(200):
        await asyncio.sleep(0)
        if any(r["final"] for r in worker.requests):
            break
    heartbeat_before = heartbeat
    for _ in range(50):
        await asyncio.sleep(0)
    assert heartbeat > heartbeat_before, "slow decode must not block the loop"
    worker.hold.set()
    await consumer
    beat_task.cancel()

    finals = [h for h in hyps if h.is_final]
    assert [f.source_turn_id for f in finals] == ["turn-1", "turn-2"]
    # The held final reached the pipeline before any turn-2 hypothesis.
    turn2_first = next(i for i, h in enumerate(hyps) if h.source_turn_id == "turn-2")
    turn1_final = next(i for i, h in enumerate(hyps) if h.is_final)
    assert turn1_final < turn2_first
    await stt.aclose()


async def test_latest_wins_partial_never_queues_more_than_one() -> None:
    worker = FakeDecodeWorker()
    worker.hold = asyncio.Event()
    first_partial_seen = asyncio.Event()

    def matcher(payload: dict[str, Any]) -> bool:
        if not payload["final"] and not first_partial_seen.is_set():
            first_partial_seen.set()
            return True
        return False

    worker.hold_matcher = matcher
    stt = _adapter(worker)
    # 1s of speech: with 100 ms partial cadence there would be ~10 partial
    # requests; while the first is held, later ones must collapse to one.
    collect = asyncio.create_task(_collect(stt, "s" * 50 + "." * 10))
    for _ in range(400):
        await asyncio.sleep(0)
        if first_partial_seen.is_set():
            break
    # Let ingestion finish producing all partial ticks while the decode holds.
    for _ in range(400):
        await asyncio.sleep(0)
    worker.hold.set()
    hyps = await collect
    partial_requests = [r for r in worker.requests if not r["final"]]
    # First (held) partial + at most one collapsed latest-wins partial...
    # which itself is replaced by the final if still queued at endpoint.
    assert len(partial_requests) <= 2, partial_requests
    assert [h for h in hyps if h.is_final], "the final must still be decoded"
    await stt.aclose()


async def test_hold_released_by_cancellation_yields_no_stale_hypothesis() -> None:
    worker = FakeDecodeWorker()
    worker.hold = asyncio.Event()
    worker.hold_matcher = lambda p: p["final"]
    worker.cancelled_after_hold = True
    stt = _adapter(worker)
    collect = asyncio.create_task(_collect(stt, "ssssssssss.........."))
    for _ in range(400):
        await asyncio.sleep(0)
        if any(r["final"] for r in worker.requests):
            break
    worker.hold.set()  # the "decode" resolves as cancelled
    hyps = await collect
    finals = [h for h in hyps if h.is_final]
    assert finals == [], "a cancelled decode must never surface a stale final"
    await stt.aclose()


async def test_resamples_source_to_16k_with_one_stateful_resampler() -> None:
    pytest.importorskip("soxr")
    worker = FakeDecodeWorker()
    stt = _adapter(worker)
    hyps = await _collect(stt, "s" * 25 + "." * 10, rate=32000)
    final_req = next(r for r in worker.requests if r["final"])
    n_samples = len(final_req["pcm"]) // 4
    # 25 speech frames + buffered trailing silence at 32 kHz, downsampled 2x:
    # duration is preserved (~500-600 ms of 16 kHz audio, not 32 kHz counts).
    assert 7000 <= n_samples <= 11000, n_samples
    assert [h for h in hyps if h.is_final]
    await stt.aclose()


async def test_mid_stream_rate_change_raises_typed_error() -> None:
    worker = FakeDecodeWorker()
    stt = _adapter(worker)

    async def frames() -> AsyncIterator[AudioFrame]:
        yield _frame(0.5, 0, rate=16000)
        yield _frame(0.5, 20, rate=48000)

    with pytest.raises(SttStreamError, match="sample rate changed"):
        async for _ in stt.stream(frames()):
            pass
    await stt.aclose()


async def test_final_backlog_over_bound_surfaces_overrun_error() -> None:
    worker = FakeDecodeWorker()
    worker.hold = asyncio.Event()  # hold everything: the queue can only grow
    stt = _adapter(worker, max_pending_turns=1)
    with pytest.raises(SttOverrunError):
        # Three quick utterances -> three finals; bound is 1.
        await _collect(stt, "ssssss......ssssss......ssssss......")
    worker.hold.set()
    await stt.aclose()


async def test_silence_only_stream_emits_nothing() -> None:
    worker = FakeDecodeWorker()
    stt = _adapter(worker)
    hyps = await _collect(stt, "." * 30)
    assert hyps == []
    assert worker.requests == []
    await stt.aclose()


async def test_empty_decode_result_skips_turn_without_final() -> None:
    worker = FakeDecodeWorker()
    worker.results_override = lambda payload: ("ok", [])
    stt = _adapter(worker)
    hyps = await _collect(stt, "ssssssssss..........")
    assert hyps == [], "empty/silence-only decode must not emit hypotheses"
    await stt.aclose()


async def test_sustained_speech_splits_at_max_utterance_into_ordered_turns() -> None:
    worker = FakeDecodeWorker()
    stt = _adapter(worker, max_utterance_ms=200)
    hyps = await _collect(stt, "s" * 30 + "." * 10)  # 600 ms sustained speech
    finals = [h for h in hyps if h.is_final]
    assert len(finals) >= 2, "sustained speech must split into adjacent turns"
    turn_ids = [f.source_turn_id for f in finals]
    assert turn_ids == sorted(turn_ids, key=lambda t: int(str(t).split("-")[1]))
    await stt.aclose()


def test_configuration_validation_fails_fast() -> None:
    with pytest.raises(ValueError, match="device"):
        WhisperSTT(device="gpu", worker=FakeDecodeWorker())
    with pytest.raises(ValueError, match="compute_type"):
        WhisperSTT(compute_type="int4", worker=FakeDecodeWorker())
    with pytest.raises(ValueError, match="language"):
        WhisperSTT(language="", worker=FakeDecodeWorker())
    with pytest.raises(ValueError, match="model_size"):
        WhisperSTT(model_size="", worker=FakeDecodeWorker())
