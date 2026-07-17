"""Playback-contract tests against the deterministic :class:`FakeAudioSink`.

The fake models presentation time on the injected :class:`ManualClock`, so the
generation-scoped contract — bounded-lookahead scheduling, started/completed
receipts, partial-stop snapshots, typed rejection of stale schedules, and
single-generation sink ownership — is exercised without hardware.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from interpret_live.audio_io import FakeAudioSink
from interpret_live.clock import ManualClock, drain_then_advance
from interpret_live.types import PlaybackRejectedError, TtsChunk


def _chunk(
    ms: int,
    *,
    uid: str = "u1",
    seg: int = 0,
    rate: int = 16000,
    final: bool = False,
    amp: float = 0.1,
) -> TtsChunk:
    n = int(ms * rate / 1000)
    return TtsChunk(
        samples=np.full(n, amp, dtype=np.float32),
        sample_rate=rate,
        segment_index=seg,
        utterance_id=uid,
        final=final,
    )


async def test_receipts_report_full_presentation(clock: ManualClock) -> None:
    sink = FakeAudioSink(clock=clock)
    gen = sink.new_generation()

    async def scenario() -> tuple:
        h1 = await sink.schedule(gen, _chunk(100, seg=0))
        h2 = await sink.schedule(gen, _chunk(50, seg=1, final=True))
        started1 = await h1.started()
        done1 = await h1.completed()
        started2 = await h2.started()
        done2 = await h2.completed()
        await sink.drain()
        return started1, done1, started2, done2

    task = asyncio.ensure_future(scenario())
    await drain_then_advance(clock)
    started1, done1, started2, done2 = await task

    assert started1.first_audible_t_ms == 0
    assert done1.completed and not done1.interrupted
    assert done1.source_samples_presented == done1.source_samples_total == 1600
    # Chunk 2 becomes audible exactly when chunk 1 ends: gapless, sequential.
    assert started2.first_audible_t_ms == 100
    assert done2.source_samples_presented == 800
    assert [c.segment_index for c in sink.played] == [0, 1]


async def test_stop_mid_playback_returns_partial_dac_passed_snapshot(
    clock: ManualClock,
) -> None:
    sink = FakeAudioSink(clock=clock)
    gen = sink.new_generation()

    async def scenario() -> tuple:
        handle = await sink.schedule(gen, _chunk(200))
        await handle.started()
        # Let 75 ms of audio become audible, then stop.
        await clock.sleep(75)
        snapshots = await sink.stop(gen)
        receipt = await handle.completed()
        return snapshots, receipt

    task = asyncio.ensure_future(scenario())
    await drain_then_advance(clock)
    snapshots, receipt = await task

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.interrupted and snap.completed
    # Only the samples whose presentation time has passed: 75 ms @ 16 kHz.
    assert snap.source_samples_presented == 1200
    assert snap.source_samples_total == 3200
    assert receipt.interrupted
    assert sink.stopped_at_ms == 75
    assert sink.stop_count == 1


async def test_blocked_schedule_is_rejected_on_stop_and_never_enqueues(
    clock: ManualClock,
) -> None:
    sink = FakeAudioSink(clock=clock, capacity=1)
    gen = sink.new_generation()
    outcome: dict[str, object] = {}

    async def scenario() -> None:
        await sink.schedule(gen, _chunk(500))

        async def blocked() -> None:
            try:
                await sink.schedule(gen, _chunk(100, seg=99))
            except PlaybackRejectedError:
                outcome["rejected"] = True

        blocked_task = asyncio.create_task(blocked())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not blocked_task.done(), "schedule must block on full capacity"
        await sink.stop(gen)  # invalidates first, then wakes the waiter
        await blocked_task

    task = asyncio.ensure_future(scenario())
    await drain_then_advance(clock)
    await task

    assert outcome.get("rejected") is True
    # The stale chunk never became audible even though capacity was released.
    assert all(c.segment_index != 99 for c in sink.played)


async def test_schedule_on_already_stopped_generation_rejects_immediately(
    clock: ManualClock,
) -> None:
    sink = FakeAudioSink(clock=clock)
    gen = sink.new_generation()

    async def scenario() -> None:
        await sink.stop(gen)
        with pytest.raises(PlaybackRejectedError):
            await sink.schedule(gen, _chunk(10))

    task = asyncio.ensure_future(scenario())
    await drain_then_advance(clock)
    await task


async def test_new_generation_waits_for_previous_and_survives_its_stop(
    clock: ManualClock,
) -> None:
    sink = FakeAudioSink(clock=clock)
    old = sink.new_generation()
    new = sink.new_generation()
    order: list[str] = []

    async def scenario() -> None:
        await sink.schedule(old, _chunk(500, uid="old"))

        async def new_turn() -> None:
            await sink.schedule(new, _chunk(50, uid="new", seg=1))
            order.append("new-scheduled")

        new_task = asyncio.create_task(new_turn())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not new_task.done(), "next generation must wait for the previous"
        order.append("stopping-old")
        await sink.stop(old)
        await new_task
        await sink.drain()

    task = asyncio.ensure_future(scenario())
    await drain_then_advance(clock)
    await task

    assert order == ["stopping-old", "new-scheduled"]
    # Stopping the old generation never erased the accepted new-turn audio.
    assert [c.utterance_id for c in sink.played if c.utterance_id == "new"] == ["new"]


async def test_drain_waits_for_final_presentation(clock: ManualClock) -> None:
    sink = FakeAudioSink(clock=clock)
    gen = sink.new_generation()
    done_at: dict[str, int] = {}

    async def scenario() -> None:
        await sink.schedule(gen, _chunk(120))
        await sink.drain()
        done_at["t"] = clock.now_ms()

    task = asyncio.ensure_future(scenario())
    await drain_then_advance(clock)
    await task

    assert done_at["t"] == 120  # drain returned only after the audio finished


async def test_aclose_is_idempotent_and_releases_everything(clock: ManualClock) -> None:
    sink = FakeAudioSink(clock=clock)
    gen = sink.new_generation()

    async def scenario() -> None:
        handle = await sink.schedule(gen, _chunk(1000))
        await handle.started()
        await sink.aclose()
        await sink.aclose()  # idempotent
        receipt = await handle.completed()
        assert receipt.interrupted

    task = asyncio.ensure_future(scenario())
    await drain_then_advance(clock)
    await task
