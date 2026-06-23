"""Manual clock + drain-then-advance + no-real-sleep determinism tests.

Asserts the manual clock's monotonicity and the drain-then-advance fixed-point
protocol, and that the deterministic core/fakes never use real time (a stray
``asyncio.sleep`` would make a timed block take real wall-clock time).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from interpret_live.clock import ManualClock, drain_then_advance


def test_manual_clock_starts_at_zero_and_is_monotonic() -> None:
    clock = ManualClock()
    assert clock.now_ms() == 0
    clock.advance(100)
    assert clock.now_ms() == 100
    with pytest.raises(ValueError, match="cannot move clock backwards"):
        clock.advance(50)


async def test_clock_sleep_blocks_until_advanced() -> None:
    clock = ManualClock()
    woke_at: list[int] = []

    async def worker() -> None:
        await clock.sleep(250)
        woke_at.append(clock.now_ms())

    t = asyncio.ensure_future(worker())
    await drain_then_advance(clock)
    await t
    assert woke_at == [250]


async def test_concurrent_sleepers_wake_in_time_order() -> None:
    clock = ManualClock()
    order: list[int] = []

    async def worker(ms: int) -> None:
        await clock.sleep(ms)
        order.append(ms)

    tasks = [asyncio.ensure_future(worker(ms)) for ms in (300, 100, 200)]
    await drain_then_advance(clock)
    await asyncio.gather(*tasks)
    assert order == [100, 200, 300]


async def test_zero_sleep_yields_without_advancing_time() -> None:
    clock = ManualClock()
    await clock.sleep(0)
    assert clock.now_ms() == 0


async def test_drain_advances_through_chained_sleeps() -> None:
    clock = ManualClock()
    stamps: list[int] = []

    async def chain() -> None:
        for _ in range(3):
            await clock.sleep(100)
            stamps.append(clock.now_ms())

    t = asyncio.ensure_future(chain())
    await drain_then_advance(clock)
    await t
    assert stamps == [100, 200, 300]


def test_suite_uses_no_real_time() -> None:
    """A timed clock block must take effectively no real wall-clock time.

    If a real ``asyncio.sleep`` leaked into the clock, advancing logical time by
    seconds would take seconds of real time. Here we drive a 10-second *logical*
    sleep and assert it completes near-instantly in real time.
    """

    async def main() -> None:
        clock = ManualClock()

        async def worker() -> None:
            await clock.sleep(10_000)  # 10 logical seconds

        t = asyncio.ensure_future(worker())
        await drain_then_advance(clock)
        await t

    start = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"logical 10s took {elapsed:.3f}s real time (real sleep leaked?)"
