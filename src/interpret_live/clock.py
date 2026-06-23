"""Injected async :class:`Clock` abstraction for deterministic asyncio tests.

The whole streaming core takes a :class:`Clock` and uses ``clock.sleep()`` /
``clock.now_ms()`` instead of :func:`asyncio.sleep` / wall time. In production a
:class:`RealClock` delegates to the event loop; in tests a :class:`ManualClock`
makes time advance explicitly via the **drain-then-advance** protocol, so the
suite is reproducible and runs in well under a wall-clock second.

**``asyncio.sleep()`` is forbidden in core + fake modules** — they must call
``clock.sleep()``. ``test_pipeline``/``test_no_real_sleep`` assert this by
bounding total wall-clock suite time.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "ManualClock", "RealClock", "drain_then_advance"]


@runtime_checkable
class Clock(Protocol):
    """An injected source of time + delay for the async core."""

    def now_ms(self) -> int:
        """Return the current logical time in milliseconds."""
        ...

    async def sleep(self, ms: int) -> None:
        """Suspend the calling task for ``ms`` logical milliseconds."""
        ...


class RealClock:
    """A :class:`Clock` backed by the running event loop (production use)."""

    __slots__ = ("_epoch",)

    def __init__(self) -> None:
        self._epoch = asyncio.get_event_loop().time()

    def now_ms(self) -> int:
        return round((asyncio.get_event_loop().time() - self._epoch) * 1000)

    async def sleep(self, ms: int) -> None:
        if ms > 0:
            await asyncio.sleep(ms / 1000)


class ManualClock:
    """A deterministic clock whose time only moves when explicitly advanced.

    A ``clock.sleep(ms)`` registers a wakeup at ``now_ms() + ms`` and suspends on
    an :class:`asyncio.Event`. Time never moves on its own; the test harness
    calls :func:`drain_then_advance` (or :meth:`advance`) to run all ready work,
    then jump to the next scheduled wakeup. This avoids the classic asyncio
    deadlock where a consumer blocked on a timer and a producer blocked on a full
    queue never make progress.
    """

    __slots__ = ("_counter", "_now", "_waiters")

    def __init__(self, start_ms: int = 0) -> None:
        self._now = start_ms
        # Min-heap of (due_ms, seq, event). ``seq`` keeps it a stable total order
        # and keeps Events out of the heap comparison.
        self._waiters: list[tuple[int, int, asyncio.Event]] = []
        self._counter = itertools.count()

    def now_ms(self) -> int:
        return self._now

    async def sleep(self, ms: int) -> None:
        if ms <= 0:
            # A zero/negative sleep still yields control once, deterministically.
            await asyncio.sleep(0)
            return
        due = self._now + ms
        event = asyncio.Event()
        heapq.heappush(self._waiters, (due, next(self._counter), event))
        await event.wait()

    def next_wakeup_ms(self) -> int | None:
        """Return the soonest pending wakeup time, or ``None`` if none pending."""
        if not self._waiters:
            return None
        return self._waiters[0][0]

    def advance(self, to_ms: int) -> int:
        """Advance logical time to ``to_ms`` and fire every wakeup due by then.

        Returns the number of waiters released. Does not itself yield to the
        loop; the harness drives ``await asyncio.sleep(0)`` between advances.
        """
        if to_ms < self._now:
            raise ValueError(f"cannot move clock backwards: {to_ms} < {self._now}")
        self._now = to_ms
        released = 0
        while self._waiters and self._waiters[0][0] <= self._now:
            _due, _seq, event = heapq.heappop(self._waiters)
            event.set()
            released += 1
        return released

    @property
    def pending(self) -> int:
        """Number of tasks currently asleep on this clock."""
        return len(self._waiters)


# Consecutive stable drain rounds required before the clock is allowed to
# advance. Chained ``loop.call_soon`` callbacks (e.g. the wakeup posted by
# ``asyncio.wait`` when a child task completes) need a few extra yields to fully
# propagate; requiring several identical snapshots in a row guarantees every such
# callback has run before we treat the system as quiescent and advance time.
_STABILITY_ROUNDS = 4


async def drain_then_advance(
    clock: ManualClock,
    *,
    max_steps: int = 100_000,
) -> None:
    """Run the manual-clock event pump until the system is fully quiescent.

    The protocol, per the determinism strategy:

    1. ``await asyncio.sleep(0)`` repeatedly so every ready coroutine (and every
       chained ``call_soon`` callback) runs, until a **fixed point** is reached:
       several consecutive yields leave the set of parked tasks unchanged.
    2. If any task is asleep on the clock, **advance** to the soonest registered
       wakeup and go back to step 1.
    3. Stop when nothing is runnable and no timers remain.

    Because internal queues are bounded, a producer blocked on a full queue is
    resolved by first draining the consumer's non-timer work (step 1); time only
    advances (step 2) when nothing else can run, guaranteeing forward progress.

    The fixed-point (rather than a single quiescence check) is essential:
    ``asyncio.wait``/``gather`` wake their parent via a ``call_soon`` callback
    that lands one tick *after* a child completes, so a naive single check could
    advance the clock while a wakeup is still queued, stranding the parent.

    Args:
        clock: The manual clock driving the system under test.
        max_steps: Safety bound to surface a genuine deadlock as a clear error
            instead of an infinite loop during development.
    """
    for _ in range(max_steps):
        # Step 1: drive the loop to a stable fixed point.
        stable = 0
        prev = _snapshot()
        for _ in range(max_steps):
            await asyncio.sleep(0)
            snap = _snapshot()
            if snap == prev:
                stable += 1
                if stable >= _STABILITY_ROUNDS:
                    break
            else:
                stable = 0
                prev = snap
        else:  # pragma: no cover - safety valve
            raise RuntimeError("drain_then_advance: loop never reached a fixed point")

        # Step 2: nothing runnable; advance to the next timer if any.
        nxt = clock.next_wakeup_ms()
        if nxt is None:
            return
        clock.advance(nxt)
    raise RuntimeError(  # pragma: no cover - safety valve
        "drain_then_advance: exceeded max_steps; suspected deadlock"
    )


def _snapshot() -> frozenset[tuple[int, int]]:
    """Snapshot every live non-current task as ``(task_id, parked_future_id)``.

    The snapshot changes whenever a task is created, completes, or transitions
    the future it awaits — i.e. whenever any real work happened on the last
    yield. Two identical consecutive snapshots therefore mean no progress was
    made, which (repeated ``_STABILITY_ROUNDS`` times) is our quiescence signal.
    """
    current = asyncio.current_task()
    items: set[tuple[int, int]] = set()
    for task in asyncio.all_tasks():
        if task is current or task.done():
            continue
        waiter = getattr(task, "_fut_waiter", None)
        # ``_fut_waiter`` is the future a task is currently awaiting; it is a
        # private attribute but stable across CPython 3.11–3.13. ``id(None)`` is
        # a constant, so a runnable (not-yet-parked) task contributes a distinct,
        # changing entry as it progresses.
        items.add((id(task), id(waiter)))
    return frozenset(items)
