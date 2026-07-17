"""Real spawned-worker lifecycle tests for :mod:`interpret_live.model_worker`.

These deliberately use wall-clock time (a real ``spawn`` child per worker):
readiness handshake, request/response, cooperative cancellation, startup
failure, error serialization, and the hard-reap path for a worker that ignores
cooperative shutdown forever. No child PID may survive ``aclose()``.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from interpret_live.model_worker import ModelWorker, WorkerError


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not expected for our children
        return True
    return True


async def test_echo_worker_round_trip_and_clean_close() -> None:
    worker = ModelWorker(
        "worker_handlers:echo_factory",
        {"prefix": "p-"},
        name="echo",
        ready_timeout_s=60.0,
        grace_s=2.0,
    )
    await worker.start()
    pid = worker.pid
    assert pid is not None
    assert await worker.request("hello") == ("ok", "p-hello")
    assert await worker.request("again") == ("ok", "p-again")
    await worker.aclose()
    assert worker.pid is None
    assert not _pid_alive(pid), "child PID must not survive aclose()"


async def test_cooperative_cancellation_returns_cancelled_status() -> None:
    worker = ModelWorker(
        "worker_handlers:slow_factory",
        {"seconds": 30.0},
        name="slow",
        ready_timeout_s=60.0,
        grace_s=2.0,
    )
    await worker.start()
    request = asyncio.create_task(worker.request("x"))
    await asyncio.sleep(0.2)  # let the handler enter its loop
    worker.signal_cancel()
    status, _ = await request
    assert status == "cancelled"
    await worker.aclose()


async def test_stuck_worker_is_hard_reaped_within_budget() -> None:
    worker = ModelWorker(
        "worker_handlers:stuck_factory",
        {},
        name="stuck",
        ready_timeout_s=60.0,
        grace_s=0.5,
    )
    await worker.start()
    request = asyncio.create_task(worker.request("x"))
    await asyncio.sleep(0.2)  # the handler is now stuck forever
    pid = worker.pid
    assert pid is not None

    started = time.monotonic()
    await worker.aclose()
    elapsed = time.monotonic() - started
    # Cooperative join fails (0.5 s), terminate reaps it: well within budget.
    assert elapsed < 5.0, f"aclose took {elapsed:.1f}s; hard-reap budget blown"
    assert not _pid_alive(pid), "stuck child PID must not survive aclose()"
    with pytest.raises(WorkerError):
        await request


async def test_startup_failure_surfaces_as_worker_error() -> None:
    worker = ModelWorker(
        "worker_handlers:failing_startup_factory",
        {},
        name="bad-start",
        ready_timeout_s=60.0,
        grace_s=1.0,
    )
    with pytest.raises(WorkerError, match="boom at load"):
        await worker.start()


async def test_handler_exception_is_serialized_not_fatal() -> None:
    worker = ModelWorker(
        "worker_handlers:error_factory",
        {},
        name="err",
        ready_timeout_s=60.0,
        grace_s=2.0,
    )
    await worker.start()
    status, value = await worker.request("boom")
    assert status == "error"
    assert "ValueError" in value and "bad payload" in value
    # The worker survives a handler error and keeps serving.
    status2, _ = await worker.request("boom2")
    assert status2 == "error"
    await worker.aclose()


async def test_aclose_is_idempotent_and_rejects_new_requests() -> None:
    worker = ModelWorker(
        "worker_handlers:echo_factory",
        {},
        name="closed",
        ready_timeout_s=60.0,
        grace_s=2.0,
    )
    await worker.start()
    await worker.aclose()
    await worker.aclose()
    with pytest.raises(WorkerError, match="closed"):
        await worker.request("x")
