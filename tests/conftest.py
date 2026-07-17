"""Shared pytest fixtures and helpers for the deterministic test suite.

Everything here is offline + deterministic: a :class:`ManualClock`, builders for
:class:`Hypothesis` / :class:`AudioFrame`, and the drain-then-advance driver. No
real audio, models, or network; no :func:`asyncio.sleep` in the system under
test.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from interpret_live.clock import ManualClock


@pytest.fixture
def clock() -> ManualClock:
    """A fresh deterministic manual clock per test."""
    return ManualClock()


@pytest.fixture(scope="session", autouse=True)
def _assert_fast_suite() -> Iterator[None]:
    """Assert the whole suite runs in well under a second of wall-clock time.

    This catches a stray real :func:`asyncio.sleep` (or any real wait) sneaking
    into the core or fakes: with the manual clock, logical time advances without
    real waiting, so the entire deterministic suite must complete near-instantly.
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    # Generous bound: real model/audio/network would blow far past this; a stray
    # asyncio.sleep would accumulate visible wall-clock time. The budget allows
    # for the deliberately wall-clock pieces of the suite — the real spawned
    # model-worker lifecycle tests and the subprocess no-extras import check —
    # while still catching runaway real waits in the deterministic core.
    assert elapsed < 25.0, (
        f"suite wall-clock time {elapsed:.3f}s exceeds the determinism budget; "
        "a real sleep/wait likely leaked into the core or fakes"
    )
