"""Spawn-safe handler factories for the real :class:`ModelWorker` tests.

Kept in an importable top-level module (pytest puts ``tests/`` on ``sys.path``,
which multiprocessing's ``spawn`` propagates to the child) so the worker child
can resolve them by dotted path.
"""

from __future__ import annotations

import time
from typing import Any

from interpret_live.model_worker import raise_if_cancelled


def echo_factory(prefix: str = "") -> Any:
    """A trivial handler: returns ``prefix + payload``."""

    def handle(payload: Any, cancel_event: Any) -> Any:
        return f"{prefix}{payload}"

    return handle


def slow_factory(seconds: float) -> Any:
    """A long-running handler that honors cooperative cancellation."""

    def handle(payload: Any, cancel_event: Any) -> Any:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            raise_if_cancelled(cancel_event)
            time.sleep(0.01)
        return "done"

    return handle


def stuck_factory() -> Any:
    """A handler that ignores cooperative shutdown forever (must be killed)."""

    def handle(payload: Any, cancel_event: Any) -> Any:
        while True:  # deliberately never checks cancel_event
            time.sleep(0.05)

    return handle


def failing_startup_factory() -> Any:
    """A factory that dies during model construction."""
    raise RuntimeError("boom at load")


def error_factory() -> Any:
    """A handler that raises on every request."""

    def handle(payload: Any, cancel_event: Any) -> Any:
        raise ValueError(f"bad payload: {payload!r}")

    return handle
