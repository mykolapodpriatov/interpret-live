"""NllbMT adapter tests against a fake in-process translate worker.

No transformers/torch, no spawn: a scripted worker double stands in at the
model-process boundary, proving the non-repetition contract (only the current
segment is translated), cancellation semantics, preflight validation, and
typed error surfacing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from helpers import make_tokens
from interpret_live.backends.nllb import MtInputTooLongError, NllbMT
from interpret_live.types import Segment


class FakeMtWorker:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.cancel_signals = 0
        self.closed = False
        self.started = False
        self.hold: asyncio.Event | None = None
        self.result: Any = None

    async def start(self) -> None:
        self.started = True

    async def request(self, payload: dict[str, Any]) -> tuple[str, Any]:
        self.requests.append(payload)
        if self.hold is not None:
            await self.hold.wait()
        if self.result is not None:
            return self.result
        return ("ok", f"<{payload['text']}>")

    def signal_cancel(self) -> None:
        self.cancel_signals += 1

    def clear_cancel(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True


def _segment(text: str, index: int = 0, *, closed: bool = True) -> Segment:
    toks = make_tokens(text.split())
    return Segment(text=text, tokens=toks, token_span=(0, len(toks)), closed=closed, index=index)


async def test_translates_only_the_current_segment_ignoring_context() -> None:
    worker = FakeMtWorker()
    mt = NllbMT(worker=worker)
    seg1 = _segment("the weather is nice.", 0)
    seg2 = _segment("let us go outside.", 1)
    out1 = await mt.translate(seg1)
    # Segment 2 arrives with segment 1 as rolling context; the request payload
    # must contain ONLY segment 2's text, so TTS can never repeat segment 1.
    out2 = await mt.translate(seg2, context=("the", "weather", "is", "nice."))
    assert out1 == "<the weather is nice.>"
    assert out2 == "<let us go outside.>"
    assert worker.requests[1]["text"] == "let us go outside."
    assert "nice" not in worker.requests[1]["text"]


async def test_slow_generate_keeps_loop_responsive_and_cancels_cleanly() -> None:
    worker = FakeMtWorker()
    worker.hold = asyncio.Event()
    mt = NllbMT(worker=worker)

    heartbeat = 0

    async def beat() -> None:
        nonlocal heartbeat
        while True:
            heartbeat += 1
            await asyncio.sleep(0)

    beat_task = asyncio.create_task(beat())
    translate_task = asyncio.create_task(mt.translate(_segment("slow one.")))
    for _ in range(30):
        await asyncio.sleep(0)
    assert heartbeat > 10, "a slow generate must not block the event loop"
    assert not translate_task.done()

    translate_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await translate_task
    # The cooperative stop reached the worker; no stale value surfaced.
    assert worker.cancel_signals == 1
    beat_task.cancel()


async def test_worker_reported_cancellation_propagates_without_value() -> None:
    worker = FakeMtWorker()
    worker.result = ("cancelled", None)
    mt = NllbMT(worker=worker)
    with pytest.raises(asyncio.CancelledError):
        await mt.translate(_segment("x."))


async def test_worker_error_surfaces_typed() -> None:
    worker = FakeMtWorker()
    worker.result = ("error", "MtInputTooLongError: segment tokenizes to 999 tokens")
    mt = NllbMT(worker=worker)
    with pytest.raises(MtInputTooLongError, match="999"):
        await mt.translate(_segment("very long."))

    worker.result = ("error", "RuntimeError: cuda out of memory")
    with pytest.raises(RuntimeError, match="cuda out of memory"):
        await mt.translate(_segment("x."))


async def test_non_closed_segment_is_rejected() -> None:
    mt = NllbMT(worker=FakeMtWorker())
    with pytest.raises(AssertionError, match="closed"):
        await mt.translate(_segment("partial", closed=False))


async def test_start_and_aclose_manage_worker_lifecycle() -> None:
    worker = FakeMtWorker()
    mt = NllbMT(worker=worker)
    await mt.start()
    assert worker.started
    await mt.aclose()
    await mt.aclose()  # idempotent
    assert worker.closed
    assert worker.cancel_signals >= 1


def test_unsupported_language_fails_at_construction_before_any_device() -> None:
    with pytest.raises(ValueError, match="not a supported NLLB language"):
        NllbMT(source_lang="xx", worker=FakeMtWorker())
    with pytest.raises(ValueError, match="not a supported NLLB language"):
        NllbMT(target_lang="klingon", worker=FakeMtWorker())


def test_configuration_validation_fails_fast() -> None:
    with pytest.raises(ValueError, match="dtype"):
        NllbMT(dtype="int8", worker=FakeMtWorker())
    with pytest.raises(ValueError, match="device"):
        NllbMT(device="tpu", worker=FakeMtWorker())
    with pytest.raises(ValueError, match="max_input_tokens"):
        NllbMT(max_input_tokens=0, worker=FakeMtWorker())
    with pytest.raises(ValueError, match="max_new_tokens"):
        NllbMT(max_new_tokens=0, worker=FakeMtWorker())
    with pytest.raises(ValueError, match="model_name"):
        NllbMT(model_name="", worker=FakeMtWorker())
