"""PiperTTS adapter tests against a fake in-process synthesis worker.

No piper/onnx, no spawn: a scripted worker double stands in at the
model-process boundary, proving incremental block streaming with one-chunk
lookahead (exactly one final block), native-rate metadata, PCM16→float32
conversion, and cancellation that stops advancing the generator.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from interpret_live.audio_codec import float32_to_pcm16
from interpret_live.backends.piper import PiperTTS, TtsVoiceError
from interpret_live.types import TtsChunk


class FakePiperWorker:
    """Scripted block server: 'start' arms a block list, 'next' pops one."""

    def __init__(self, blocks: list[bytes], *, rate: int = 22050) -> None:
        self.script = blocks
        self.rate = rate
        self.requests: list[dict[str, Any]] = []
        self.cancel_signals = 0
        self.closed = False
        self.started = False
        self.hold_next: asyncio.Event | None = None
        self.start_error: str | None = None
        self._remaining: list[bytes] = []

    async def start(self) -> None:
        self.started = True

    async def request(self, payload: dict[str, Any]) -> tuple[str, Any]:
        self.requests.append(payload)
        op = payload["op"]
        if op == "start":
            if self.start_error is not None:
                return ("error", self.start_error)
            self._remaining = list(self.script)
            return ("ok", {"rate": self.rate})
        if op == "next":
            if self.hold_next is not None:
                await self.hold_next.wait()
            if not self._remaining:
                return ("ok", {"end": True})
            return ("ok", {"end": False, "pcm": self._remaining.pop(0)})
        return ("ok", {})

    def signal_cancel(self) -> None:
        self.cancel_signals += 1

    def clear_cancel(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True


def _pcm_block(value: float, n: int = 220) -> bytes:
    return float32_to_pcm16(np.full(n, value, dtype=np.float32))


async def _collect(tts: PiperTTS, text: str = "hola.") -> list[TtsChunk]:
    return [c async for c in tts.synthesize(text, segment_index=0, utterance_id="u1")]


async def test_streams_blocks_incrementally_before_synthesis_finishes() -> None:
    worker = FakePiperWorker([_pcm_block(0.1), _pcm_block(0.2), _pcm_block(0.3)])
    tts = PiperTTS(model_path="voice.onnx", worker=worker)
    agen = tts.synthesize("hola amigo.", segment_index=0, utterance_id="u1")
    first = await anext(agen)
    # The first block is yielded while the remaining blocks have not been
    # produced yet: at most start + two NEXT steps (one-block lookahead) —
    # never a list() materialization of the whole utterance.
    next_ops = [r for r in worker.requests if r["op"] == "next"]
    assert len(next_ops) == 2
    assert worker._remaining, "later blocks must still be pending in the worker"
    assert not first.final
    rest = [c async for c in agen]
    assert len(rest) == 2
    await tts.aclose()


async def test_exactly_one_final_block_including_single_block_case() -> None:
    worker = FakePiperWorker([_pcm_block(0.1)])
    tts = PiperTTS(model_path="voice.onnx", worker=worker)
    chunks = await _collect(tts)
    assert [c.final for c in chunks] == [True]

    worker3 = FakePiperWorker([_pcm_block(0.1), _pcm_block(0.2), _pcm_block(0.3)])
    tts3 = PiperTTS(model_path="voice.onnx", worker=worker3)
    chunks3 = await _collect(tts3)
    assert [c.final for c in chunks3] == [False, False, True]


async def test_sample_rate_comes_from_voice_metadata() -> None:
    worker = FakePiperWorker([_pcm_block(0.1)], rate=16000)
    tts = PiperTTS(model_path="voice.onnx", worker=worker)
    chunks = await _collect(tts)
    assert chunks[0].sample_rate == 16000  # not an assumed 22050


async def test_pcm16_blocks_convert_to_canonical_float32() -> None:
    worker = FakePiperWorker([_pcm_block(0.25, n=8)])
    tts = PiperTTS(model_path="voice.onnx", worker=worker)
    chunks = await _collect(tts)
    samples = chunks[0].samples
    assert samples.dtype == np.float32
    assert np.all(np.abs(samples - 0.25) < 1e-3)


async def test_cancel_after_first_block_stops_advancing_and_signals_worker() -> None:
    worker = FakePiperWorker([_pcm_block(0.1), _pcm_block(0.2), _pcm_block(0.3)])
    worker.hold_next = asyncio.Event()
    tts = PiperTTS(model_path="voice.onnx", worker=worker)

    received: list[TtsChunk] = []

    async def consume() -> None:
        async for chunk in tts.synthesize("hola.", segment_index=0, utterance_id="u1"):
            received.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()  # barge-in while a NEXT step is in flight
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker.cancel_signals == 1
    assert received == [], "no block of the cancelled utterance may be yielded"
    # No stray consumer/worker task remains.
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []
    await tts.aclose()


async def test_next_utterance_restarts_generator_in_child() -> None:
    worker = FakePiperWorker([_pcm_block(0.1)])
    tts = PiperTTS(model_path="voice.onnx", worker=worker)
    await _collect(tts, "one.")
    await _collect(tts, "two.")
    starts = [r for r in worker.requests if r["op"] == "start"]
    assert [s["text"] for s in starts] == ["one.", "two."]


async def test_zero_block_synthesis_yields_nothing() -> None:
    worker = FakePiperWorker([])
    tts = PiperTTS(model_path="voice.onnx", worker=worker)
    assert await _collect(tts) == []


async def test_start_error_surfaces_as_typed_voice_error() -> None:
    worker = FakePiperWorker([])
    worker.start_error = "TtsVoiceError: failed to load Piper voice 'x.onnx'"
    tts = PiperTTS(model_path="x.onnx", worker=worker)
    with pytest.raises(TtsVoiceError, match="failed to load"):
        await _collect(tts)


async def test_lifecycle_start_and_idempotent_aclose() -> None:
    worker = FakePiperWorker([])
    tts = PiperTTS(model_path="voice.onnx", worker=worker)
    await tts.start()
    assert worker.started
    await tts.aclose()
    await tts.aclose()
    assert worker.closed
    assert worker.cancel_signals >= 1


def test_empty_model_path_rejected() -> None:
    with pytest.raises(ValueError, match="model_path"):
        PiperTTS(model_path="", worker=FakePiperWorker([]))
