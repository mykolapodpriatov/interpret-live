"""Offline streaming TTS adapter backed by Piper (``piper`` extra).

Import-guarded: constructing :class:`PiperTTS` without the ``piper`` extra
raises a clear :class:`~interpret_live.backends.guard.MissingExtraError` (the
spawned worker shares this environment, so the parent-side import check is a
valid preflight).

Design (plan Task 4):

* ``PiperVoice`` is loaded — and its lazy ``synthesize_stream_raw()``
  generator advanced — only inside a dedicated spawned
  :class:`~interpret_live.model_worker.ModelWorker`. The parent sends one
  bounded ``NEXT`` request at a time and awaits one block result, which is
  natural downstream backpressure: no cross-thread producer queue, no model
  work on the event loop, and never a ``list()`` materialization.
* One-chunk lookahead: the adapter buffers at most one block so the final
  produced block alone carries ``TtsChunk.final=True`` (including the
  single-block case).
* The native output rate comes from the loaded voice's configuration — no
  independent ``sample_rate=22050`` assumption.
* Piper PCM16 is converted to the canonical float32 representation at the
  voice's native rate; the speaker sink owns any device-rate conversion.
* On coroutine cancellation/barge-in the worker's cooperative cancel signal is
  set, no further blocks are requested, and the child closes the stale
  generator when the next utterance starts. A native call that never returns
  is terminated and reaped by ``ModelWorker.aclose()``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np

from ..audio_codec import pcm16_to_float32
from ..model_worker import ModelWorker, raise_if_cancelled
from ..types import TtsChunk
from .guard import require

__all__ = ["PiperTTS", "build_piper_handler"]


class TtsVoiceError(RuntimeError):
    """A typed missing/corrupt Piper voice or synthesis failure."""


def build_piper_handler(*, model_path: str, config_path: str | None) -> Any:
    """Child-process factory: load the voice, return the block-step handler.

    The handler is stateful: ``start`` opens a fresh lazy synthesis generator
    (closing any stale one from a cancelled utterance), ``next`` advances it
    exactly one block, and ``stop`` closes it.
    """
    if not os.path.isfile(model_path):
        raise TtsVoiceError(f"Piper voice model not found: {model_path}")
    if config_path is not None and not os.path.isfile(config_path):
        raise TtsVoiceError(f"Piper voice config not found: {config_path}")

    from piper import PiperVoice

    try:
        voice = PiperVoice.load(model_path, config_path=config_path)
        rate = int(voice.config.sample_rate)
    except Exception as exc:
        raise TtsVoiceError(f"failed to load Piper voice {model_path!r}: {exc}") from exc
    if rate <= 0:
        raise TtsVoiceError(f"Piper voice {model_path!r} declares invalid rate {rate}")

    state: dict[str, Any] = {"gen": None}

    def _close_gen() -> None:
        gen = state["gen"]
        state["gen"] = None
        if gen is not None:
            close = getattr(gen, "close", None)
            if callable(close):
                close()

    def handle(payload: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
        raise_if_cancelled(cancel_event)
        op = payload["op"]
        if op == "start":
            _close_gen()  # a cancelled utterance's generator dies here
            state["gen"] = voice.synthesize_stream_raw(payload["text"])
            return {"rate": rate}
        if op == "next":
            gen = state["gen"]
            if gen is None:
                return {"end": True}
            try:
                raw = next(gen)
            except StopIteration:
                state["gen"] = None
                return {"end": True}
            return {"end": False, "pcm": bytes(raw)}
        if op == "stop":
            _close_gen()
            return {}
        raise ValueError(f"unknown Piper worker op: {op!r}")

    return handle


class TtsWorker(Protocol):
    """The worker surface :class:`PiperTTS` needs (injectable in tests)."""

    async def start(self) -> None:
        """Spawn/await voice readiness."""
        ...

    async def request(self, payload: Any) -> tuple[str, Any]:
        """Send one synthesis step; return ``(status, value)``."""
        ...

    def signal_cancel(self) -> None:
        """Set the cooperative cancel signal."""
        ...

    def clear_cancel(self) -> None:
        """Re-arm the cancel signal."""
        ...

    async def aclose(self) -> None:
        """Shut the worker down within its bounded budget."""
        ...


class PiperTTS:
    """Streaming TTS over a Piper voice model.

    Args:
        model_path: Path to a Piper ``.onnx`` voice model.
        config_path: Optional path to the voice's ``.json`` config.
        worker: Injectable synthesis worker (tests); defaults to a spawned
            :class:`~interpret_live.model_worker.ModelWorker` running
            :func:`build_piper_handler`.
        ready_timeout_s: Voice-load readiness budget for the default worker.
        grace_s: Per-stage shutdown budget for the default worker.
    """

    def __init__(
        self,
        *,
        model_path: str,
        config_path: str | None = None,
        worker: TtsWorker | None = None,
        ready_timeout_s: float = 60.0,
        grace_s: float = 2.0,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must be a non-empty Piper voice path")
        if worker is None:
            # The spawned child shares this interpreter's environment, so a
            # parent-side import check is a valid (and fail-fast) preflight;
            # file existence is validated for a clear pre-spawn error too.
            require("piper", backend="piper", extra="piper")
            if not os.path.isfile(model_path):
                raise TtsVoiceError(f"Piper voice model not found: {model_path}")
            if config_path is not None and not os.path.isfile(config_path):
                raise TtsVoiceError(f"Piper voice config not found: {config_path}")
            worker = ModelWorker(
                "interpret_live.backends.piper:build_piper_handler",
                {"model_path": model_path, "config_path": config_path},
                name="piper-worker",
                ready_timeout_s=ready_timeout_s,
                grace_s=grace_s,
            )
        self._worker: TtsWorker = worker
        self._closed = False

    async def start(self) -> None:
        """Start the voice worker and wait for readiness (preflight)."""
        await self._worker.start()

    async def aclose(self) -> None:
        """Cancel in-flight synthesis and reap the worker (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._worker.signal_cancel()
        await self._worker.aclose()

    def synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        """Return the streamed chunk iterator for ``text``."""
        return self._synthesize(text, segment_index=segment_index, utterance_id=utterance_id)

    async def _step(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            status, value = await self._worker.request(payload)
        except asyncio.CancelledError:
            # Barge-in: stop advancing; the worker's cooperative signal aborts
            # any in-flight native call, and the child closes the stale
            # generator on the next utterance's ``start``.
            self._worker.signal_cancel()
            raise
        if status == "cancelled":
            raise asyncio.CancelledError("synthesis was cancelled mid-utterance")
        if status == "error":
            raise TtsVoiceError(f"Piper synthesis failed: {value}")
        return dict(value)

    async def _synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        started = await self._step({"op": "start", "text": text})
        rate = int(started["rate"])
        pending: bytes | None = None
        while True:
            step = await self._step({"op": "next"})
            end = bool(step.get("end"))
            if pending is not None:
                # One-block lookahead: only the last produced block is final.
                yield self._chunk(pending, rate, segment_index, utterance_id, final=end)
            if end:
                break
            pcm = step.get("pcm")
            assert isinstance(pcm, bytes)
            pending = pcm

    @staticmethod
    def _chunk(
        pcm: bytes, rate: int, segment_index: int, utterance_id: str, *, final: bool
    ) -> TtsChunk:
        samples = np.clip(pcm16_to_float32(pcm), -1.0, 1.0)
        return TtsChunk(
            samples=samples,
            sample_rate=rate,
            segment_index=segment_index,
            utterance_id=utterance_id,
            final=final,
        )
