"""Offline streaming STT adapter backed by faster-whisper (``whisper`` extra).

Import-guarded: constructing :class:`WhisperSTT` without the ``whisper`` extra
raises a clear :class:`~interpret_live.backends.guard.MissingExtraError` (the
spawned worker uses the same environment, so a parent-side import check is a
valid preflight).

The adapter owns offline turn lifecycle (architecture decision 5):

* One stateful 16 kHz :class:`~interpret_live.audio_codec.StreamingResampler`
  converts the continuous source; it is flushed only at source EOF, never at
  utterance boundaries.
* A deterministic :class:`~interpret_live.vad.UtteranceEndpointDetector`
  (around :class:`~interpret_live.vad.EnergyVAD`) starts/ends turns; the
  adapter emits full-prefix partial :class:`~interpret_live.types.Hypothesis`
  values plus exactly one ``is_final=True`` per detected non-empty utterance
  while continuing to consume the same live source.
* ``WhisperModel`` is constructed inside a long-lived spawned
  :class:`~interpret_live.model_worker.ModelWorker`; decode requests are
  paced (no more than one per ``partial_interval_ms`` plus one final per
  turn) with a one-slot latest-wins queued-partial policy, so slow inference
  can never build an unbounded backlog or block the event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..audio_codec import StreamingResampler
from ..model_worker import ModelWorker, raise_if_cancelled
from ..types import AudioFrame, Hypothesis, Token
from ..vad import EnergyVAD, UtteranceEndpointDetector
from .guard import require

__all__ = ["WhisperSTT", "build_whisper_handler"]

_DECODE_RATE = 16000

_VALID_DEVICES = {"cpu", "cuda", "auto"}
_VALID_COMPUTE_TYPES = {
    "auto",
    "int8",
    "int8_float16",
    "int8_float32",
    "float16",
    "float32",
}


class SttOverrunError(RuntimeError):
    """Inference cannot keep up: too many finalized turns await decoding."""


class SttStreamError(RuntimeError):
    """A typed live-STT stream failure (e.g. mid-stream sample-rate change)."""


def build_whisper_handler(
    *,
    model_source: str,
    language: str,
    device: str,
    compute_type: str,
) -> Any:
    """Child-process factory: construct ``WhisperModel`` and return a handler.

    Runs inside the spawned worker. The returned handler decodes one full
    utterance buffer per request and checks the cooperative cancel event
    between lazily decoded segments, closing the generator on cancellation.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_source, device=device, compute_type=compute_type)

    def handle(payload: dict[str, Any], cancel_event: Any) -> list[tuple[str, int, int]]:
        raise_if_cancelled(cancel_event)
        audio = np.frombuffer(payload["pcm"], dtype=np.float32)
        segments, _info = model.transcribe(audio, language=language, word_timestamps=True)
        tokens: list[tuple[str, int, int]] = []
        try:
            for seg in segments:
                # Cooperative cancellation between returned segments.
                raise_if_cancelled(cancel_event)
                for word in getattr(seg, "words", None) or []:
                    text = str(word.word).strip()
                    if text:
                        tokens.append((text, int(word.start * 1000), int(word.end * 1000)))
        finally:
            close = getattr(segments, "close", None)
            if callable(close):
                close()
        return tokens

    return handle


class SttWorker(Protocol):
    """The worker surface :class:`WhisperSTT` needs (injectable in tests)."""

    async def start(self) -> None:
        """Spawn/await model readiness."""
        ...

    async def request(self, payload: Any) -> tuple[str, Any]:
        """Send one decode request; return ``(status, value)``."""
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


@dataclass(slots=True)
class _DecodeRequest:
    turn_id: str
    onset_t_ms: int
    final: bool
    pcm: bytes


@dataclass(slots=True)
class _Dispatch:
    """Bounded decode dispatch: FIFO finals, one latest-wins queued partial."""

    max_pending_turns: int
    queue: deque[_DecodeRequest] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    eof: bool = False
    error: BaseException | None = None

    def push_partial(self, request: _DecodeRequest) -> None:
        # Invariant: at most one non-final request may be queued, always at the
        # tail (partials only come from the currently open turn).
        if self.queue and not self.queue[-1].final:
            self.queue[-1] = request  # latest wins
        else:
            self.queue.append(request)
        self.wake.set()

    def push_final(self, request: _DecodeRequest) -> None:
        # The bound applies on every path: replacing a queued partial with a
        # final still adds one pending final decode.
        pending_finals = sum(1 for r in self.queue if r.final)
        if pending_finals >= self.max_pending_turns:
            exc = SttOverrunError(
                f"{pending_finals} finalized turns already await decoding; "
                "inference cannot keep up with live speech"
            )
            self.fail(exc)
            raise exc
        if self.queue and not self.queue[-1].final and self.queue[-1].turn_id == request.turn_id:
            self.queue[-1] = request  # the final clears/replaces the queued partial
        else:
            self.queue.append(request)
        self.wake.set()

    def mark_eof(self) -> None:
        self.eof = True
        self.wake.set()

    def fail(self, exc: BaseException) -> None:
        if self.error is None:
            self.error = exc
        self.wake.set()


class WhisperSTT:
    """Streaming STT over faster-whisper with adapter-owned endpointing.

    Args:
        model_size: faster-whisper model id/alias (e.g. ``"small"``) or a
            resolved local snapshot path.
        language: Source language hint (e.g. ``"en"``).
        device: ``"cpu"`` / ``"cuda"`` / ``"auto"``.
        compute_type: faster-whisper compute type (e.g. ``"int8"``).
        vad_threshold: RMS speech threshold for the endpointing VAD.
        vad_hangover_ms: EnergyVAD hangover inside the endpoint detector.
        pre_roll_ms: Leading audio retained before a detected onset.
        partial_interval_ms: Minimum interval between partial decodes.
        end_silence_ms: Trailing silence that finalizes a turn.
        max_utterance_ms: Hard per-turn cap (longer speech is split).
        max_pending_turns: Bound on finalized turns awaiting decode before a
            typed :class:`SttOverrunError` is surfaced.
        worker: Injectable decode worker (tests); defaults to a spawned
            :class:`~interpret_live.model_worker.ModelWorker` running
            :func:`build_whisper_handler`.
        ready_timeout_s: Model-load readiness budget for the default worker.
        grace_s: Per-stage shutdown budget for the default worker.
    """

    def __init__(
        self,
        *,
        model_size: str = "small",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        vad_threshold: float = 0.02,
        vad_hangover_ms: int = 200,
        pre_roll_ms: int = 200,
        partial_interval_ms: int = 500,
        end_silence_ms: int = 500,
        max_utterance_ms: int = 30_000,
        max_pending_turns: int = 8,
        worker: SttWorker | None = None,
        ready_timeout_s: float = 120.0,
        grace_s: float = 2.0,
    ) -> None:
        if not model_size:
            raise ValueError("model_size must be a non-empty model id or local path")
        if not language or not language.replace("-", "").isalpha():
            raise ValueError(f"language must be a language code, got {language!r}")
        if device not in _VALID_DEVICES:
            raise ValueError(f"device must be one of {sorted(_VALID_DEVICES)}, got {device!r}")
        if compute_type not in _VALID_COMPUTE_TYPES:
            raise ValueError(
                f"compute_type must be one of {sorted(_VALID_COMPUTE_TYPES)}, got {compute_type!r}"
            )
        if max_pending_turns < 1:
            raise ValueError(f"max_pending_turns must be >= 1, got {max_pending_turns}")
        self._language = language
        self._vad_threshold = vad_threshold
        self._vad_hangover_ms = vad_hangover_ms
        self._pre_roll_ms = pre_roll_ms
        self._partial_interval_ms = partial_interval_ms
        self._end_silence_ms = end_silence_ms
        self._max_utterance_ms = max_utterance_ms
        self._max_pending_turns = max_pending_turns
        if worker is None:
            # The spawned child shares this interpreter's environment, so a
            # parent-side import check is a valid (and fail-fast) preflight.
            require("faster_whisper", backend="whisper", extra="whisper")
            worker = ModelWorker(
                "interpret_live.backends.whisper:build_whisper_handler",
                {
                    "model_source": model_size,
                    "language": language,
                    "device": device,
                    "compute_type": compute_type,
                },
                name="whisper-worker",
                ready_timeout_s=ready_timeout_s,
                grace_s=grace_s,
            )
        self._worker: SttWorker = worker
        self._closed = False

    async def start(self) -> None:
        """Start the model worker and wait for readiness (before devices open)."""
        await self._worker.start()

    async def aclose(self) -> None:
        """Cancel in-flight decode work and reap the worker (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._worker.signal_cancel()
        await self._worker.aclose()

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        """Return the streaming hypothesis iterator for ``audio``."""
        return self._stream(audio)

    async def _stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Hypothesis]:
        await self.start()
        dispatch = _Dispatch(self._max_pending_turns)
        out_q: asyncio.Queue[Hypothesis | BaseException | None] = asyncio.Queue()
        ingest = asyncio.create_task(self._ingest(audio, dispatch, out_q), name="whisper-ingest")
        pump = asyncio.create_task(self._decode_pump(dispatch, out_q), name="whisper-pump")
        try:
            while True:
                item = await out_q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
            # Surface a late ingest failure (e.g. bad final frame) if any.
            await ingest
        finally:
            for task in (ingest, pump):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # ----- ingestion: resample -> endpoint -> buffer -> pace decodes ---------

    async def _ingest(
        self,
        audio: AsyncIterator[AudioFrame],
        dispatch: _Dispatch,
        out_q: asyncio.Queue[Hypothesis | BaseException | None],
    ) -> None:
        detector = UtteranceEndpointDetector(
            EnergyVAD(threshold=self._vad_threshold, hangover_ms=self._vad_hangover_ms),
            pre_roll_ms=self._pre_roll_ms,
            partial_interval_ms=self._partial_interval_ms,
            end_silence_ms=self._end_silence_ms,
            max_utterance_ms=self._max_utterance_ms,
        )
        resampler: StreamingResampler | None = None
        in_rate: int | None = None
        turn_id: str | None = None
        onset_t_ms = 0
        buffer: list[NDArray[np.float32]] = []
        last_t_ms = 0
        try:
            async for frame in audio:
                if in_rate is None:
                    in_rate = frame.sample_rate
                    resampler = StreamingResampler(in_rate, _DECODE_RATE)
                elif frame.sample_rate != in_rate:
                    raise SttStreamError(
                        f"source sample rate changed mid-stream: {in_rate} -> "
                        f"{frame.sample_rate}; live capture must keep one rate"
                    )
                assert resampler is not None
                block = resampler.process(frame.samples)
                last_t_ms = frame.t_ms
                if block.size == 0:
                    continue
                rframe = AudioFrame(
                    samples=np.clip(block, -1.0, 1.0),
                    sample_rate=_DECODE_RATE,
                    t_ms=frame.t_ms,
                )
                action = detector.feed(rframe)
                if action.end_reason is not None and turn_id is not None:
                    self._submit(dispatch, turn_id, onset_t_ms, buffer, final=True)
                    turn_id = None
                    buffer = []
                if action.started_turn_id is not None:
                    turn_id = action.started_turn_id
                    onset_t_ms = action.onset_t_ms or rframe.t_ms
                for buffered in action.frames:
                    buffer.append(buffered.samples)
                if action.partial_due and turn_id is not None and buffer:
                    self._submit(dispatch, turn_id, onset_t_ms, buffer, final=False)
            # Source EOF: flush the resampler exactly once; a tail while a turn
            # is open belongs to that turn.
            if resampler is not None:
                tail = resampler.flush()
                if tail.size and detector.in_turn:
                    tail_frame = AudioFrame(
                        samples=np.clip(tail, -1.0, 1.0),
                        sample_rate=_DECODE_RATE,
                        t_ms=last_t_ms,
                    )
                    action = detector.feed(tail_frame)
                    for buffered in action.frames:
                        buffer.append(buffered.samples)
            if detector.flush() == "eof" and turn_id is not None and buffer:
                self._submit(dispatch, turn_id, onset_t_ms, buffer, final=True)
            dispatch.mark_eof()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            # Surface the failure directly to the stream consumer as well: the
            # pump may be parked inside a long decode request and unable to
            # check the dispatch error promptly.
            dispatch.fail(exc)
            out_q.put_nowait(exc)
            raise

    def _submit(
        self,
        dispatch: _Dispatch,
        turn_id: str,
        onset_t_ms: int,
        buffer: list[NDArray[np.float32]],
        *,
        final: bool,
    ) -> None:
        pcm = np.concatenate(buffer).astype(np.float32, copy=False).tobytes()
        request = _DecodeRequest(turn_id=turn_id, onset_t_ms=onset_t_ms, final=final, pcm=pcm)
        if final:
            dispatch.push_final(request)
        else:
            dispatch.push_partial(request)

    # ----- decode pump: serial worker requests, ordered results ---------------

    async def _decode_pump(
        self,
        dispatch: _Dispatch,
        out_q: asyncio.Queue[Hypothesis | BaseException | None],
    ) -> None:
        try:
            while True:
                while not dispatch.queue:
                    if dispatch.error is not None:
                        raise dispatch.error
                    if dispatch.eof:
                        out_q.put_nowait(None)
                        return
                    dispatch.wake.clear()
                    await dispatch.wake.wait()
                if dispatch.error is not None:
                    raise dispatch.error
                request = dispatch.queue.popleft()
                self._worker.clear_cancel()
                status, value = await self._worker.request(
                    {"pcm": request.pcm, "final": request.final, "turn": request.turn_id}
                )
                if status == "cancelled":
                    continue  # stale by definition: cancellation discards it
                if status == "error":
                    raise RuntimeError(f"whisper decode failed: {value}")
                tokens = tuple(
                    Token(text=text, start_ms=start, end_ms=end) for text, start, end in value
                )
                if not tokens:
                    continue  # never emit empty/silence-only hypotheses
                out_q.put_nowait(
                    Hypothesis(
                        tokens=tokens,
                        is_final=request.final,
                        source_turn_id=request.turn_id,
                        speech_started_at_ms=request.onset_t_ms,
                    )
                )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            out_q.put_nowait(exc)
