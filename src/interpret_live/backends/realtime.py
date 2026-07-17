"""OpenAI Realtime transport for the persistent S2S protocol (``openai`` extra).

Implements :class:`RealtimeS2S` — the provider side of
:class:`~interpret_live.backends.S2S` — over the official asynchronous Python
SDK's WebSocket surface (``AsyncOpenAI().realtime.connect``; the SDK's
``realtime`` extra installs the transport). ``OPENAI_API_KEY`` is resolved by
the SDK from the process environment: it is never accepted as a constructor
value, logged, or included in exceptions.

Connection anatomy (plan Task 7):

* **Input encoder** — one stateful 24 kHz resampler for the continuous
  source, little-endian PCM16/base64 append commands, and a cumulative
  sent-audio ledger mapping provider-buffer milliseconds back to original
  frame timestamps in the injected clock domain (pruned as speech-start
  offsets resolve, bounded by unresolved audio, flushed exactly once at EOF).
* **One serialized outbound command pump** — the only task that writes to the
  WebSocket. Each small send has a bounded timeout; appends keep their order;
  a barge-in's cancel+truncate ships as one prioritized, non-interleavable
  group immediately after any already-started send.
* **Receiver** — maps server events into the typed S2S event union (statuses
  distinguished via ``response.done``); server ``error`` events are typed
  failures unless they reference one of our own targeted cancel/commit client
  event ids with a benign already-finished/empty-buffer code.
* **EOF state machine** — ``empty → speech_pending → auto_committed /
  manual_commit_sent → manual_committed → response_started → response_done``
  under the connection-state lock: no duplicate commit, exactly one
  ``response.create``, races with server auto-commit resolved benignly.

There is no transparent reconnect once audio has been sent (replay could
duplicate or lose speech); a bounded reconnect is allowed only while the
initial connection has not yet carried a frame.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..audio_codec import StreamingResampler, float32_to_pcm16, pcm16_to_float32
from ..clock import Clock, RealClock
from ..types import (
    AudioFrame,
    S2SAudioChunk,
    S2SContentDone,
    S2SEvent,
    S2SInterruptTarget,
    S2SResponseDone,
    S2SResponseStarted,
    S2SSpeechCommitted,
    S2SSpeechStarted,
)
from .guard import require

__all__ = ["DEFAULT_REALTIME_MODEL", "RealtimeError", "RealtimeS2S"]

#: The tested default Realtime model (configurable per session).
DEFAULT_REALTIME_MODEL = "gpt-realtime"

_WIRE_RATE = 24000  # OpenAI Realtime PCM is 24 kHz mono little-endian PCM16

#: Server error codes that are benign when they reference our own targeted
#: control event (the response finished before the cancel/commit landed).
_BENIGN_CONTROL_CODES = {
    "response_cancel_not_active",
    "response_not_active",
    "no_active_response",
    "input_audio_buffer_commit_empty",
}


class RealtimeError(RuntimeError):
    """A typed OpenAI Realtime transport failure."""


@dataclass(slots=True)
class _LedgerEntry:
    start_sample: int
    n_samples: int
    source_t_ms: int


class _SentAudioLedger:
    """Map provider input-buffer positions back to source-clock timestamps.

    The provider reports speech offsets as milliseconds into the cumulative
    input audio buffer; this ledger keeps (buffer-sample-range -> original
    frame timestamp) anchors so those offsets translate into the injected
    clock domain regardless of how local VAD would have segmented the frames.
    """

    def __init__(self, *, max_entries: int = 4096) -> None:
        self._entries: deque[_LedgerEntry] = deque(maxlen=max_entries)
        self._total_samples = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def record(self, n_samples: int, source_t_ms: int) -> None:
        if n_samples <= 0:
            return
        self._entries.append(_LedgerEntry(self._total_samples, n_samples, source_t_ms))
        self._total_samples += n_samples

    def source_time_at_buffer_ms(self, buffer_ms: float) -> int:
        """Translate a provider buffer offset to the original source time."""
        target = round(buffer_ms * _WIRE_RATE / 1000)
        for entry in self._entries:
            if target < entry.start_sample:
                return entry.source_t_ms  # pruned/early: clamp to entry start
            if target < entry.start_sample + entry.n_samples:
                within = target - entry.start_sample
                return round(entry.source_t_ms + 1000 * within / _WIRE_RATE)
        if self._entries:
            last = self._entries[-1]
            return round(last.source_t_ms + 1000 * last.n_samples / _WIRE_RATE)
        return 0

    def prune_resolved_before_ms(self, buffer_ms: float) -> None:
        """Drop entries fully behind a resolved speech-start offset.

        The cumulative sample anchor is retained (entries carry absolute
        positions), so later offsets still map correctly.
        """
        target = round(buffer_ms * _WIRE_RATE / 1000)
        while self._entries and (
            self._entries[0].start_sample + self._entries[0].n_samples <= target
        ):
            self._entries.popleft()


class _CommandPump:
    """The single WebSocket writer: ordered appends, prioritized control groups."""

    def __init__(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        send_timeout_s: float,
        maxsize: int,
    ) -> None:
        self._send = send
        self._send_timeout_s = send_timeout_s
        self._normal: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=maxsize)
        self._priority: deque[list[dict[str, Any]]] = deque()
        self._wake = asyncio.Event()
        self._failed: BaseException | None = None

    async def enqueue(self, payload: dict[str, Any]) -> None:
        """Queue one ordinary command (bounded: applies input backpressure)."""
        if self._failed is not None:
            raise RealtimeError(f"outbound pump failed: {self._failed}")
        await self._normal.put(payload)
        self._wake.set()

    def enqueue_priority_group(self, group: list[dict[str, Any]]) -> None:
        """Queue a non-interleavable control group (sent before new appends)."""
        self._priority.append(list(group))
        self._wake.set()

    async def close(self) -> None:
        await self._normal.put(None)
        self._wake.set()

    async def run(self) -> None:
        """Serialize every outbound write with a bounded per-send timeout."""
        try:
            while True:
                while not self._priority and self._normal.empty():
                    self._wake.clear()
                    await self._wake.wait()
                if self._priority:
                    # The whole group goes out back-to-back: no newer append
                    # may interleave between cancel and truncate.
                    group = self._priority.popleft()
                    for payload in group:
                        await self._timed_send(payload)
                    continue
                item = self._normal.get_nowait()
                if item is None:
                    return
                await self._timed_send(item)
        except BaseException as exc:
            self._failed = exc
            raise

    async def _timed_send(self, payload: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(self._send(payload), self._send_timeout_s)
        except TimeoutError as exc:
            raise RealtimeError(
                f"outbound send of {payload.get('type')!r} exceeded "
                f"{self._send_timeout_s:.1f}s; closing the session (a concurrent "
                "WebSocket send is never attempted)"
            ) from exc


class _EofState:
    """The EOF commit/create state machine (guarded by the connection lock)."""

    EMPTY = "empty"
    SPEECH_PENDING = "speech_pending"
    AUTO_COMMITTED = "auto_committed"
    MANUAL_COMMIT_SENT = "manual_commit_sent"
    MANUAL_COMMITTED = "manual_committed"
    RESPONSE_STARTED = "response_started"
    RESPONSE_DONE = "response_done"

    def __init__(self) -> None:
        self.state = self.EMPTY
        self.commit_event_id: str | None = None
        self.committed = asyncio.Event()
        self.response_done = asyncio.Event()


class RealtimeS2S:
    """Speech-to-speech via the OpenAI Realtime API (persistent session).

    Args:
        model: Realtime model id (tested default: ``gpt-realtime``).
        voice: Provider output voice, applied to
            ``session.audio.output.voice`` before the first response.
        source_lang: Source language for the translation-only instructions.
        target_lang: Target language for the translation-only instructions.
        clock: Injected clock (defaults to a :class:`RealClock`).
        connect: Injectable connection factory (tests); the default uses
            ``AsyncOpenAI().realtime.connect(model=...)`` with the API key
            resolved by the SDK from the environment.
        send_timeout_s: Bounded per-send WebSocket write budget.
        vad_settle_ms: Bounded wait for a server auto-commit at EOF before a
            manual commit is sent for still-pending speech.
        final_response_timeout_ms: Bounded wait for the final response at EOF.
        queue_maxsize: Bound for the outbound/inbound event queues.
        connect_attempts: Initial-connection attempts (before any audio is
            sent); once audio has flowed there is never a reconnect.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_REALTIME_MODEL,
        voice: str = "marin",
        source_lang: str = "en",
        target_lang: str = "es",
        clock: Clock | None = None,
        connect: Callable[[], Awaitable[Any]] | None = None,
        send_timeout_s: float = 10.0,
        vad_settle_ms: int = 1500,
        final_response_timeout_ms: int = 30_000,
        queue_maxsize: int = 64,
        connect_attempts: int = 2,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty Realtime model id")
        if not voice:
            raise ValueError("voice must be a non-empty provider voice name")
        if not source_lang or not target_lang:
            raise ValueError("source_lang and target_lang must be non-empty")
        if connect_attempts < 1:
            raise ValueError(f"connect_attempts must be >= 1, got {connect_attempts}")
        self._model = model
        self._voice = voice
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._clock: Clock | None = clock  # resolved lazily inside the running loop
        self._send_timeout_s = send_timeout_s
        self._vad_settle_ms = vad_settle_ms
        self._final_response_timeout_ms = final_response_timeout_ms
        self._queue_maxsize = queue_maxsize
        self._connect_attempts = connect_attempts
        if connect is None:
            openai = require("openai", backend="realtime", extra="openai")
            require("websockets", backend="realtime", extra="openai")

            async def _connect() -> Any:
                client = openai.AsyncOpenAI()  # API key from the environment only
                manager = client.realtime.connect(model=self._model)
                return await manager.enter()

            connect = _connect
        self._connect = connect
        self._pump: _CommandPump | None = None
        self._control_counter = 0
        self._pending_controls: dict[str, str] = {}  # event_id -> command type
        self._state_lock = asyncio.Lock()
        self._eof = _EofState()
        self._audio_sent = False
        self._closing = False

    # ----- public protocol surface ----------------------------------------------

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[S2SEvent]:
        """Yield typed provider events for the continuous source ``audio``."""
        return self._stream(audio)

    async def interrupt(self, target: S2SInterruptTarget) -> None:
        """Cancel exactly the snapshotted response (and truncate heard audio).

        The cancel/truncate pair ships through the serialized command pump as
        one prioritized, non-interleavable group; the connection stays
        available for the next server-VAD turn.
        """
        pump = self._pump
        if pump is None:
            raise RealtimeError("interrupt before the Realtime session started")
        group: list[dict[str, Any]] = []
        cancel_id = self._next_control_id("cancel")
        self._pending_controls[cancel_id] = "response.cancel"
        group.append(
            {
                "type": "response.cancel",
                "event_id": cancel_id,
                "response_id": target.response_id,  # never an unqualified cancel
            }
        )
        if target.cursor is not None:
            truncate_id = self._next_control_id("truncate")
            self._pending_controls[truncate_id] = "conversation.item.truncate"
            group.append(
                {
                    "type": "conversation.item.truncate",
                    "event_id": truncate_id,
                    "item_id": target.cursor.item_id,
                    "content_index": target.cursor.content_index,
                    "audio_end_ms": target.cursor.audio_end_ms,
                }
            )
        pump.enqueue_priority_group(group)

    def _next_control_id(self, kind: str) -> str:
        self._control_counter += 1
        return f"il-{kind}-{self._control_counter}"

    # ----- session orchestration ---------------------------------------------------

    async def _open_connection(self) -> Any:
        last: BaseException | None = None
        for attempt in range(1, self._connect_attempts + 1):
            try:
                return await self._connect()
            except BaseException as exc:
                last = exc
                if attempt == self._connect_attempts:
                    break
        raise RealtimeError(f"failed to open the Realtime connection: {last}") from last

    def _session_update_payload(self) -> dict[str, Any]:
        instructions = (
            "You are a professional simultaneous interpreter. Translate every "
            f"utterance you hear from {self._source_lang} into {self._target_lang} "
            "and speak only the translation. Never answer questions, never add "
            "commentary, never switch roles: only translate."
        )
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": _WIRE_RATE},
                        "turn_detection": {
                            "type": "server_vad",
                            # The server creates responses for detected turns...
                            "create_response": True,
                            # ...but never auto-cancels them: local barge-in
                            # owns cancellation.
                            "interrupt_response": False,
                            # No unsolicited idle/empty-turn responses: every
                            # response maps to a tracked turn or the EOF commit.
                            "idle_timeout_ms": None,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": _WIRE_RATE},
                        "voice": self._voice,
                    },
                },
            },
        }

    async def _stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[S2SEvent]:
        # Fresh per-session state (a new stream is a new provider session).
        if self._clock is None:
            self._clock = RealClock()  # needs the running loop; never in __init__
        self._eof = _EofState()
        self._audio_sent = False
        self._closing = False
        self._pending_controls.clear()
        conn = await self._open_connection()
        out_q: asyncio.Queue[S2SEvent | BaseException | None] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )
        ledger = _SentAudioLedger()
        pump = _CommandPump(
            conn.send, send_timeout_s=self._send_timeout_s, maxsize=self._queue_maxsize
        )
        self._pump = pump
        await pump.enqueue(self._session_update_payload())

        pump_task = asyncio.create_task(pump.run(), name="realtime-pump")
        receiver_task = asyncio.create_task(
            self._receive(conn, ledger, out_q), name="realtime-receiver"
        )
        encoder_task = asyncio.create_task(
            self._encode_input(audio, ledger, pump), name="realtime-encoder"
        )
        tasks = (pump_task, receiver_task, encoder_task)
        pending_tasks: set[asyncio.Task[None]] = set(tasks)
        try:
            while True:
                get_out = asyncio.create_task(out_q.get(), name="realtime-out-get")
                done, _pending = await asyncio.wait(
                    {get_out, *pending_tasks}, return_when=asyncio.FIRST_COMPLETED
                )
                if get_out in done:
                    item = get_out.result()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
                    continue
                get_out.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_out
                for task in list(pending_tasks):
                    if task.done():
                        pending_tasks.discard(task)
                        exc = task.exception()
                        if exc is not None:
                            raise RealtimeError(f"realtime session task failed: {exc}") from exc
                if encoder_task.done() and not self._closing:
                    # Clean encoder EOF (commit/create machine finished): close
                    # the connection so the receiver ends the event stream.
                    self._closing = True
                    with contextlib.suppress(Exception):
                        await conn.close()
        finally:
            self._closing = True
            for task in (*tasks,):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            with contextlib.suppress(Exception):
                await conn.close()
            self._pump = None

    # ----- input path -------------------------------------------------------------

    async def _encode_input(
        self,
        audio: AsyncIterator[AudioFrame],
        ledger: _SentAudioLedger,
        pump: _CommandPump,
    ) -> None:
        resampler: StreamingResampler | None = None
        in_rate: int | None = None
        base_t_ms = 0
        out_samples = 0
        async for frame in audio:
            if in_rate is None:
                in_rate = frame.sample_rate
                base_t_ms = frame.t_ms
                resampler = StreamingResampler(in_rate, _WIRE_RATE)
            elif frame.sample_rate != in_rate:
                raise RealtimeError(
                    f"source sample rate changed mid-stream: {in_rate} -> {frame.sample_rate}"
                )
            assert resampler is not None
            block = resampler.process(frame.samples)
            if block.size:
                # Anchor each output block at its cumulative position in the
                # (duration-preserving) output stream, so the resampler's
                # internal buffering never skews buffer-offset -> source-time
                # mapping for a contiguous live capture.
                block_t_ms = base_t_ms + round(1000 * out_samples / _WIRE_RATE)
                ledger.record(len(block), block_t_ms)
                out_samples += len(block)
                await self._append_audio(pump, block)
        # Source EOF: flush the input resampler exactly once...
        if resampler is not None:
            tail = resampler.flush()
            if tail.size:
                tail_t_ms = base_t_ms + round(1000 * out_samples / _WIRE_RATE)
                ledger.record(len(tail), tail_t_ms)
                out_samples += len(tail)
                await self._append_audio(pump, tail)
        # ...then run the acknowledged commit/create state machine.
        await self._finish_input(pump)

    async def _append_audio(self, pump: _CommandPump, block: Any) -> None:
        payload = base64.b64encode(float32_to_pcm16(block)).decode("ascii")
        self._audio_sent = True
        await pump.enqueue({"type": "input_audio_buffer.append", "audio": payload})

    async def _finish_input(self, pump: _CommandPump) -> None:
        """EOF: commit pending speech exactly once, then await the response."""
        async with self._state_lock:
            state = self._eof.state
            if state == _EofState.EMPTY or not self._audio_sent:
                return  # nothing to commit; the receiver will end the stream
        if state == _EofState.SPEECH_PENDING:
            # Bounded VAD-settle window: the server may auto-commit the turn
            # while the receiver keeps processing.
            assert self._clock is not None
            await self._clock.sleep(self._vad_settle_ms)
            async with self._state_lock:
                if self._eof.state == _EofState.SPEECH_PENDING:
                    commit_id = self._next_control_id("commit")
                    self._eof.commit_event_id = commit_id
                    self._pending_controls[commit_id] = "input_audio_buffer.commit"
                    self._eof.state = _EofState.MANUAL_COMMIT_SENT
                    await pump.enqueue({"type": "input_audio_buffer.commit", "event_id": commit_id})
            if self._eof.state == _EofState.MANUAL_COMMIT_SENT:
                # Await the commit acknowledgement before creating a response.
                await self._eof.committed.wait()
                async with self._state_lock:
                    if self._eof.state == _EofState.MANUAL_COMMITTED:
                        # No automatic response won the race: exactly one create.
                        await pump.enqueue({"type": "response.create"})
        # Await the final response (or the bounded timeout) before closing.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._eof.response_done.wait(), self._final_response_timeout_ms / 1000
            )

    # ----- receive path ------------------------------------------------------------

    async def _receive(
        self,
        conn: Any,
        ledger: _SentAudioLedger,
        out_q: asyncio.Queue[S2SEvent | BaseException | None],
    ) -> None:
        try:
            async for event in conn:
                mapped = await self._map_event(event, ledger)
                for item in mapped:
                    await out_q.put(item)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if self._closing:
                await out_q.put(None)  # our own clean close ended the iteration
                return
            # Disconnects/rate-limit failures after audio has been sent are
            # terminal: replaying input could duplicate or lose speech — there
            # is deliberately no transparent reconnect here.
            await out_q.put(RealtimeError(f"realtime connection failed: {exc}"))
            return
        await out_q.put(None)

    async def _map_event(self, event: Any, ledger: _SentAudioLedger) -> list[S2SEvent]:
        etype = str(getattr(event, "type", ""))
        if etype == "input_audio_buffer.speech_started":
            buffer_ms = float(getattr(event, "audio_start_ms", 0) or 0)
            source_ms = ledger.source_time_at_buffer_ms(buffer_ms)
            ledger.prune_resolved_before_ms(buffer_ms)
            async with self._state_lock:
                if self._eof.state in (_EofState.EMPTY, _EofState.AUTO_COMMITTED):
                    self._eof.state = _EofState.SPEECH_PENDING
            return [
                S2SSpeechStarted(
                    input_item_id=str(getattr(event, "item_id", "")),
                    source_started_at_ms=source_ms,
                )
            ]
        if etype == "input_audio_buffer.committed":
            async with self._state_lock:
                if self._eof.state == _EofState.MANUAL_COMMIT_SENT:
                    self._eof.state = _EofState.MANUAL_COMMITTED
                elif self._eof.state == _EofState.SPEECH_PENDING:
                    self._eof.state = _EofState.AUTO_COMMITTED
                self._eof.committed.set()
            return [S2SSpeechCommitted(input_item_id=str(getattr(event, "item_id", "")))]
        if etype == "response.created":
            async with self._state_lock:
                if self._eof.state != _EofState.RESPONSE_DONE:
                    self._eof.state = _EofState.RESPONSE_STARTED
                # An automatic response supersedes any manual commit/create
                # still pending: wake the EOF machine so it never duplicates.
                self._eof.committed.set()
            response = getattr(event, "response", None)
            return [
                S2SResponseStarted(
                    response_id=str(
                        getattr(response, "id", "") or getattr(event, "response_id", "")
                    )
                )
            ]
        if etype == "response.output_audio.delta":
            raw = getattr(event, "delta", "")
            try:
                pcm = base64.b64decode(raw, validate=True)
                samples = pcm16_to_float32(pcm)
            except (binascii.Error, ValueError) as exc:
                raise RealtimeError(f"malformed response audio delta: {exc}") from exc
            return [
                S2SAudioChunk(
                    samples=samples,
                    sample_rate=_WIRE_RATE,
                    response_id=str(getattr(event, "response_id", "")),
                    item_id=str(getattr(event, "item_id", "")),
                    output_index=int(getattr(event, "output_index", 0) or 0),
                    content_index=int(getattr(event, "content_index", 0) or 0),
                    final=False,
                )
            ]
        if etype == "response.output_audio.done":
            # Content completion only — never a response/natural completion.
            return [
                S2SContentDone(
                    response_id=str(getattr(event, "response_id", "")),
                    item_id=str(getattr(event, "item_id", "")),
                    content_index=int(getattr(event, "content_index", 0) or 0),
                )
            ]
        if etype == "response.done":
            response = getattr(event, "response", None)
            status = str(getattr(response, "status", "") or "completed")
            details = getattr(response, "status_details", None)
            reason = str(getattr(details, "reason", "") or "") or None
            async with self._state_lock:
                self._eof.state = _EofState.RESPONSE_DONE
                self._eof.response_done.set()
            return [
                S2SResponseDone(
                    response_id=str(getattr(response, "id", "") or ""),
                    status=status,
                    reason=reason,
                )
            ]
        if etype == "error":
            error = getattr(event, "error", None)
            code = str(getattr(error, "code", "") or "")
            related = str(getattr(error, "event_id", "") or "")
            if related and related in self._pending_controls and code in _BENIGN_CONTROL_CODES:
                # The documented no-active-response/empty-buffer outcome for a
                # control command we sent that lost a benign race.
                self._pending_controls.pop(related, None)
                if related == self._eof.commit_event_id:
                    self._eof.committed.set()
                return []
            message = str(getattr(error, "message", "") or "unknown server error")
            raise RealtimeError(f"realtime server error ({code or 'unknown'}): {message}")
        return []  # session.created / session.updated / unrelated events
