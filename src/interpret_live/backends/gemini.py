"""Gemini Live transport for the persistent S2S protocol (``gemini`` extra).

Implements :class:`GeminiS2S` — the provider side of
:class:`~interpret_live.backends.S2S` — over ``google-genai``'s asynchronous
``client.aio.live.connect()`` WebSocket session. The API key is resolved by the
SDK from the process environment (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``): it
is never accepted as a constructor value, logged, or included in exceptions,
exactly as on the Realtime path.

Connection anatomy:

* **Input encoder** — one stateful 16 kHz resampler for the continuous source
  (Live input is 16 kHz mono little-endian PCM16), each block sent as one
  ``send_realtime_input(audio=...)`` with a bounded write budget. The encoder
  task is the *only* writer, so no outbound serializer is needed here: unlike
  the Realtime path, a barge-in sends nothing (see :meth:`interrupt`).
* **Receiver** — ``session.receive()`` ends at every ``turn_complete``, so it is
  re-entered in an outer loop to keep one session spanning many turns. Server
  messages are mapped into the typed S2S event union.
* **Turn identity** — the Live API carries no response/item ids, so the adapter
  mints them (``gemini-in-N`` / ``gemini-resp-N`` / ``gemini-item-N``) and opens
  exactly one turn per model turn. A turn opens on the provider's *first*
  output for it, never on local audio: turn creation stays the provider's.
* **Speech onset** — the Live API also reports no input-speech offset, so the
  onset that dates each utterance comes from an
  :class:`~interpret_live.vad.EnergyVAD` reading the same source frames the
  encoder sends, and is used only to *timestamp* a turn the provider has
  already started. A provider ``voice_activity`` signal, for accounts that
  receive one, overrides it.
* **EOF state machine** — ``audio_stream_end`` is sent exactly once, then the
  final turn is awaited: a bounded VAD-settle window for a turn that has not
  started yet, then a bounded wait for its completion.

There is no transparent reconnect once audio has been sent (replay could
duplicate or lose speech); a bounded retry is allowed only while the initial
connection has not yet carried a frame.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

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
from ..vad import VAD, EnergyVAD
from .guard import require

if TYPE_CHECKING:
    from . import S2S

__all__ = ["DEFAULT_GEMINI_MODEL", "DEFAULT_GEMINI_VOICE", "GeminiError", "GeminiS2S"]

#: The tested default Gemini Live model (configurable per session).
DEFAULT_GEMINI_MODEL = "gemini-live-2.5-flash-preview"
#: Default prebuilt output voice.
DEFAULT_GEMINI_VOICE = "Kore"

_INPUT_RATE = 16000  # Gemini Live input is 16 kHz mono little-endian PCM16
_INPUT_MIME = f"audio/pcm;rate={_INPUT_RATE}"
_DEFAULT_OUTPUT_RATE = 24000  # Live output rate when the mime type omits it

#: ``turn_complete_reason`` values that still describe a natural completion.
_BENIGN_TURN_REASONS = {
    "",
    "TURN_COMPLETE_REASON_UNSPECIFIED",
    "NEED_MORE_INPUT",
}


class GeminiError(RuntimeError):
    """A typed Gemini Live transport failure."""


def _rate_from_mime(mime_type: str) -> int:
    """Parse ``audio/pcm;rate=24000`` into ``24000``.

    Raises:
        GeminiError: If the part is audio the adapter cannot decode (anything
            but raw PCM), rather than handing the sink noise.
    """
    media, _, params = mime_type.partition(";")
    if media.strip() not in ("audio/pcm", "audio/l16"):
        raise GeminiError(f"unsupported response audio mime type {mime_type!r}; expected audio/pcm")
    for param in params.split(";"):
        key, _, value = param.partition("=")
        if key.strip() != "rate":
            continue
        try:
            rate = int(value.strip())
        except ValueError as exc:
            raise GeminiError(f"malformed rate in mime type {mime_type!r}") from exc
        if rate <= 0:
            raise GeminiError(f"non-positive rate in mime type {mime_type!r}")
        return rate
    return _DEFAULT_OUTPUT_RATE


def _duration_to_ms(value: object) -> int | None:
    """Parse a protobuf duration string (``"1.5s"``) into milliseconds."""
    if value is None:
        return None
    text = str(value).strip().removesuffix("s")
    try:
        return round(float(text) * 1000)
    except ValueError:
        return None


class _OnsetTracker:
    """Dates provider turns from local speech onsets in the source stream.

    The Live API reports no input-speech offset, so something has to supply the
    ``utterance_start`` timestamp every latency metric is measured from. This
    watches the same frames the encoder sends and remembers where the most
    recent speech onset was; the value is *claimed* only when the provider
    actually opens a turn, so a local false positive can never roll a turn of
    its own. A provider-supplied offset, when one arrives, replaces the local
    estimate for the next claim.
    """

    __slots__ = ("_in_speech", "_onset_ms", "_vad")

    def __init__(self, vad: VAD) -> None:
        self._vad = vad
        self._in_speech = False
        self._onset_ms: int | None = None

    def feed(self, frame: AudioFrame) -> None:
        """Record a silence → speech transition's frame timestamp."""
        speech = self._vad.is_speech(frame)
        if speech and not self._in_speech:
            self._onset_ms = frame.t_ms
        self._in_speech = speech

    def note_provider_onset(self, source_t_ms: int) -> None:
        """Override the local estimate with an authoritative provider offset."""
        self._onset_ms = source_t_ms

    def claim(self, fallback_ms: int) -> int:
        """Return (and consume) the onset for a turn the provider just opened."""
        onset = self._onset_ms
        self._onset_ms = None
        return fallback_ms if onset is None else onset


class _Turn:
    """The synthetic identity the adapter mints for one provider turn."""

    __slots__ = ("done_emitted", "input_item_id", "item_id", "response_id")

    def __init__(self, index: int) -> None:
        self.input_item_id = f"gemini-in-{index}"
        self.response_id = f"gemini-resp-{index}"
        self.item_id = f"gemini-item-{index}"
        # An interrupted turn reports `interrupted` and *then* `turn_complete`;
        # the first of the two is the one that states the terminal status.
        self.done_emitted = False


class GeminiS2S:
    """Speech-to-speech via the Gemini Live API (persistent session).

    Args:
        model: Live model id (tested default: ``gemini-live-2.5-flash-preview``).
        voice: Prebuilt output voice name.
        source_lang: Source language for the translation-only instructions.
        target_lang: Target language for the instructions and for
            ``speech_config.language_code``.
        native_translation: Send the Live API's own ``translation_config``
            alongside the instructions. Off by default: it requires a
            translation-capable Live model, which rejects the session outright
            everywhere else.
        clock: Injected clock (defaults to a :class:`RealClock`).
        vad: Injected VAD, used only to timestamp speech onsets (defaults to an
            :class:`~interpret_live.vad.EnergyVAD`).
        connect: Injectable connection factory receiving the session config
            (tests); the default enters ``genai.Client().aio.live.connect(...)``
            with the API key resolved by the SDK from the environment.
        send_timeout_s: Bounded per-send WebSocket write budget.
        vad_settle_ms: Bounded wait at EOF for a final turn to start.
        final_response_timeout_ms: Bounded wait for that turn to complete.
        queue_maxsize: Bound for the outbound event queue.
        connect_attempts: Initial-connection attempts (before any audio is
            sent); once audio has flowed there is never a reconnect.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        voice: str = DEFAULT_GEMINI_VOICE,
        source_lang: str = "en",
        target_lang: str = "es",
        native_translation: bool = False,
        clock: Clock | None = None,
        vad: VAD | None = None,
        connect: Callable[[dict[str, Any]], Awaitable[Any]] | None = None,
        send_timeout_s: float = 10.0,
        vad_settle_ms: int = 1500,
        final_response_timeout_ms: int = 30_000,
        queue_maxsize: int = 64,
        connect_attempts: int = 2,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty Gemini Live model id")
        if not voice:
            raise ValueError("voice must be a non-empty prebuilt voice name")
        if not source_lang or not target_lang:
            raise ValueError("source_lang and target_lang must be non-empty")
        if connect_attempts < 1:
            raise ValueError(f"connect_attempts must be >= 1, got {connect_attempts}")
        self._model = model
        self._voice = voice
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._native_translation = native_translation
        self._clock: Clock | None = clock  # resolved lazily inside the running loop
        self._vad: VAD = vad if vad is not None else EnergyVAD()
        self._send_timeout_s = send_timeout_s
        self._vad_settle_ms = vad_settle_ms
        self._final_response_timeout_ms = final_response_timeout_ms
        self._queue_maxsize = queue_maxsize
        self._connect_attempts = connect_attempts
        self._manager: Any | None = None
        if connect is None:
            # google-genai depends on websockets itself, so one guard is enough.
            genai = require("google.genai", backend="gemini", extra="gemini")

            async def _connect(config: dict[str, Any]) -> Any:
                client = genai.Client()  # API key from the environment only
                manager = client.aio.live.connect(model=self._model, config=config)
                session = await manager.__aenter__()
                self._manager = manager
                return session

            connect = _connect
        self._connect = connect
        self._onsets = _OnsetTracker(self._vad)
        self._turn: _Turn | None = None
        self._turn_count = 0
        self._turn_started = asyncio.Event()
        self._turn_finished = asyncio.Event()
        self._abandoned: set[str] = set()
        self._base_t_ms = 0
        self._sent_ms = 0
        self._audio_sent = False
        self._started = False
        self._closing = False
        self._session_closed = False
        self._go_away: str | None = None

    # ----- public protocol surface ----------------------------------------------

    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[S2SEvent]:
        """Yield typed provider events for the continuous source ``audio``."""
        return self._stream(audio)

    async def interrupt(self, target: S2SInterruptTarget) -> None:
        """Abandon exactly ``target.response_id`` — locally, by necessity.

        The Live API has **no** response-scoped cancel and no conversation
        truncation: a turn is only ever cut short by the provider's own VAD, and
        this session deliberately disables that (``NO_INTERRUPTION``) so local
        barge-in stays the single authority. The cancellation this method can
        honestly perform is therefore its own half — the abandoned response's
        remaining audio is dropped at the adapter boundary and never reaches the
        sink, while the provider finishes generating it into the void. The
        session stays open and the next turn is unaffected.

        ``target.cursor`` is accepted for protocol conformance and cannot be
        used: there is nothing to truncate the model's context against.
        """
        if not self._started:
            raise GeminiError("interrupt before the Gemini Live session started")
        self._abandoned.add(target.response_id)

    # ----- session configuration ------------------------------------------------

    def _instructions(self) -> str:
        return (
            "You are a professional simultaneous interpreter. Translate every "
            f"utterance you hear from {self._source_lang} into {self._target_lang} "
            "and speak only the translation. Never answer questions, never add "
            "commentary, never switch roles: only translate."
        )

    def _connect_config(self) -> dict[str, Any]:
        """The ``LiveConnectConfig`` payload, as the plain dict the SDK accepts.

        Built as a dict on purpose: the adapter then needs no ``google.genai``
        type imports, so nothing but the connection factory touches the SDK.
        """
        config: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self._instructions(),
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": self._voice}},
                "language_code": self._target_lang,
            },
            "realtime_input_config": {
                # Server VAD owns turn detection...
                "automatic_activity_detection": {"disabled": False},
                # ...but never cuts a response short on its own: local barge-in
                # is the single authority, as on the Realtime path.
                "activity_handling": "NO_INTERRUPTION",
                "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY",
            },
        }
        if self._native_translation:
            config["translation_config"] = {
                "target_language_code": self._target_lang,
                "echo_target_language": False,
            }
        return config

    # ----- session orchestration ---------------------------------------------------

    async def _open_connection(self) -> Any:
        last: BaseException | None = None
        for attempt in range(1, self._connect_attempts + 1):
            try:
                return await self._connect(self._connect_config())
            except BaseException as exc:
                last = exc
                if attempt == self._connect_attempts:
                    break
        raise GeminiError(f"failed to open the Gemini Live connection: {last}") from last

    async def _stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[S2SEvent]:
        # Fresh per-session state (a new stream is a new provider session).
        if self._clock is None:
            self._clock = RealClock()  # needs the running loop; never in __init__
        self._reset_session_state()
        session = await self._open_connection()
        self._started = True
        out_q: asyncio.Queue[S2SEvent | BaseException | None] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )
        receiver_task = asyncio.create_task(self._receive(session, out_q), name="gemini-receiver")
        encoder_task = asyncio.create_task(
            self._encode_input(audio, session), name="gemini-encoder"
        )
        tasks = (receiver_task, encoder_task)
        pending_tasks: set[asyncio.Task[None]] = set(tasks)
        get_out: asyncio.Task[Any] | None = None
        try:
            while True:
                get_out = asyncio.create_task(out_q.get(), name="gemini-out-get")
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
                            raise GeminiError(f"Gemini Live session task failed: {exc}") from exc
                if encoder_task.done() and not self._closing:
                    # Clean encoder EOF (the final turn settled): close the
                    # session so the receiver ends the event stream.
                    self._closing = True
                    await self._close_session(session)
        finally:
            self._closing = True
            if get_out is not None and not get_out.done():
                get_out.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_out
            for task in tasks:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await self._close_session(session)

    def _reset_session_state(self) -> None:
        self._turn = None
        self._turn_count = 0
        self._turn_started = asyncio.Event()
        self._turn_finished = asyncio.Event()
        self._abandoned.clear()
        self._vad.reset()
        self._onsets = _OnsetTracker(self._vad)
        self._base_t_ms = 0
        self._sent_ms = 0
        self._audio_sent = False
        self._closing = False
        self._session_closed = False
        self._go_away = None

    async def _close_session(self, session: Any) -> None:
        """Close the session (and the SDK context manager) exactly once."""
        if self._session_closed:
            return
        self._session_closed = True
        with contextlib.suppress(Exception):
            await session.close()
        manager, self._manager = self._manager, None
        if manager is not None:
            with contextlib.suppress(Exception):
                await manager.__aexit__(None, None, None)

    # ----- input path -------------------------------------------------------------

    async def _encode_input(self, audio: AsyncIterator[AudioFrame], session: Any) -> None:
        resampler: StreamingResampler | None = None
        in_rate: int | None = None
        out_samples = 0
        async for frame in audio:
            if in_rate is None:
                in_rate = frame.sample_rate
                self._base_t_ms = frame.t_ms
                resampler = StreamingResampler(in_rate, _INPUT_RATE)
            elif frame.sample_rate != in_rate:
                raise GeminiError(
                    f"source sample rate changed mid-stream: {in_rate} -> {frame.sample_rate}"
                )
            assert resampler is not None
            self._onsets.feed(frame)
            block = resampler.process(frame.samples)
            if block.size:
                out_samples += len(block)
                self._sent_ms = round(1000 * out_samples / _INPUT_RATE)
                await self._append_audio(session, block)
        # Source EOF: flush the input resampler exactly once...
        if resampler is not None:
            tail = resampler.flush()
            if tail.size:
                out_samples += len(tail)
                self._sent_ms = round(1000 * out_samples / _INPUT_RATE)
                await self._append_audio(session, tail)
        # ...then close the input stream and await the final turn.
        await self._finish_input(session)

    async def _append_audio(self, session: Any, block: Any) -> None:
        self._audio_sent = True
        await self._timed_send(
            "audio",
            session.send_realtime_input(
                audio={"data": float32_to_pcm16(block), "mime_type": _INPUT_MIME}
            ),
        )

    async def _timed_send(self, what: str, pending: Awaitable[None]) -> None:
        try:
            # asyncio.timeout (not wait_for): on 3.11 wait_for can swallow an
            # external cancellation that races the send's completion, burning
            # the task's single cancel request and leaving it uncancellable.
            async with asyncio.timeout(self._send_timeout_s):
                await pending
        except TimeoutError as exc:
            raise GeminiError(
                f"outbound send of {what!r} exceeded {self._send_timeout_s:.1f}s; "
                "closing the session"
            ) from exc

    async def _finish_input(self, session: Any) -> None:
        """EOF: end the input stream once, then await the final turn."""
        if not self._audio_sent:
            return  # nothing was ever sent; the receiver ends the stream
        await self._timed_send(
            "audio_stream_end", session.send_realtime_input(audio_stream_end=True)
        )
        if self._turn is None:
            # Bounded settle window: the provider may still open a final turn
            # for speech it has not finished endpointing.
            await self._wait_or_settle(self._turn_started, self._vad_settle_ms)
            if self._turn is None:
                return
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self._final_response_timeout_ms / 1000):
                await self._turn_finished.wait()

    async def _wait_or_settle(self, event: asyncio.Event, settle_ms: int) -> None:
        """Wait for ``event``, giving up after ``settle_ms`` on the injected clock."""
        assert self._clock is not None
        waiter = asyncio.create_task(event.wait(), name="gemini-settle-wait")
        timer = asyncio.create_task(self._clock.sleep(settle_ms), name="gemini-settle-timer")
        try:
            await asyncio.wait({waiter, timer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (waiter, timer):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # ----- receive path ------------------------------------------------------------

    async def _receive(
        self, session: Any, out_q: asyncio.Queue[S2SEvent | BaseException | None]
    ) -> None:
        try:
            while True:
                seen = 0
                # `session.receive()` returns at every turn_complete, so one
                # persistent session means re-entering it per turn.
                async for message in session.receive():
                    seen += 1
                    for item in self._map_message(message):
                        await out_q.put(item)
                if seen == 0 or self._closing:
                    break
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if self._closing:
                await out_q.put(None)  # our own clean close ended the iteration
                return
            # A disconnect after audio has been sent is terminal: replaying
            # input could duplicate or lose speech, so there is deliberately no
            # transparent reconnect here.
            detail = (
                f" (the server had signalled go_away: {self._go_away})" if self._go_away else ""
            )
            await out_q.put(GeminiError(f"Gemini Live connection failed: {exc}{detail}"))
            return
        await out_q.put(None)

    def _map_message(self, message: Any) -> list[S2SEvent]:
        """Map one ``LiveServerMessage`` into the typed S2S event union."""
        events: list[S2SEvent] = []
        activity = getattr(message, "voice_activity", None)
        if activity is not None:
            self._note_voice_activity(activity)
        go_away = getattr(message, "go_away", None)
        if go_away is not None:
            self._go_away = str(getattr(go_away, "time_left", "") or "imminently")
        content = getattr(message, "server_content", None)
        if content is None:
            return events  # setup_complete / usage_metadata / tool traffic
        parts = self._audio_parts(content)
        if self._turn is None and not self._opens_a_turn(content, parts):
            return events
        turn = self._turn if self._turn is not None else self._open_turn(events)
        if turn.response_id not in self._abandoned:
            events.extend(self._audio_chunks(turn, parts))
        if getattr(content, "generation_complete", None) and not turn.done_emitted:
            events.append(
                S2SContentDone(response_id=turn.response_id, item_id=turn.item_id, content_index=0)
            )
        done = self._turn_done_event(turn, content)
        if done is not None:
            events.append(done)
            turn.done_emitted = True
        if getattr(content, "turn_complete", None):
            # `turn_complete` always closes the turn — including the one that
            # trails an interruption, which must not open a phantom next turn.
            self._close_turn(turn)
        return events

    @staticmethod
    def _audio_parts(content: Any) -> list[Any]:
        """The inline-data blobs of a model turn, in order."""
        model_turn = getattr(content, "model_turn", None)
        out: list[Any] = []
        for part in getattr(model_turn, "parts", None) or ():
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                out.append(inline)
        return out

    @staticmethod
    def _opens_a_turn(content: Any, parts: list[Any]) -> bool:
        """Is this the provider's first output for a turn we have not opened?

        Input-side transcription deliberately does not qualify: it can arrive
        while the model is still listening, and opening a turn on it would
        announce a response that has not started.
        """
        return bool(
            parts
            or getattr(content, "output_transcription", None) is not None
            or getattr(content, "generation_complete", None)
            or getattr(content, "turn_complete", None)
            or getattr(content, "interrupted", None)
        )

    def _open_turn(self, events: list[S2SEvent]) -> _Turn:
        """Mint a turn and emit its speech-started / committed / started trio."""
        self._turn_count += 1
        turn = _Turn(self._turn_count)
        self._turn = turn
        self._turn_started.set()
        self._turn_finished.clear()
        onset = self._onsets.claim(self._base_t_ms + self._sent_ms)
        events.append(
            S2SSpeechStarted(input_item_id=turn.input_item_id, source_started_at_ms=onset)
        )
        # Server VAD only starts generating once it has endpointed the turn, so
        # the provider's first output is also the commit acknowledgement.
        events.append(S2SSpeechCommitted(input_item_id=turn.input_item_id))
        events.append(
            S2SResponseStarted(response_id=turn.response_id, input_item_id=turn.input_item_id)
        )
        return turn

    def _close_turn(self, turn: _Turn) -> None:
        self._turn = None
        self._abandoned.discard(turn.response_id)
        self._turn_finished.set()
        self._turn_started.clear()

    def _audio_chunks(self, turn: _Turn, parts: list[Any]) -> list[S2SEvent]:
        return [
            S2SAudioChunk(
                samples=pcm16_to_float32(bytes(inline.data)),
                sample_rate=_rate_from_mime(str(getattr(inline, "mime_type", "") or "")),
                response_id=turn.response_id,
                item_id=turn.item_id,
                output_index=0,
                content_index=0,
                final=False,
            )
            for inline in parts
        ]

    @staticmethod
    def _turn_done_event(turn: _Turn, content: Any) -> S2SResponseDone | None:
        """The turn's terminal status, or ``None`` while it continues."""
        if turn.done_emitted:
            return None  # the trailing turn_complete of an interrupted turn
        if getattr(content, "interrupted", None):
            # Only reachable if the provider cancels a turn despite the
            # NO_INTERRUPTION session config — never our own barge-in.
            return S2SResponseDone(
                response_id=turn.response_id,
                status="cancelled",
                reason="the provider interrupted the turn",
            )
        if not getattr(content, "turn_complete", None):
            return None
        reason = str(getattr(content, "turn_complete_reason", "") or "")
        if reason in _BENIGN_TURN_REASONS:
            return S2SResponseDone(response_id=turn.response_id, status="completed")
        return S2SResponseDone(response_id=turn.response_id, status="failed", reason=reason)

    def _note_voice_activity(self, activity: Any) -> None:
        """Use an authoritative provider speech-start offset when one arrives."""
        kind = str(getattr(activity, "voice_activity_type", "") or "")
        if not kind.endswith("ACTIVITY_START"):
            return
        offset_ms = _duration_to_ms(getattr(activity, "audio_offset", None))
        if offset_ms is not None:
            # The 16 kHz stream we send preserves source duration, so an offset
            # into it is an offset into the source itself.
            self._onsets.note_provider_onset(self._base_t_ms + offset_ms)


def _static_protocol_conformance(adapter: GeminiS2S) -> S2S:  # pragma: no cover
    """mypy-enforced: GeminiS2S must keep satisfying the shared S2S protocol."""
    return adapter
