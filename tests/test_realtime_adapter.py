"""RealtimeS2S tests against a scripted fake SDK connection (no network).

Proves: exact session configuration (voice/formats/turn-detection), stateful
input PCM conversion + sent-audio ledger mapping, event/status mapping,
response-ID-scoped cancel/truncate ordering through the single-writer pump,
control-error correlation, EOF commit/create races, reconnect policy, and
clean teardown.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from interpret_live.audio_codec import pcm16_to_float32
from interpret_live.backends.realtime import RealtimeError, RealtimeS2S
from interpret_live.clock import ManualClock
from interpret_live.types import (
    AudioFrame,
    PlaybackCursor,
    S2SAudioChunk,
    S2SInterruptTarget,
    S2SResponseDone,
    S2SSpeechStarted,
)

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")


class FakeConnection:
    """A scripted stand-in for the SDK's realtime WebSocket connection."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = 0
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._ended = asyncio.Event()
        self.send_gate: asyncio.Event | None = None

    async def send(self, payload: dict[str, Any]) -> None:
        if self.send_gate is not None:
            await self.send_gate.wait()
        self.sent.append(payload)

    def push(self, event: Any) -> None:
        self._events.put_nowait(event)

    async def close(self) -> None:
        self.closed += 1
        self._ended.set()
        self._events.put_nowait(None)

    def __aiter__(self) -> FakeConnection:
        return self

    async def __anext__(self) -> Any:
        item = await self._events.get()
        if item is None:
            raise ConnectionError("connection closed")
        return item

    # -- helpers ---------------------------------------------------------------

    def sent_types(self) -> list[str]:
        return [p["type"] for p in self.sent]


def _adapter(conn: FakeConnection, clock: ManualClock, **overrides: Any) -> RealtimeS2S:
    async def connect() -> FakeConnection:
        return conn

    kwargs: dict[str, Any] = {
        "voice": "marin",
        "source_lang": "en",
        "target_lang": "es",
        "clock": clock,
        "connect": connect,
        "vad_settle_ms": 100,
        "final_response_timeout_ms": 200,
        "send_timeout_s": 5.0,
    }
    kwargs.update(overrides)
    return RealtimeS2S(**kwargs)


def _frames(count: int, *, rate: int = 24000, ms: int = 20, start_ms: int = 0) -> list[AudioFrame]:
    n = int(ms * rate / 1000)
    return [
        AudioFrame(
            samples=np.full(n, 0.1, dtype=np.float32), sample_rate=rate, t_ms=start_ms + i * ms
        )
        for i in range(count)
    ]


async def _feed(frames: list[AudioFrame], *, hold: asyncio.Event | None = None) -> Any:
    async def gen() -> Any:
        for f in frames:
            yield f
            await asyncio.sleep(0)
        if hold is not None:
            await hold.wait()

    return gen()


def _speech_started(item: str = "item-1", ms: int = 0) -> Any:
    return SimpleNamespace(
        type="input_audio_buffer.speech_started", item_id=item, audio_start_ms=ms
    )


def _committed(item: str = "item-1") -> Any:
    return SimpleNamespace(type="input_audio_buffer.committed", item_id=item)


def _resp_created(rid: str = "resp-1") -> Any:
    return SimpleNamespace(type="response.created", response=SimpleNamespace(id=rid))


def _delta(rid: str = "resp-1", item: str = "out-1", data: bytes = b"\x00\x01" * 10) -> Any:
    return SimpleNamespace(
        type="response.output_audio.delta",
        response_id=rid,
        item_id=item,
        output_index=0,
        content_index=0,
        delta=base64.b64encode(data).decode(),
    )


def _audio_done(rid: str = "resp-1", item: str = "out-1") -> Any:
    return SimpleNamespace(
        type="response.output_audio.done", response_id=rid, item_id=item, content_index=0
    )


def _resp_done(rid: str = "resp-1", status: str = "completed", reason: str | None = None) -> Any:
    details = SimpleNamespace(reason=reason) if reason else None
    return SimpleNamespace(
        type="response.done",
        response=SimpleNamespace(id=rid, status=status, status_details=details),
    )


def _error(code: str, event_id: str = "", message: str = "boom") -> Any:
    return SimpleNamespace(
        type="error", error=SimpleNamespace(code=code, event_id=event_id, message=message)
    )


async def _drive(clock: ManualClock, cond: Any, *, rounds: int = 600) -> None:
    for _ in range(rounds):
        for _ in range(6):
            await asyncio.sleep(0)
            if cond():
                return
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    raise AssertionError("condition never became true")


async def _collect_session(
    adapter: RealtimeS2S, conn: FakeConnection, frames: list[AudioFrame], clock: ManualClock
) -> list[Any]:
    """Feed frames, let the EOF machine close the session, return events."""
    events: list[Any] = []

    async def consume() -> None:
        async for event in adapter.stream(await _feed(frames)):
            events.append(event)

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: task.done())
    await task
    return events


async def test_session_update_configures_voice_formats_and_vad() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock, voice="cedar")
    await _collect_session(adapter, conn, _frames(0), clock)

    assert conn.sent_types()[0] == "session.update"
    session = conn.sent[0]["session"]
    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    assert "en" in session["instructions"] and "es" in session["instructions"]
    assert session["audio"]["output"]["voice"] == "cedar"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    vad = session["audio"]["input"]["turn_detection"]
    assert vad["type"] == "server_vad"
    assert vad["create_response"] is True
    assert vad["interrupt_response"] is False  # local barge-in owns cancellation
    assert vad["idle_timeout_ms"] is None  # no unsolicited empty-turn responses


async def test_input_is_resampled_pcm16_base64_with_ledger_mapping() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    events: list[Any] = []
    hold = asyncio.Event()

    async def consume() -> None:
        # 48 kHz source frames starting at t=1000ms: the encoder must resample
        # statefully to 24 kHz and the ledger must map buffer offsets back.
        async for event in adapter.stream(
            await _feed(_frames(12, rate=48000, start_ms=1000), hold=hold)
        ):
            events.append(event)

    task = asyncio.create_task(consume())
    await _drive(
        clock, lambda: len([p for p in conn.sent if p["type"] == "input_audio_buffer.append"]) >= 3
    )

    appends = [p for p in conn.sent if p["type"] == "input_audio_buffer.append"]
    total_samples = sum(len(base64.b64decode(p["audio"])) // 2 for p in appends)
    # 240 ms of source audio -> up to 5760 samples at 24 kHz; the resampler
    # tail still in flight explains the lower bound.
    assert 2000 <= total_samples <= 5800

    # Server reports speech starting 40 ms into the provider buffer: that maps
    # to source time 1000 + 40 = 1040 ms — regardless of local VAD opinions.
    conn.push(_speech_started(ms=40))
    await _drive(clock, lambda: any(isinstance(e, S2SSpeechStarted) for e in events))
    started = next(e for e in events if isinstance(e, S2SSpeechStarted))
    assert started.input_item_id == "item-1"
    assert abs(started.source_started_at_ms - 1040) <= 2

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_event_and_status_mapping_round_trip() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    pcm = b"\x00\x40" * 240
    for event in (
        _speech_started(),
        _committed(),
        _resp_created(),
        _delta(data=pcm),
        _audio_done(),
        _resp_done(status="completed"),
    ):
        conn.push(event)
    events = await _collect_session(adapter, conn, _frames(2), clock)

    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "S2SSpeechStarted",
        "S2SSpeechCommitted",
        "S2SResponseStarted",
        "S2SAudioChunk",
        "S2SContentDone",
        "S2SResponseDone",
    ]
    chunk = next(e for e in events if isinstance(e, S2SAudioChunk))
    assert chunk.sample_rate == 24000
    assert chunk.response_id == "resp-1" and chunk.item_id == "out-1"
    assert np.allclose(chunk.samples, pcm16_to_float32(pcm))
    done = next(e for e in events if isinstance(e, S2SResponseDone))
    assert done.status == "completed"


async def test_failed_status_with_reason_maps_through() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    for event in (
        _speech_started(),
        _committed(),
        _resp_created(),
        _resp_done(status="failed", reason="server_error"),
    ):
        conn.push(event)
    events = await _collect_session(adapter, conn, _frames(2), clock)
    done = next(e for e in events if isinstance(e, S2SResponseDone))
    assert done.status == "failed" and done.reason == "server_error"


async def test_interrupt_sends_contiguous_response_scoped_cancel_truncate() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    hold = asyncio.Event()
    events: list[Any] = []

    async def consume() -> None:
        async for event in adapter.stream(await _feed(_frames(50), hold=hold)):
            events.append(event)

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: len(conn.sent) >= 10)

    cursor = PlaybackCursor(
        response_id="resp-1", item_id="out-1", content_index=0, audio_end_ms=150
    )
    await adapter.interrupt(S2SInterruptTarget(response_id="resp-1", cursor=cursor))
    await _drive(clock, lambda: "response.cancel" in conn.sent_types())

    types = conn.sent_types()
    cancel_at = types.index("response.cancel")
    # The prioritized group is contiguous: truncate follows the cancel with no
    # newer append interleaved between them.
    assert types[cancel_at + 1] == "conversation.item.truncate"
    cancel = conn.sent[cancel_at]
    truncate = conn.sent[cancel_at + 1]
    assert cancel["response_id"] == "resp-1"  # never an unqualified cancel
    assert cancel["event_id"]
    assert truncate["item_id"] == "out-1"
    assert truncate["content_index"] == 0
    assert truncate["audio_end_ms"] == 150

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_interrupt_without_cursor_sends_cancel_only() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    hold = asyncio.Event()

    async def consume() -> None:
        async for _event in adapter.stream(await _feed(_frames(10), hold=hold)):
            pass

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: len(conn.sent) >= 3)
    await adapter.interrupt(S2SInterruptTarget(response_id="resp-9", cursor=None))
    await _drive(clock, lambda: "response.cancel" in conn.sent_types())
    assert "conversation.item.truncate" not in conn.sent_types()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_benign_control_error_for_our_cancel_is_tolerated() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    hold = asyncio.Event()
    events: list[Any] = []
    failed: list[BaseException] = []

    async def consume() -> None:
        try:
            async for event in adapter.stream(await _feed(_frames(10), hold=hold)):
                events.append(event)
        except BaseException as exc:
            failed.append(exc)

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: len(conn.sent) >= 2)
    await adapter.interrupt(S2SInterruptTarget(response_id="resp-1"))
    await _drive(clock, lambda: "response.cancel" in conn.sent_types())
    cancel_id = next(p["event_id"] for p in conn.sent if p["type"] == "response.cancel")

    # The documented no-active-response error referencing OUR cancel event.
    conn.push(_error("response_cancel_not_active", event_id=cancel_id))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not failed, "a benign correlated control error must not kill the session"

    # An unrelated error (different event id) is fatal.
    conn.push(_error("rate_limit_exceeded", event_id="someone-else"))
    await _drive(clock, lambda: task.done())
    await task
    assert failed and isinstance(failed[0], RealtimeError)
    assert "rate_limit_exceeded" in str(failed[0])


async def test_malformed_base64_delta_is_a_typed_failure() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    conn.push(_speech_started())
    conn.push(_resp_created())
    conn.push(
        SimpleNamespace(
            type="response.output_audio.delta",
            response_id="resp-1",
            item_id="out-1",
            output_index=0,
            content_index=0,
            delta="!!!not-base64!!!",
        )
    )
    with pytest.raises(RealtimeError, match="malformed"):
        await _collect_session(adapter, conn, _frames(2), clock)
    # No stray tasks survive the failure.
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


async def test_eof_no_speech_closes_without_commit_or_create() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    await _collect_session(adapter, conn, _frames(3), clock)
    types = conn.sent_types()
    assert "input_audio_buffer.commit" not in types
    assert "response.create" not in types
    assert conn.closed >= 1


async def test_eof_auto_commit_observed_suppresses_manual_duplicate() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    # The server auto-committed and responded before EOF.
    for event in (
        _speech_started(),
        _committed(),
        _resp_created(),
        _resp_done(status="completed"),
    ):
        conn.push(event)
    await _collect_session(adapter, conn, _frames(3), clock)
    types = conn.sent_types()
    assert "input_audio_buffer.commit" not in types, "no manual duplicate commit"
    assert "response.create" not in types


async def test_eof_pending_speech_manual_commit_then_single_create() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    conn.push(_speech_started())  # speech detected, never auto-committed

    events: list[Any] = []

    async def consume() -> None:
        async for event in adapter.stream(await _feed(_frames(3))):
            events.append(event)

    task = asyncio.create_task(consume())
    # After the settle window a manual, event-id-tagged commit goes out.
    await _drive(clock, lambda: "input_audio_buffer.commit" in conn.sent_types())
    commit = next(p for p in conn.sent if p["type"] == "input_audio_buffer.commit")
    assert commit["event_id"]
    assert "response.create" not in conn.sent_types(), "create must await the commit ack"
    # Acknowledge the commit: exactly one response.create follows.
    conn.push(_committed())
    await _drive(clock, lambda: "response.create" in conn.sent_types())
    conn.push(_resp_created())
    conn.push(_resp_done(status="completed"))
    await _drive(clock, lambda: task.done())
    await task
    assert conn.sent_types().count("response.create") == 1


async def test_eof_auto_commit_racing_settle_window_suppresses_manual() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock, vad_settle_ms=500)
    conn.push(_speech_started())

    events: list[Any] = []

    async def consume() -> None:
        async for event in adapter.stream(await _feed(_frames(3))):
            events.append(event)

    task = asyncio.create_task(consume())
    # While the settle window is pending, the server auto-commits + responds.
    await _drive(clock, lambda: any(isinstance(e, S2SSpeechStarted) for e in events))
    conn.push(_committed())
    conn.push(_resp_created())
    conn.push(_resp_done(status="completed"))
    await _drive(clock, lambda: task.done())
    await task
    assert "input_audio_buffer.commit" not in conn.sent_types()
    assert "response.create" not in conn.sent_types()


async def test_initial_connect_retries_but_never_reconnects_after_audio() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    attempts = 0

    async def flaky_connect() -> FakeConnection:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("first dial failed")
        return conn

    adapter = _adapter(conn, clock, connect=flaky_connect)
    await _collect_session(adapter, conn, _frames(2), clock)
    assert attempts == 2, "one bounded retry before any audio is sent"

    # After audio has flowed, a receiver failure is terminal (no reconnect).
    conn2 = FakeConnection()
    attempts2 = 0

    async def connect2() -> FakeConnection:
        nonlocal attempts2
        attempts2 += 1
        return conn2

    adapter2 = _adapter(conn2, clock, connect=connect2)
    hold = asyncio.Event()
    failures: list[BaseException] = []

    async def consume() -> None:
        try:
            async for _event in adapter2.stream(await _feed(_frames(10), hold=hold)):
                pass
        except BaseException as exc:
            failures.append(exc)

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: len(conn2.sent) >= 3)
    conn2._events.put_nowait(None)  # simulate a mid-session disconnect
    await _drive(clock, lambda: task.done())
    await task
    assert failures and isinstance(failures[0], RealtimeError)
    assert attempts2 == 1, "never a transparent reconnect after audio was sent"


async def test_no_api_key_material_in_outbound_payloads() -> None:
    clock = ManualClock()
    conn = FakeConnection()
    adapter = _adapter(conn, clock)
    for event in (_speech_started(), _committed(), _resp_created(), _resp_done()):
        conn.push(event)
    await _collect_session(adapter, conn, _frames(2), clock)
    blob = repr(conn.sent)
    assert "sk-" not in blob and "api_key" not in blob and "Authorization" not in blob
