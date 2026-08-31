"""GeminiS2S tests against a scripted fake Live session (no network, no SDK).

Proves: exact session configuration, 16 kHz input encoding, the minted turn
identity and its speech/response event trio, onset timestamping (local VAD and
the provider's own signal), audio mapping and mime-rate parsing, multi-turn
re-entry of ``receive()``, every terminal turn status, locally-enforced
barge-in, the EOF ``audio_stream_end`` machine, reconnect policy, and clean
teardown.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from interpret_live.audio_codec import pcm16_to_float32
from interpret_live.backends import S2S
from interpret_live.backends.gemini import (
    DEFAULT_GEMINI_MODEL,
    GeminiError,
    GeminiS2S,
    _duration_to_ms,
    _rate_from_mime,
)
from interpret_live.clock import ManualClock
from interpret_live.types import (
    AudioFrame,
    S2SAudioChunk,
    S2SContentDone,
    S2SInterruptTarget,
    S2SResponseDone,
    S2SResponseStarted,
    S2SSpeechCommitted,
    S2SSpeechStarted,
)

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")


class FakeSession:
    """A scripted stand-in for the SDK's ``AsyncSession``.

    ``receive()`` deliberately returns at every ``turn_complete``, exactly as
    the real one does, so the adapter's outer re-entry loop is exercised.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = 0
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self.send_gate: asyncio.Event | None = None

    async def send_realtime_input(self, **payload: Any) -> None:
        if self.send_gate is not None:
            await self.send_gate.wait()
        self.sent.append(payload)

    def push(self, message: Any) -> None:
        self._events.put_nowait(message)

    async def close(self) -> None:
        self.closed += 1
        self._events.put_nowait(None)

    async def receive(self) -> Any:
        while True:
            item = await self._events.get()
            if item is None:
                raise ConnectionError("connection closed")
            yield item
            content = getattr(item, "server_content", None)
            if content is not None and getattr(content, "turn_complete", None):
                return

    # -- helpers ---------------------------------------------------------------

    def audio_payloads(self) -> list[dict[str, Any]]:
        return [p["audio"] for p in self.sent if "audio" in p]


def _adapter(
    session: FakeSession,
    clock: ManualClock,
    captured: dict[str, Any] | None = None,
    **overrides: Any,
) -> GeminiS2S:
    async def connect(config: dict[str, Any]) -> FakeSession:
        if captured is not None:
            captured.update(config)
        return session

    kwargs: dict[str, Any] = {
        "source_lang": "en",
        "target_lang": "es",
        "clock": clock,
        "connect": connect,
        "vad_settle_ms": 100,
        "final_response_timeout_ms": 200,
        "send_timeout_s": 5.0,
    }
    kwargs.update(overrides)
    return GeminiS2S(**kwargs)


def _frames(
    count: int, *, rate: int = 16000, ms: int = 20, start_ms: int = 0, level: float = 0.4
) -> list[AudioFrame]:
    n = int(ms * rate / 1000)
    return [
        AudioFrame(
            samples=np.full(n, level, dtype=np.float32), sample_rate=rate, t_ms=start_ms + i * ms
        )
        for i in range(count)
    ]


async def _feed(
    frames: list[AudioFrame],
    *,
    hold: asyncio.Event | None = None,
    drained: asyncio.Event | None = None,
) -> Any:
    async def gen() -> Any:
        for frame in frames:
            yield frame
            await asyncio.sleep(0)
        if drained is not None:
            drained.set()
        if hold is not None:
            await hold.wait()

    return gen()


async def _drive(clock: ManualClock, cond: Any, *, rounds: int = 600) -> None:
    """Yield until ``cond`` holds, advancing the manual clock when parked."""
    for _ in range(rounds):
        for _ in range(6):
            await asyncio.sleep(0)
            if cond():
                return
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    raise AssertionError("condition never became true")


# -- message builders (shaped like google.genai's LiveServerMessage) ------------


def _content(**fields: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "model_turn": None,
        "turn_complete": None,
        "turn_complete_reason": None,
        "generation_complete": None,
        "interrupted": None,
        "input_transcription": None,
        "output_transcription": None,
    }
    base.update(fields)
    return SimpleNamespace(
        setup_complete=None,
        server_content=SimpleNamespace(**base),
        voice_activity=None,
        go_away=None,
    )


def _model_turn(data: bytes = b"\x00\x01" * 10, rate: int = 24000) -> SimpleNamespace:
    inline = SimpleNamespace(data=data, mime_type=f"audio/pcm;rate={rate}")
    return SimpleNamespace(parts=[SimpleNamespace(inline_data=inline)])


def _audio_msg(data: bytes = b"\x00\x01" * 10, rate: int = 24000) -> SimpleNamespace:
    return _content(model_turn=_model_turn(data, rate))


def _turn_complete(reason: str | None = None) -> SimpleNamespace:
    return _content(turn_complete=True, turn_complete_reason=reason)


def _voice_activity(kind: str = "ACTIVITY_START", offset: str | None = "1.0s") -> SimpleNamespace:
    return SimpleNamespace(
        setup_complete=None,
        server_content=None,
        go_away=None,
        voice_activity=SimpleNamespace(voice_activity_type=kind, audio_offset=offset),
    )


async def _collect(
    adapter: GeminiS2S, frames: list[Any], session: FakeSession, clock: ManualClock
) -> list[Any]:
    """Feed frames, let the EOF machine close the session, return the events."""
    events: list[Any] = []

    async def consume() -> None:
        async for event in adapter.stream(await _feed(frames)):
            events.append(event)

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: task.done())
    await task
    return events


# -- pure helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("mime", "expected"),
    [
        ("audio/pcm;rate=24000", 24000),
        ("audio/pcm; rate=16000", 16000),
        ("audio/pcm", 24000),  # documented Live default when the rate is absent
        ("audio/l16;rate=8000", 8000),
    ],
)
def test_rate_from_mime(mime: str, expected: int) -> None:
    assert _rate_from_mime(mime) == expected


@pytest.mark.parametrize("mime", ["audio/mpeg;rate=24000", "text/plain", ""])
def test_undecodable_audio_mime_is_rejected(mime: str) -> None:
    """A codec we cannot decode is an error, never noise handed to the sink."""
    with pytest.raises(GeminiError, match="unsupported response audio mime type"):
        _rate_from_mime(mime)


@pytest.mark.parametrize("mime", ["audio/pcm;rate=abc", "audio/pcm;rate=0"])
def test_malformed_rate_is_rejected(mime: str) -> None:
    with pytest.raises(GeminiError):
        _rate_from_mime(mime)


@pytest.mark.parametrize(
    ("value", "expected"), [("1.5s", 1500), ("0s", 0), ("2", 2000), (None, None), ("abc", None)]
)
def test_duration_parsing(value: object, expected: int | None) -> None:
    assert _duration_to_ms(value) == expected


# -- construction ---------------------------------------------------------------


def test_satisfies_the_s2s_protocol() -> None:
    adapter = _adapter(FakeSession(), ManualClock())
    assert isinstance(adapter, S2S)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"voice": ""},
        {"source_lang": ""},
        {"target_lang": ""},
        {"connect_attempts": 0},
    ],
)
def test_invalid_construction_fails_fast(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _adapter(FakeSession(), ManualClock(), **overrides)


def test_default_model_is_the_tested_live_model() -> None:
    assert DEFAULT_GEMINI_MODEL.startswith("gemini-live")


# -- session configuration ------------------------------------------------------


async def test_session_config_is_translation_only_audio_with_local_barge_in() -> None:
    session, clock, captured = FakeSession(), ManualClock(), {}
    adapter = _adapter(session, clock, captured, voice="Puck", target_lang="fr")

    await _collect(adapter, [], session, clock)

    assert captured["response_modalities"] == ["AUDIO"]
    assert "only translate" in captured["system_instruction"]
    assert "en" in captured["system_instruction"] and "fr" in captured["system_instruction"]
    voice_config = captured["speech_config"]["voice_config"]["prebuilt_voice_config"]
    assert voice_config["voice_name"] == "Puck"
    assert captured["speech_config"]["language_code"] == "fr"
    realtime = captured["realtime_input_config"]
    # Server VAD detects turns; it must never cancel one on its own.
    assert realtime["automatic_activity_detection"] == {"disabled": False}
    assert realtime["activity_handling"] == "NO_INTERRUPTION"
    assert "translation_config" not in captured


async def test_native_translation_is_opt_in() -> None:
    session, clock, captured = FakeSession(), ManualClock(), {}
    adapter = _adapter(session, clock, captured, native_translation=True, target_lang="de")

    await _collect(adapter, [], session, clock)

    assert captured["translation_config"] == {
        "target_language_code": "de",
        "echo_target_language": False,
    }


# -- input path -----------------------------------------------------------------


async def test_source_audio_is_sent_as_16k_pcm16_blobs() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    frames = _frames(3, rate=48000, ms=20)

    await _collect(adapter, frames, session, clock)

    payloads = session.audio_payloads()
    assert payloads, "no audio reached the provider"
    assert {p["mime_type"] for p in payloads} == {"audio/pcm;rate=16000"}
    assert all(isinstance(p["data"], bytes) for p in payloads)
    # 60 ms of source at 16 kHz is 960 samples => 1920 PCM16 bytes (± resampler
    # warm-up, which the EOF flush returns).
    assert sum(len(p["data"]) for p in payloads) == pytest.approx(1920, abs=64)


async def test_sample_rate_change_mid_stream_is_rejected() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    frames = _frames(1, rate=16000) + _frames(1, rate=48000, start_ms=20)

    with pytest.raises(GeminiError, match="sample rate changed mid-stream"):
        await _collect(adapter, frames, session, clock)


async def test_audio_stream_end_is_sent_exactly_once_after_the_last_block() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    await _collect(adapter, _frames(2), session, clock)

    ends = [p for p in session.sent if p.get("audio_stream_end")]
    assert len(ends) == 1
    assert session.sent[-1] is ends[0]


async def test_no_audio_means_no_stream_end() -> None:
    """An empty source never claims to have closed an input stream."""
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    await _collect(adapter, [], session, clock)

    assert session.sent == []


# -- turn lifecycle -------------------------------------------------------------


async def _run_with_script(
    adapter: GeminiS2S,
    session: FakeSession,
    frames: list[AudioFrame],
    script: list[Any],
    clock: ManualClock,
) -> list[Any]:
    """Send every frame first, then play ``script`` at the provider, and collect.

    Frames go out before the script so an onset the encoder observed is already
    recorded when the provider opens its turn — the real ordering.
    """
    events: list[Any] = []
    hold, drained = asyncio.Event(), asyncio.Event()

    async def consume() -> None:
        async for event in adapter.stream(await _feed(frames, hold=hold, drained=drained)):
            events.append(event)

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: drained.is_set())
    for _ in range(6):
        await asyncio.sleep(0)
    for message in script:
        session.push(message)
    hold.set()
    await _drive(clock, lambda: task.done())
    await task
    return events


async def test_a_provider_turn_opens_with_speech_committed_and_response_started() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(
        adapter, session, _frames(2), [_audio_msg(), _turn_complete()], clock
    )

    kinds = [type(e) for e in events]
    assert kinds == [
        S2SSpeechStarted,
        S2SSpeechCommitted,
        S2SResponseStarted,
        S2SAudioChunk,
        S2SResponseDone,
    ]
    started = events[0]
    assert isinstance(started, S2SSpeechStarted)
    response = events[2]
    assert isinstance(response, S2SResponseStarted)
    # The response is attributed to the input turn it answers.
    assert response.input_item_id == started.input_item_id
    assert events[-1].status == "completed"


async def test_response_audio_carries_provenance_and_the_mime_rate() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    payload = b"\x10\x20" * 8

    events = await _run_with_script(
        adapter, session, _frames(2), [_audio_msg(payload, rate=16000), _turn_complete()], clock
    )

    chunk = next(e for e in events if isinstance(e, S2SAudioChunk))
    assert chunk.sample_rate == 16000
    assert chunk.response_id == "gemini-resp-1"
    assert chunk.item_id == "gemini-item-1"
    assert np.allclose(chunk.samples, pcm16_to_float32(payload))


async def test_generation_complete_closes_only_the_content_stream() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(
        adapter,
        session,
        _frames(2),
        [_audio_msg(), _content(generation_complete=True), _turn_complete()],
        clock,
    )

    content_done = [e for e in events if isinstance(e, S2SContentDone)]
    assert len(content_done) == 1
    # The response is still open at that point: it ends only on turn_complete.
    assert events.index(content_done[0]) < len(events) - 1


async def test_one_session_spans_several_turns_with_distinct_ids() -> None:
    """`receive()` returns at every turn_complete; the adapter re-enters it."""
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(
        adapter,
        session,
        _frames(2),
        [_audio_msg(), _turn_complete(), _audio_msg(), _turn_complete()],
        clock,
    )

    ids = [e.response_id for e in events if isinstance(e, S2SResponseStarted)]
    assert ids == ["gemini-resp-1", "gemini-resp-2"]
    assert [e.input_item_id for e in events if isinstance(e, S2SSpeechStarted)] == [
        "gemini-in-1",
        "gemini-in-2",
    ]


async def test_provider_interruption_ends_the_turn_as_cancelled() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(
        adapter,
        session,
        _frames(2),
        [_audio_msg(), _content(interrupted=True), _turn_complete()],
        clock,
    )

    # The Live API reports `interrupted` and then `turn_complete` for the SAME
    # turn: exactly one terminal status, and no phantom second turn after it.
    done = [e for e in events if isinstance(e, S2SResponseDone)]
    assert [d.status for d in done] == ["cancelled"]
    assert done[0].reason == "the provider interrupted the turn"
    assert len([e for e in events if isinstance(e, S2SResponseStarted)]) == 1


async def test_an_abnormal_turn_reason_is_reported_as_failed() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(
        adapter, session, _frames(2), [_audio_msg(), _turn_complete("BLOCKLIST")], clock
    )

    done = next(e for e in events if isinstance(e, S2SResponseDone))
    assert (done.status, done.reason) == ("failed", "BLOCKLIST")


async def test_a_benign_turn_reason_is_still_a_natural_completion() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(
        adapter, session, _frames(2), [_audio_msg(), _turn_complete("NEED_MORE_INPUT")], clock
    )

    assert next(e for e in events if isinstance(e, S2SResponseDone)).status == "completed"


async def test_a_turn_that_produces_no_audio_is_still_a_complete_turn() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(adapter, session, _frames(2), [_turn_complete()], clock)

    assert [type(e) for e in events] == [
        S2SSpeechStarted,
        S2SSpeechCommitted,
        S2SResponseStarted,
        S2SResponseDone,
    ]


async def test_input_transcription_alone_never_opens_a_turn() -> None:
    """Input-side text can arrive while the model is still listening."""
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    events = await _run_with_script(
        adapter,
        session,
        _frames(2),
        [_content(input_transcription=SimpleNamespace(text="hola")), _turn_complete()],
        clock,
    )

    assert [type(e) for e in events][:1] == [S2SSpeechStarted]
    assert len([e for e in events if isinstance(e, S2SResponseStarted)]) == 1


# -- speech onsets --------------------------------------------------------------


async def test_the_utterance_onset_comes_from_the_local_speech_transition() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    # 40 ms of silence, then speech starting at t=40 ms.
    frames = _frames(2, level=0.0) + _frames(3, start_ms=40, level=0.5)

    events = await _run_with_script(
        adapter, session, frames, [_audio_msg(), _turn_complete()], clock
    )

    started = events[0]
    assert isinstance(started, S2SSpeechStarted)
    assert started.source_started_at_ms == 40


async def test_a_provider_activity_signal_overrides_the_local_estimate() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    frames = _frames(4, start_ms=1000, level=0.5)

    events = await _run_with_script(
        adapter,
        session,
        frames,
        [_voice_activity(offset="0.5s"), _audio_msg(), _turn_complete()],
        clock,
    )

    started = events[0]
    assert isinstance(started, S2SSpeechStarted)
    # 500 ms into the stream that began at the first frame's t_ms.
    assert started.source_started_at_ms == 1500


async def test_a_malformed_activity_offset_falls_back_to_the_local_onset() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    frames = _frames(2, level=0.0) + _frames(3, start_ms=40, level=0.5)

    events = await _run_with_script(
        adapter,
        session,
        frames,
        [_voice_activity(offset="???"), _audio_msg(), _turn_complete()],
        clock,
    )

    started = events[0]
    assert isinstance(started, S2SSpeechStarted)
    assert started.source_started_at_ms == 40


async def test_an_activity_end_signal_does_not_move_the_onset() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    frames = _frames(2, level=0.0) + _frames(3, start_ms=40, level=0.5)

    events = await _run_with_script(
        adapter,
        session,
        frames,
        [_voice_activity("ACTIVITY_END", "9.0s"), _audio_msg(), _turn_complete()],
        clock,
    )

    started = events[0]
    assert isinstance(started, S2SSpeechStarted)
    assert started.source_started_at_ms == 40


# -- barge-in -------------------------------------------------------------------


async def test_interrupt_drops_the_abandoned_response_audio_but_keeps_its_lifecycle() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    events: list[Any] = []
    hold = asyncio.Event()

    async def consume() -> None:
        async for event in adapter.stream(await _feed(_frames(2), hold=hold)):
            events.append(event)

    task = asyncio.create_task(consume())
    session.push(_audio_msg())
    await _drive(clock, lambda: any(isinstance(e, S2SAudioChunk) for e in events))

    await adapter.interrupt(S2SInterruptTarget(response_id="gemini-resp-1"))
    session.push(_audio_msg(b"\x7f\x7f" * 8))  # generated into the void
    session.push(_turn_complete())
    hold.set()
    await _drive(clock, lambda: task.done())
    await task

    # Exactly the pre-interrupt chunk survives; the turn still closes cleanly,
    # so the harness never leaks the abandoned response's state.
    assert len([e for e in events if isinstance(e, S2SAudioChunk)]) == 1
    assert [e.status for e in events if isinstance(e, S2SResponseDone)] == ["completed"]
    # Nothing extra was sent to the provider: the Live API has no cancel to send.
    assert all("audio" in p or "audio_stream_end" in p for p in session.sent)


async def test_a_later_turn_is_unaffected_by_an_earlier_interrupt() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    events: list[Any] = []
    hold = asyncio.Event()

    async def consume() -> None:
        async for event in adapter.stream(await _feed(_frames(2), hold=hold)):
            events.append(event)

    task = asyncio.create_task(consume())
    session.push(_audio_msg())
    await _drive(clock, lambda: any(isinstance(e, S2SAudioChunk) for e in events))

    await adapter.interrupt(S2SInterruptTarget(response_id="gemini-resp-2"))
    session.push(_turn_complete())
    session.push(_audio_msg())
    session.push(_turn_complete())
    hold.set()
    await _drive(clock, lambda: task.done())
    await task

    # The interrupt named the turn that had not started yet, so only that one
    # loses its audio: abandonment is response-scoped, never session-wide.
    chunks = [e for e in events if isinstance(e, S2SAudioChunk)]
    assert [c.response_id for c in chunks] == ["gemini-resp-1"]
    assert [e.response_id for e in events if isinstance(e, S2SResponseStarted)] == [
        "gemini-resp-1",
        "gemini-resp-2",
    ]


async def test_interrupt_before_the_session_started_is_an_error() -> None:
    adapter = _adapter(FakeSession(), ManualClock())
    with pytest.raises(GeminiError, match="before the Gemini Live session started"):
        await adapter.interrupt(S2SInterruptTarget(response_id="gemini-resp-1"))


# -- EOF and teardown -----------------------------------------------------------


async def test_eof_waits_for_an_in_flight_turn_before_closing() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock, final_response_timeout_ms=30_000)
    events: list[Any] = []

    async def consume() -> None:
        async for event in adapter.stream(await _feed(_frames(2))):
            events.append(event)

    task = asyncio.create_task(consume())
    session.push(_audio_msg())
    await _drive(clock, lambda: any(isinstance(e, S2SAudioChunk) for e in events))
    for _ in range(20):
        await asyncio.sleep(0)

    # The source is exhausted, but the turn is still generating: closing here
    # would cut the translation off mid-sentence.
    assert not task.done()
    session.push(_turn_complete())
    await _drive(clock, lambda: task.done())
    await task

    assert isinstance(events[-1], S2SResponseDone)
    assert session.closed == 1


async def test_the_session_is_closed_exactly_once() -> None:
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)

    await _collect(adapter, _frames(2), session, clock)

    assert session.closed == 1


async def test_a_connection_failure_is_retried_then_reported() -> None:
    attempts = 0

    async def connect(_config: dict[str, Any]) -> Any:
        nonlocal attempts
        attempts += 1
        raise OSError("no route to host")

    adapter = GeminiS2S(clock=ManualClock(), connect=connect, connect_attempts=3)

    with pytest.raises(GeminiError, match="failed to open the Gemini Live connection"):
        _ = [event async for event in adapter.stream(await _feed([]))]
    assert attempts == 3


async def test_a_disconnect_mid_session_is_terminal() -> None:
    """No transparent reconnect: replaying input could duplicate speech."""
    session, clock = FakeSession(), ManualClock()
    adapter = _adapter(session, clock)
    hold = asyncio.Event()

    async def consume() -> None:
        async for _event in adapter.stream(await _feed(_frames(2), hold=hold)):
            pass

    task = asyncio.create_task(consume())
    session.push(None)  # the fake session raises ConnectionError on None
    await _drive(clock, lambda: task.done())
    hold.set()

    with pytest.raises(GeminiError, match="Gemini Live connection failed"):
        await task


# -- the real SDK entry point (stubbed module, still no network) -----------------


async def test_the_default_connect_factory_never_takes_a_key_and_closes_its_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only SDK-touching code path: `genai.Client()` + `aio.live.connect`."""
    import sys
    import types

    seen: dict[str, Any] = {}
    session = FakeSession()

    class _Manager:
        async def __aenter__(self) -> FakeSession:
            return session

        async def __aexit__(self, *_exc: object) -> None:
            seen["exited"] = True

    class _Live:
        def connect(self, *, model: str, config: dict[str, Any]) -> _Manager:
            seen["model"] = model
            seen["config"] = config
            return _Manager()

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            seen["client_kwargs"] = kwargs
            self.aio = SimpleNamespace(live=_Live())

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = _Client  # type: ignore[attr-defined]
    google.genai = genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    clock = ManualClock()
    adapter = GeminiS2S(clock=clock, target_lang="fr", vad_settle_ms=10)
    assert isinstance(adapter, S2S)

    events: list[Any] = []

    async def consume() -> None:
        async for event in adapter.stream(await _feed([])):
            events.append(event)

    task = asyncio.create_task(consume())
    await _drive(clock, lambda: task.done())
    await task

    # No credential ever crosses the constructor: the SDK reads the environment.
    assert seen["client_kwargs"] == {}
    assert seen["model"] == DEFAULT_GEMINI_MODEL
    assert seen["config"]["speech_config"]["language_code"] == "fr"
    assert seen["exited"] is True
    assert session.closed == 1
    assert events == []
