"""Mocked-``sounddevice`` tests for :class:`MicSource` and :class:`SpeakerSink`.

No real audio hardware or the ``audio`` extra is required: a stub
``sounddevice`` module is injected into ``sys.modules`` and the PortAudio
callbacks are driven by hand with scripted ADC/DAC timestamps, so overflow,
status, calibration, gapless lookahead, partial-stop accounting, and stream
lifecycle are all deterministic.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import numpy as np
import pytest

from interpret_live.audio_io import AudioStreamError, MicSource, SpeakerSink, list_devices
from interpret_live.clock import ManualClock
from interpret_live.types import PlaybackRejectedError, TtsChunk

BASE_STREAM_TIME = 100.0  # deliberately non-zero: PortAudio time != clock time


class _Flags:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def __bool__(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:
        return self.text


class _TimeInfo:
    def __init__(self, adc: float = 0.0, dac: float = 0.0) -> None:
        self.inputBufferAdcTime = adc  # PortAudio field name
        self.outputBufferDacTime = dac


class _FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs.get("callback")
        self.time = BASE_STREAM_TIME
        self.started = 0
        self.aborted = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def abort(self) -> None:
        self.aborted += 1

    def close(self) -> None:
        self.closed += 1


def make_fake_sd(
    *,
    input_check_error: str | None = None,
    output_check_error: str | None = None,
) -> types.ModuleType:
    mod = types.ModuleType("sounddevice")
    streams: list[_FakeStream] = []
    mod.streams = streams  # type: ignore[attr-defined]

    def _input_stream(**kwargs: Any) -> _FakeStream:
        stream = _FakeStream(**kwargs)
        streams.append(stream)
        return stream

    def _output_stream(**kwargs: Any) -> _FakeStream:
        stream = _FakeStream(**kwargs)
        streams.append(stream)
        return stream

    def _check_input(**kwargs: Any) -> None:
        if input_check_error:
            raise ValueError(input_check_error)

    def _check_output(**kwargs: Any) -> None:
        if output_check_error:
            raise ValueError(output_check_error)

    def _query_devices(device: object = None, kind: object = None) -> Any:
        info = {
            "name": "fake-device",
            "max_input_channels": 2,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        }
        if device is None and kind is None:
            return [info, dict(info, name="fake-2", max_input_channels=0)]
        return info

    mod.InputStream = _input_stream  # type: ignore[attr-defined]
    mod.OutputStream = _output_stream  # type: ignore[attr-defined]
    mod.check_input_settings = _check_input  # type: ignore[attr-defined]
    mod.check_output_settings = _check_output  # type: ignore[attr-defined]
    mod.query_devices = _query_devices  # type: ignore[attr-defined]
    mod.default = types.SimpleNamespace(device=(0, 1))  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mod = make_fake_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return mod


async def _ticks(n: int = 6) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def _mic_block(n: int = 320, amp: float = 0.05) -> np.ndarray:
    return np.full((n, 1), amp, dtype=np.float32)


# ----- device enumeration -----------------------------------------------------


def test_list_devices_reports_defaults_directions_and_rates(
    fake_sd: types.ModuleType,
) -> None:
    infos = list_devices()
    assert [d.index for d in infos] == [0, 1]
    assert infos[0].is_default_input and not infos[0].is_default_output
    assert infos[1].is_default_output
    assert infos[0].default_samplerate == 48000.0
    assert infos[0].max_input_channels == 2
    assert infos[1].max_input_channels == 0


# ----- MicSource ----------------------------------------------------------------


async def test_mic_overflow_drops_oldest_without_raising(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    mic = MicSource(sample_rate=16000, frame_ms=20, clock=clock, queue_frames=4)
    frames_iter = aiter(mic.frames())
    first = asyncio.ensure_future(frames_iter.__anext__())
    await _ticks()  # generator opens the stream and parks on the queue
    stream = fake_sd.streams[-1]

    for i in range(10):
        stream.callback(_mic_block(), 320, _TimeInfo(adc=BASE_STREAM_TIME + i * 0.02), _Flags())
    await _ticks()

    frame = await first
    assert frame.sample_rate == 16000
    # 10 blocks were offered; the consumer had capacity 4 -> oldest 6 dropped.
    assert mic.dropped == 6
    await frames_iter.aclose()
    assert stream.closed == 1


async def test_mic_calibrates_adc_time_to_injected_clock(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    mic = MicSource(sample_rate=16000, frame_ms=20, clock=clock)
    frames_iter = aiter(mic.frames())
    first = asyncio.ensure_future(frames_iter.__anext__())
    await _ticks()
    stream = fake_sd.streams[-1]

    # ADC time is 0.5 s past the calibration base -> 500 ms in the clock domain,
    # even though the PortAudio clock (100.x) shares no epoch with the ManualClock.
    stream.callback(_mic_block(), 320, _TimeInfo(adc=BASE_STREAM_TIME + 0.5), _Flags())
    await _ticks()
    frame = await first
    assert frame.t_ms == 500
    await frames_iter.aclose()


async def test_mic_surfaces_callback_status_as_typed_error(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    mic = MicSource(sample_rate=16000, frame_ms=20, clock=clock)
    frames_iter = aiter(mic.frames())
    first = asyncio.ensure_future(frames_iter.__anext__())
    await _ticks()
    stream = fake_sd.streams[-1]

    stream.callback(_mic_block(), 320, _TimeInfo(adc=BASE_STREAM_TIME), _Flags("input overflow"))
    await _ticks()
    with pytest.raises(AudioStreamError, match="input overflow"):
        await first
    # The failed iterator still closed its stream.
    assert stream.closed == 1


async def test_mic_validates_device_before_opening(
    monkeypatch: pytest.MonkeyPatch, clock: ManualClock
) -> None:
    mod = make_fake_sd(input_check_error="unsupported samplerate")
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    mic = MicSource(sample_rate=96000, frame_ms=20, clock=clock)
    with pytest.raises(AudioStreamError, match="cannot capture mono"):
        await aiter(mic.frames()).__anext__()
    assert not mod.streams  # no stream was ever opened


async def test_mic_cancellation_closes_stream(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    mic = MicSource(sample_rate=16000, frame_ms=20, clock=clock)
    frames_iter = aiter(mic.frames())
    pending = asyncio.ensure_future(frames_iter.__anext__())
    await _ticks()
    stream = fake_sd.streams[-1]
    assert stream.started == 1

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await frames_iter.aclose()
    assert stream.aborted >= 1
    assert stream.closed == 1


# ----- SpeakerSink ---------------------------------------------------------------


def _tts(ms: int, *, rate: int = 16000, seg: int = 0, uid: str = "u1") -> TtsChunk:
    n = int(ms * rate / 1000)
    return TtsChunk(
        samples=np.full(n, 0.1, dtype=np.float32),
        sample_rate=rate,
        segment_index=seg,
        utterance_id=uid,
    )


async def test_adjacent_chunks_are_gapless_with_no_underrun(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    sink = SpeakerSink(device_rate=16000, clock=clock, ring_ms=1000)
    gen = sink.new_generation()
    h1 = await sink.schedule(gen, _tts(50, seg=0))
    h2 = await sink.schedule(gen, _tts(50, seg=1))
    stream = fake_sd.streams[-1]

    out1 = np.zeros((800, 1), dtype=np.float32)
    out2 = np.zeros((800, 1), dtype=np.float32)
    stream.callback(out1, 800, _TimeInfo(dac=BASE_STREAM_TIME + 0.01), _Flags())
    stream.callback(out2, 800, _TimeInfo(dac=BASE_STREAM_TIME + 0.06), _Flags())
    await _ticks()

    # Both chunks were consumed back-to-back: no zero-fill at their boundary.
    assert np.all(out1 == 0.1)
    assert np.all(out2 == 0.1)
    assert sink.underruns == 0

    # Advance past both DAC spans; the next callback resolves both receipts.
    clock.advance(120)
    stream.callback(np.zeros((160, 1), dtype=np.float32), 160, _TimeInfo(dac=0.0), _Flags())
    await _ticks()
    r1 = await h1.completed()
    r2 = await h2.completed()
    assert r1.completed and not r1.interrupted
    assert r2.completed and not r2.interrupted
    assert r1.source_samples_presented == 800
    assert r2.source_samples_presented == 800
    # Silence after everything drained is not an underrun.
    assert sink.underruns == 0
    await sink.drain()
    await sink.aclose()
    assert stream.closed == 1


async def test_first_audible_uses_dac_calibration_despite_clock_offset(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    sink = SpeakerSink(device_rate=16000, clock=clock)
    gen = sink.new_generation()
    handle = await sink.schedule(gen, _tts(100))
    stream = fake_sd.streams[-1]

    # The device consumes the block but will only present it at DAC time
    # base+0.2 s (200 ms of output latency).
    stream.callback(
        np.zeros((1600, 1), dtype=np.float32), 1600, _TimeInfo(dac=BASE_STREAM_TIME + 0.2), _Flags()
    )
    await _ticks()
    assert not handle.progress().first_audible_t_ms  # nothing audible yet

    clock.advance(250)  # now 50 ms into audible playback
    stream.callback(np.zeros((160, 1), dtype=np.float32), 160, _TimeInfo(dac=0.0), _Flags())
    await _ticks()
    receipt = await handle.started()
    # First audible at the calibrated DAC time: 200 ms in the injected domain.
    assert receipt.first_audible_t_ms == 200


async def test_stop_returns_only_dac_passed_samples_with_device_latency(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    sink = SpeakerSink(device_rate=16000, clock=clock)
    gen = sink.new_generation()
    handle = await sink.schedule(gen, _tts(100))
    stream = fake_sd.streams[-1]

    # Device consumed everything instantly into its buffer; playback of the
    # 1600 frames begins at DAC base+0.1.
    stream.callback(
        np.zeros((1600, 1), dtype=np.float32), 1600, _TimeInfo(dac=BASE_STREAM_TIME + 0.1), _Flags()
    )
    await _ticks()

    clock.advance(150)  # 50 ms of the audio has actually been presented
    snapshots = await sink.stop(gen)
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.interrupted
    # Only 50 ms @ 16 kHz = 800 samples had passed the DAC; the other 800 were
    # still in the device buffer and must not be counted.
    assert snap.source_samples_presented == 800
    assert stream.aborted == 1  # queued device audio dropped via the abort path
    receipt = await handle.completed()
    assert receipt.interrupted
    await sink.aclose()


async def test_blocked_schedule_rejected_on_stop_real_sink(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    sink = SpeakerSink(device_rate=16000, clock=clock, capacity=1)
    gen = sink.new_generation()
    await sink.schedule(gen, _tts(100, seg=0))

    rejected = asyncio.Event()

    async def blocked() -> None:
        try:
            await sink.schedule(gen, _tts(100, seg=1))
        except PlaybackRejectedError:
            rejected.set()

    task = asyncio.create_task(blocked())
    await _ticks()
    assert not task.done(), "schedule must block on full capacity"

    await sink.stop(gen)
    await _ticks()
    await task
    assert rejected.is_set()
    await sink.aclose()


async def test_second_generation_waits_for_first_to_drain(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    sink = SpeakerSink(device_rate=16000, clock=clock)
    gen1 = sink.new_generation()
    gen2 = sink.new_generation()
    await sink.schedule(gen1, _tts(50, uid="old"))
    stream = fake_sd.streams[-1]

    scheduled_new = asyncio.Event()

    async def new_turn() -> None:
        await sink.schedule(gen2, _tts(50, uid="new"))
        scheduled_new.set()

    task = asyncio.create_task(new_turn())
    await _ticks()
    assert not scheduled_new.is_set(), "generation 2 must wait for generation 1"

    # Present generation 1 fully.
    stream.callback(
        np.zeros((800, 1), dtype=np.float32), 800, _TimeInfo(dac=BASE_STREAM_TIME), _Flags()
    )
    clock.advance(60)
    stream.callback(np.zeros((160, 1), dtype=np.float32), 160, _TimeInfo(dac=0.0), _Flags())
    await _ticks()
    await task
    assert scheduled_new.is_set()
    await sink.aclose()


async def test_ring_backpressure_waits_for_consumption(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    # Ring holds only 50 ms @ 16 kHz = 800 frames; the chunk is 1600.
    sink = SpeakerSink(device_rate=16000, clock=clock, ring_ms=50)
    gen = sink.new_generation()

    done = asyncio.Event()

    async def scheduler() -> None:
        await sink.schedule(gen, _tts(100))
        done.set()

    task = asyncio.create_task(scheduler())
    await _ticks()
    stream = fake_sd.streams[-1]
    assert not done.is_set(), "schedule must block while the ring is full"

    # Device consumes 800 frames -> space frees -> the writer finishes.
    stream.callback(
        np.zeros((800, 1), dtype=np.float32), 800, _TimeInfo(dac=BASE_STREAM_TIME), _Flags()
    )
    await _ticks()
    await task
    assert done.is_set()
    await sink.aclose()


async def test_mid_stream_starvation_counts_underrun_and_zero_fills(
    fake_sd: types.ModuleType, clock: ManualClock
) -> None:
    sink = SpeakerSink(device_rate=16000, clock=clock, ring_ms=50)
    gen = sink.new_generation()

    task = asyncio.create_task(sink.schedule(gen, _tts(100)))
    await _ticks()
    stream = fake_sd.streams[-1]

    # The device asks for more than the ring currently holds while the chunk
    # is still mid-write: a genuine starvation gap.
    out = np.full((1000, 1), 7.0, dtype=np.float32)
    stream.callback(out, 1000, _TimeInfo(dac=BASE_STREAM_TIME), _Flags())
    assert sink.underruns == 1
    # The unfulfilled tail is zero-filled, never stale memory.
    assert np.all(out[800:] == 0.0)
    await _ticks()
    await task
    await sink.aclose()


async def test_output_device_validation_fails_before_stream_opens(
    monkeypatch: pytest.MonkeyPatch, clock: ManualClock
) -> None:
    mod = make_fake_sd(output_check_error="bad rate")
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    sink = SpeakerSink(device_rate=192000, clock=clock)
    with pytest.raises(AudioStreamError, match="cannot play mono"):
        await sink.schedule(sink.new_generation(), _tts(10))
    assert not mod.streams
