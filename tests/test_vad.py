"""VAD + barge-in tests: threshold/hangover, debounced onset.

Covers the EnergyVAD threshold and hangover behaviour and the BargeInDetector
debounce: a single noisy frame does NOT interrupt, a sustained ``onset_ms`` run
after silence does, and silence re-arms the detector for the next onset.
"""

from __future__ import annotations

import pytest

from helpers import frame
from interpret_live.clock import ManualClock
from interpret_live.vad import BargeInDetector, EnergyVAD, rms

# ----- EnergyVAD threshold ----------------------------------------------------


def test_rms_of_silence_is_zero() -> None:
    assert rms(frame(0.0).samples) == 0.0


def test_loud_frame_is_speech_quiet_frame_is_silence() -> None:
    vad = EnergyVAD(threshold=0.02, hangover_ms=0)
    assert vad.is_speech(frame(0.5)) is True
    vad.reset()
    assert vad.is_speech(frame(0.001)) is False


def test_threshold_boundary_inclusive() -> None:
    vad = EnergyVAD(threshold=0.1, hangover_ms=0)
    # A constant-amplitude frame's RMS equals the amplitude; exactly at threshold
    # counts as speech (>=).
    assert vad.is_speech(frame(0.1)) is True


# ----- EnergyVAD hangover -----------------------------------------------------


def test_hangover_bridges_brief_silence() -> None:
    vad = EnergyVAD(threshold=0.02, hangover_ms=40)
    assert vad.is_speech(frame(0.5, n=320)) is True  # 20ms loud frame
    # A single 20ms quiet frame is still within the 40ms hangover -> speech.
    assert vad.is_speech(frame(0.0, n=320)) is True
    # A second 20ms quiet frame reaches 40ms -> flips to silence.
    assert vad.is_speech(frame(0.0, n=320)) is False


def test_hangover_resets_on_new_speech() -> None:
    vad = EnergyVAD(threshold=0.02, hangover_ms=40)
    vad.is_speech(frame(0.5))  # speech
    vad.is_speech(frame(0.0))  # within hangover
    vad.is_speech(frame(0.5))  # speech again resets silence accumulator
    # Now one quiet frame should still be within hangover, not silence yet.
    assert vad.is_speech(frame(0.0)) is True


# ----- BargeInDetector debounce -----------------------------------------------


def _barge(onset_ms: int) -> BargeInDetector:
    return BargeInDetector(
        EnergyVAD(threshold=0.02, hangover_ms=0),
        onset_ms=onset_ms,
        clock=ManualClock(),
    )


def test_single_blip_does_not_interrupt() -> None:
    det = _barge(onset_ms=60)  # needs 60ms continuous speech
    # One 20ms loud frame, then silence: below the 60ms onset -> no interrupt.
    assert det.feed(frame(0.5, n=320)) is False
    assert det.feed(frame(0.0, n=320)) is False


def test_sustained_speech_triggers_onset() -> None:
    det = _barge(onset_ms=60)
    det.feed(frame(0.0, n=320))  # silence first -> arms the detector
    assert det.feed(frame(0.5, n=320)) is False  # 20ms
    assert det.feed(frame(0.5, n=320)) is False  # 40ms
    assert det.feed(frame(0.5, n=320)) is True  # 60ms -> onset fires


def test_onset_fires_once_until_silence_rearms() -> None:
    det = _barge(onset_ms=40)
    det.feed(frame(0.0, n=320))  # silence first -> arms the detector
    det.feed(frame(0.5, n=320))  # 20ms
    assert det.feed(frame(0.5, n=320)) is True  # 40ms -> fires
    # Continued speech does not re-fire (disarmed).
    assert det.feed(frame(0.5, n=320)) is False
    # Silence re-arms.
    det.feed(frame(0.0, n=320))
    det.feed(frame(0.5, n=320))  # 20ms
    assert det.feed(frame(0.5, n=320)) is True  # 40ms -> fires again


def test_silence_resets_partial_speech_run() -> None:
    det = _barge(onset_ms=60)
    det.feed(frame(0.5, n=320))  # 20ms
    det.feed(frame(0.0, n=320))  # silence: run resets
    det.feed(frame(0.5, n=320))  # 20ms (fresh)
    assert det.feed(frame(0.5, n=320)) is False  # only 40ms accumulated


async def test_watch_invokes_callback_on_onset() -> None:
    det = _barge(onset_ms=40)
    fired: list[int] = []

    async def gen():
        yield frame(0.0, n=320, t_ms=0)  # silence first -> arms the detector
        yield frame(0.5, n=320, t_ms=20)  # 20ms
        yield frame(0.5, n=320, t_ms=40)  # 40ms -> onset
        yield frame(0.0, n=320, t_ms=60)

    async def on_onset(_f) -> None:
        fired.append(_f.t_ms)

    await det.watch(gen(), on_onset)
    assert fired == [40]


# ----- Construction guards ----------------------------------------------------


def test_negative_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        EnergyVAD(threshold=-0.1)


def test_negative_hangover_raises() -> None:
    with pytest.raises(ValueError, match="hangover_ms must be >= 0"):
        EnergyVAD(hangover_ms=-1)


def test_negative_onset_raises() -> None:
    with pytest.raises(ValueError, match="onset_ms must be >= 0"):
        BargeInDetector(EnergyVAD(), onset_ms=-1)
