"""Deterministic tests for :class:`UtteranceEndpointDetector` (synthetic frames)."""

from __future__ import annotations

from helpers import frame
from interpret_live.types import AudioFrame
from interpret_live.vad import EnergyVAD, UtteranceEndpointDetector


def _detector(
    *,
    pre_roll_ms: int = 40,
    partial_interval_ms: int = 100,
    end_silence_ms: int = 100,
    max_utterance_ms: int = 30_000,
    hangover_ms: int = 0,
) -> UtteranceEndpointDetector:
    return UtteranceEndpointDetector(
        EnergyVAD(threshold=0.02, hangover_ms=hangover_ms),
        pre_roll_ms=pre_roll_ms,
        partial_interval_ms=partial_interval_ms,
        end_silence_ms=end_silence_ms,
        max_utterance_ms=max_utterance_ms,
    )


def _frames(pattern: str, *, start_ms: int = 0) -> list[AudioFrame]:
    """Build 20 ms frames from a pattern string: 's' = speech, '.' = silence."""
    return [
        frame(0.5 if ch == "s" else 0.0, t_ms=start_ms + i * 20, n=320)
        for i, ch in enumerate(pattern)
    ]


def test_silence_speech_silence_speech_yields_two_turns_with_onsets() -> None:
    det = _detector()
    events = [det.feed(f) for f in _frames("..ssssss........ssss.....")]
    starts = [e for e in events if e.started_turn_id is not None]
    ends = [e for e in events if e.end_reason is not None]
    assert [e.started_turn_id for e in starts] == ["turn-1", "turn-2"]
    # Onset is the first VAD-positive frame's timestamp, not the pre-roll start.
    assert starts[0].onset_t_ms == 40
    assert starts[1].onset_t_ms == 320
    assert [e.end_reason for e in ends] == ["silence", "silence"]


def test_pre_roll_frames_are_included_at_turn_start() -> None:
    det = _detector(pre_roll_ms=40)
    events = [det.feed(f) for f in _frames("....ss")]
    start = next(e for e in events if e.started_turn_id is not None)
    # 40 ms of pre-roll (two 20 ms frames) plus the onset frame itself.
    assert len(start.frames) == 3
    assert [f.t_ms for f in start.frames] == [40, 60, 80]
    assert start.onset_t_ms == 80


def test_max_duration_split_starts_next_turn_immediately_keeping_boundary() -> None:
    det = _detector(max_utterance_ms=100, end_silence_ms=100)
    # 10 continuous speech frames = 200 ms of sustained speech.
    events = [det.feed(f) for f in _frames("ssssssssss")]
    split = next(e for e in events if e.end_reason == "max_duration")
    # The split both ends the old turn and opens the next with the boundary
    # frame — no artificial silence gap, no dropped frame.
    assert split.started_turn_id == "turn-2"
    assert split.frames, "the boundary frame must open the next turn"
    starts = [e.started_turn_id for e in events if e.started_turn_id]
    assert starts == ["turn-1", "turn-2"]


def test_partial_ticks_pace_at_partial_interval() -> None:
    det = _detector(partial_interval_ms=100)
    events = [det.feed(f) for f in _frames("ssssssssssss")]  # 240 ms speech
    ticks = [i for i, e in enumerate(events) if e.partial_due]
    # One tick per 100 ms of accumulated turn audio (5 frames -> 100ms, ...).
    assert len(ticks) == 2
    assert ticks[0] < ticks[1]


def test_intra_turn_silence_shorter_than_end_silence_stays_in_turn() -> None:
    det = _detector(end_silence_ms=100)
    # 60 ms mid-utterance pause (3 frames) must not finalize.
    events = [det.feed(f) for f in _frames("..ssss...ssss")]
    assert all(e.end_reason is None for e in events)
    assert det.in_turn


def test_flush_closes_open_turn_at_eof() -> None:
    det = _detector()
    for f in _frames("..ssss"):
        det.feed(f)
    assert det.in_turn
    assert det.flush() == "eof"
    assert not det.in_turn
    assert det.flush() is None
