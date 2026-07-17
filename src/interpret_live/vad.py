"""Voice-activity detection and debounced barge-in onset detection.

* :class:`EnergyVAD` — RMS energy over a frame vs a threshold, with *hangover*
  (silence must persist ``hangover_ms`` before the VAD flips back to silence),
  smoothing isolated dropouts.
* :class:`BargeInDetector` — debounces a **speech onset**: it raises an interrupt
  only after ``onset_ms`` of *continuous* speech following a silence gap, so a
  single noisy frame never falsely interrupts in-flight TTS.

All thresholds/timings are explicit and tested on synthetic energy frames. The
detector uses the injected :class:`~interpret_live.clock.Clock` only for
timestamps; debounce is measured from frame timestamps, never via real sleeps.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .types import AudioFrame

__all__ = [
    "VAD",
    "BargeInDetector",
    "EndpointAction",
    "EnergyVAD",
    "UtteranceEndpointDetector",
    "rms",
]


def rms(samples: np.ndarray) -> float:
    """Root-mean-square amplitude of ``samples`` (0.0 for an empty frame)."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


@runtime_checkable
class VAD(Protocol):
    """Voice-activity detector over individual audio frames."""

    def is_speech(self, frame: AudioFrame) -> bool:
        """Return ``True`` if ``frame`` is classified as speech."""
        ...

    def reset(self) -> None:
        """Reset any internal hangover/smoothing state."""
        ...


class EnergyVAD:
    """RMS-energy VAD with a configurable threshold and hangover.

    A frame is speech when its RMS exceeds ``threshold``. Once speech is
    detected, the detector stays "in speech" until ``hangover_ms`` of continuous
    sub-threshold audio has elapsed, which bridges brief intra-word dips.

    Args:
        threshold: RMS amplitude above which a frame counts as speech.
        hangover_ms: Trailing silence required before flipping back to silence.
    """

    __slots__ = ("_hangover_ms", "_in_speech", "_silence_ms", "_threshold")

    def __init__(self, threshold: float = 0.02, hangover_ms: int = 200) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if hangover_ms < 0:
            raise ValueError(f"hangover_ms must be >= 0, got {hangover_ms}")
        self._threshold = threshold
        self._hangover_ms = hangover_ms
        self._in_speech = False
        self._silence_ms = 0

    @property
    def threshold(self) -> float:
        """The RMS speech threshold."""
        return self._threshold

    def is_speech(self, frame: AudioFrame) -> bool:
        """Classify ``frame``, applying hangover to bridge brief silences."""
        loud = rms(frame.samples) >= self._threshold
        if loud:
            self._in_speech = True
            self._silence_ms = 0
            return True
        # Sub-threshold frame.
        if self._in_speech:
            self._silence_ms += frame.duration_ms
            if self._silence_ms >= self._hangover_ms:
                self._in_speech = False
                return False
            return True  # still within hangover window
        return False

    def reset(self) -> None:
        self._in_speech = False
        self._silence_ms = 0


class BargeInDetector:
    """Detect a debounced speech onset on the *source* mic and fire a callback.

    An onset fires only after ``onset_ms`` of continuous speech that *follows* a
    period of silence — so the speaker resuming is detected, but the speaker's
    own ongoing speech (before any silence) and one-frame blips are not.

    Args:
        vad: The frame-level :class:`VAD` used for speech classification.
        onset_ms: Continuous-speech duration required to confirm an onset.
        clock: Injected clock (used only for the fired event's timestamp).

    The detector starts **disarmed** and arms only after it has observed a
    silence frame, so an onset requires a *speech → silence → speech* pattern.
    This means the speaker's very first utterance (speech from the start, with no
    preceding silence — e.g. before any TTS is playing) does **not** fire a
    spurious barge-in; only speech that *resumes* after a silence gap does.
    """

    __slots__ = ("_armed", "_clock", "_onset_ms", "_speech_run_ms", "_vad")

    def __init__(self, vad: VAD, onset_ms: int = 150, clock: object | None = None) -> None:
        if onset_ms < 0:
            raise ValueError(f"onset_ms must be >= 0, got {onset_ms}")
        self._vad = vad
        self._onset_ms = onset_ms
        self._clock = clock
        self._speech_run_ms = 0
        # Start disarmed: a fresh onset can fire only after silence has been seen,
        # so the initial speech run (no preceding silence) never barges in.
        self._armed = False

    def reset(self) -> None:
        """Reset debounce state; stays disarmed until silence is observed again."""
        self._vad.reset()
        self._speech_run_ms = 0
        self._armed = False

    def feed(self, frame: AudioFrame) -> bool:
        """Process one source frame; return ``True`` exactly on a debounced onset.

        After an onset fires, the detector disarms until silence returns, so a
        single sustained utterance fires at most one onset.
        """
        if self._vad.is_speech(frame):
            if not self._armed:
                return False
            self._speech_run_ms += frame.duration_ms
            if self._speech_run_ms >= self._onset_ms:
                self._armed = False
                self._speech_run_ms = 0
                return True
            return False
        # Silence: re-arm and reset the speech run.
        self._armed = True
        self._speech_run_ms = 0
        return False

    async def watch(
        self,
        frames: AsyncIterator[AudioFrame],
        on_barge_in: Callable[[AudioFrame], Awaitable[None]],
    ) -> None:
        """Consume ``frames`` and ``await on_barge_in(frame)`` on each onset.

        Runs until the frame iterator is exhausted or the task is cancelled.
        Used by the pipeline against a fanned-out copy of the source mic.
        """
        async for frame in frames:
            if self.feed(frame):
                await on_barge_in(frame)


@dataclass(frozen=True, slots=True)
class EndpointAction:
    """The endpoint detector's decision for one fed frame.

    ``end_reason`` (if any) closes the *previous* turn **before** this frame is
    considered, so a max-duration split followed by continued speech both ends
    the old turn and starts the next one in a single :meth:`feed` call without
    dropping the boundary frame.

    Attributes:
        end_reason: ``"silence"`` or ``"max_duration"`` when a turn just
            ended; ``None`` otherwise.
        started_turn_id: Set when a new turn begins at this frame.
        onset_t_ms: The new turn's immutable speech onset — the timestamp of
            its first VAD-positive frame (pre-roll frames come earlier but do
            not define the onset).
        frames: Frames to append to the *current* turn's utterance buffer (on
            a start this includes the retained pre-roll).
        partial_due: ``True`` when at least ``partial_interval_ms`` of new
            turn audio accumulated since the last partial tick.
    """

    end_reason: str | None = None
    started_turn_id: str | None = None
    onset_t_ms: int | None = None
    frames: tuple[AudioFrame, ...] = ()
    partial_due: bool = False


class UtteranceEndpointDetector:
    """Deterministic utterance endpointing around a frame-level :class:`VAD`.

    Owns turn lifecycle for the offline STT adapter: it retains ``pre_roll_ms``
    of leading audio, starts a turn on the first VAD-positive frame, keeps
    consuming the same live stream across turns, finalizes after
    ``end_silence_ms`` of trailing silence or at ``max_utterance_ms``, and
    paces partial-decode ticks every ``partial_interval_ms``. A max-duration
    split during sustained speech starts the next turn immediately (the
    boundary frame opens the new turn; nothing is dropped).

    Note: when composed with an :class:`EnergyVAD` hangover, trailing silence
    is measured *after* the hangover flips to silence, so the effective
    end-of-utterance delay is ``hangover_ms + end_silence_ms``.

    Args:
        vad: Frame-level VAD used for speech classification.
        pre_roll_ms: Leading audio retained before the detected onset.
        partial_interval_ms: Minimum new-audio interval between partial ticks.
        end_silence_ms: Trailing silence that finalizes a turn.
        max_utterance_ms: Hard cap; a longer turn is split at this duration.
    """

    __slots__ = (
        "_end_silence_ms",
        "_in_turn",
        "_max_utterance_ms",
        "_onset_t_ms",
        "_partial_acc_ms",
        "_partial_interval_ms",
        "_pre_roll",
        "_pre_roll_ms",
        "_silence_run_ms",
        "_turn_count",
        "_turn_ms",
        "_vad",
    )

    def __init__(
        self,
        vad: VAD,
        *,
        pre_roll_ms: int = 200,
        partial_interval_ms: int = 500,
        end_silence_ms: int = 500,
        max_utterance_ms: int = 30_000,
    ) -> None:
        if pre_roll_ms < 0:
            raise ValueError(f"pre_roll_ms must be >= 0, got {pre_roll_ms}")
        if partial_interval_ms <= 0:
            raise ValueError(f"partial_interval_ms must be > 0, got {partial_interval_ms}")
        if end_silence_ms < 0:
            raise ValueError(f"end_silence_ms must be >= 0, got {end_silence_ms}")
        if max_utterance_ms <= 0:
            raise ValueError(f"max_utterance_ms must be > 0, got {max_utterance_ms}")
        self._vad = vad
        self._pre_roll_ms = pre_roll_ms
        self._partial_interval_ms = partial_interval_ms
        self._end_silence_ms = end_silence_ms
        self._max_utterance_ms = max_utterance_ms
        self._pre_roll: deque[AudioFrame] = deque()
        self._in_turn = False
        self._turn_count = 0
        self._turn_ms = 0
        self._silence_run_ms = 0
        self._partial_acc_ms = 0
        self._onset_t_ms = 0

    @property
    def in_turn(self) -> bool:
        """``True`` while a turn is open (its buffer is accumulating)."""
        return self._in_turn

    def flush(self) -> str | None:
        """Close an open turn at source EOF; returns ``"eof"`` if one closed."""
        if not self._in_turn:
            return None
        self._reset_turn_state()
        return "eof"

    def _reset_turn_state(self) -> None:
        self._in_turn = False
        self._turn_ms = 0
        self._silence_run_ms = 0
        self._partial_acc_ms = 0
        self._pre_roll.clear()

    def _retain_pre_roll(self, frame: AudioFrame) -> None:
        self._pre_roll.append(frame)
        total = sum(f.duration_ms for f in self._pre_roll)
        while self._pre_roll and total - self._pre_roll[0].duration_ms >= self._pre_roll_ms:
            total -= self._pre_roll.popleft().duration_ms

    def _start_turn(self, frame: AudioFrame) -> tuple[str, int, tuple[AudioFrame, ...]]:
        self._turn_count += 1
        self._in_turn = True
        self._onset_t_ms = frame.t_ms
        frames = (*self._pre_roll, frame)
        self._pre_roll.clear()
        self._turn_ms = sum(f.duration_ms for f in frames)
        self._silence_run_ms = 0
        self._partial_acc_ms = self._turn_ms
        return f"turn-{self._turn_count}", frame.t_ms, frames

    def feed(self, frame: AudioFrame) -> EndpointAction:
        """Advance the state machine by one frame and return the decision."""
        speech = self._vad.is_speech(frame)

        if not self._in_turn:
            if not speech:
                self._retain_pre_roll(frame)
                return EndpointAction()
            turn_id, onset, frames = self._start_turn(frame)
            return EndpointAction(
                started_turn_id=turn_id,
                onset_t_ms=onset,
                frames=frames,
                partial_due=self._take_partial_tick(),
            )

        # Inside a turn: a max-duration split ends the old turn *before* this
        # frame, and continued speech immediately opens the next turn with the
        # boundary frame (no artificial silence gap, no dropped frame).
        if self._turn_ms + frame.duration_ms > self._max_utterance_ms:
            self._reset_turn_state()
            if speech:
                turn_id, onset, frames = self._start_turn(frame)
                return EndpointAction(
                    end_reason="max_duration",
                    started_turn_id=turn_id,
                    onset_t_ms=onset,
                    frames=frames,
                    partial_due=self._take_partial_tick(),
                )
            self._retain_pre_roll(frame)
            return EndpointAction(end_reason="max_duration")

        self._turn_ms += frame.duration_ms
        self._partial_acc_ms += frame.duration_ms
        if speech:
            self._silence_run_ms = 0
            return EndpointAction(frames=(frame,), partial_due=self._take_partial_tick())

        # Silence inside a turn: buffer it (it may be an intra-sentence pause)
        # and finalize once enough trailing silence has accumulated.
        self._silence_run_ms += frame.duration_ms
        if self._silence_run_ms >= self._end_silence_ms:
            self._reset_turn_state()
            self._retain_pre_roll(frame)
            return EndpointAction(end_reason="silence")
        return EndpointAction(frames=(frame,), partial_due=self._take_partial_tick())

    def _take_partial_tick(self) -> bool:
        if self._partial_acc_ms >= self._partial_interval_ms:
            self._partial_acc_ms = 0
            return True
        return False
