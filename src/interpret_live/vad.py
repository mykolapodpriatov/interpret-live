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

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol, runtime_checkable

import numpy as np

from .types import AudioFrame

__all__ = ["VAD", "BargeInDetector", "EnergyVAD", "rms"]


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
