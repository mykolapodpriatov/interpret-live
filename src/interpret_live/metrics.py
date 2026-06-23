"""Deterministic latency/quality metrics derived from the event log.

The pipeline appends :class:`~interpret_live.types.MetricEvent` records (with
injected-clock timestamps) to an in-memory log. :class:`MetricsLog` derives, per
utterance and overall:

* **first-audio-out latency** — ``t(first_tts_out) − t(utterance_start)``: how
  long after the user starts speaking the listener first hears target audio
  (simultaneity).
* **commit lag** — time from utterance start to the *first* stabilizer commit.
* **retraction count** — number of audio retractions; ``0`` on the stable path
  by construction (the committed prefix never retracts).
* **barge-in stop time** — ``t(sink_stopped) − t(interrupt)``: how promptly
  in-flight TTS halts on barge-in.
* **post_commit_disagreement** — count of disagreement events (a tuning signal).

Everything is computed from the log, so metrics are reproducible in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import MetricEvent

__all__ = ["MetricsLog", "MetricsReport", "UtteranceMetrics"]


@dataclass(frozen=True, slots=True)
class UtteranceMetrics:
    """Per-utterance derived metrics (``None`` where the event was absent)."""

    utterance_id: str
    first_audio_out_ms: int | None
    commit_lag_ms: int | None
    barge_in_stop_ms: int | None
    retraction_count: int
    post_commit_disagreement: int


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """Aggregate report across all utterances in a run."""

    utterances: tuple[UtteranceMetrics, ...]
    total_retractions: int
    total_post_commit_disagreement: int

    @property
    def max_first_audio_out_ms(self) -> int | None:
        """Worst (largest) first-audio-out latency across utterances."""
        vals = [u.first_audio_out_ms for u in self.utterances if u.first_audio_out_ms is not None]
        return max(vals) if vals else None

    @property
    def max_barge_in_stop_ms(self) -> int | None:
        """Worst (largest) barge-in-stop time across utterances."""
        vals = [u.barge_in_stop_ms for u in self.utterances if u.barge_in_stop_ms is not None]
        return max(vals) if vals else None


class MetricsLog:
    """An append-only, in-memory log of :class:`MetricEvent` with derivations."""

    __slots__ = ("_events", "_retractions", "_retractions_by_utt")

    def __init__(self) -> None:
        self._events: list[MetricEvent] = []
        # Audio retractions are tracked explicitly; on the stable path this stays
        # 0 (the committed prefix never retracts, so no spoken audio is recalled).
        self._retractions = 0
        # Per-utterance retraction tallies, so ``for_utterance`` reports a real
        # count rather than a hardcoded 0 (retractions attributed to an utterance
        # are recorded with its id).
        self._retractions_by_utt: dict[str, int] = {}

    @property
    def events(self) -> tuple[MetricEvent, ...]:
        """All recorded events, in append order."""
        return tuple(self._events)

    @property
    def retraction_count(self) -> int:
        """Total audio retractions recorded (``0`` on the stable path)."""
        return self._retractions

    def append(self, event: MetricEvent) -> None:
        """Record a metric event."""
        self._events.append(event)

    def record_retraction(self, count: int = 1, *, utterance_id: str | None = None) -> None:
        """Record ``count`` audio retractions (used only if audio is recalled).

        When ``utterance_id`` is given, the retractions are also attributed to
        that utterance so :meth:`for_utterance` reports them per-utterance; the
        aggregate :attr:`retraction_count` always includes them.
        """
        self._retractions += count
        if utterance_id is not None:
            self._retractions_by_utt[utterance_id] = (
                self._retractions_by_utt.get(utterance_id, 0) + count
            )

    def _utterance_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for ev in self._events:
            seen.setdefault(ev.utterance_id, None)
        return list(seen)

    def _first(self, uid: str, kind: str) -> MetricEvent | None:
        for ev in self._events:
            if ev.utterance_id == uid and ev.kind == kind:
                return ev
        return None

    def for_utterance(self, uid: str) -> UtteranceMetrics:
        """Derive :class:`UtteranceMetrics` for utterance ``uid``."""
        start = self._first(uid, "utterance_start")
        first_tts = self._first(uid, "first_tts_out")
        first_commit = self._first(uid, "commit")
        interrupt = self._first(uid, "interrupt")
        stopped = self._first(uid, "sink_stopped")

        first_audio = first_tts.t_ms - start.t_ms if (start and first_tts) else None
        commit_lag = first_commit.t_ms - start.t_ms if (start and first_commit) else None
        barge_in = stopped.t_ms - interrupt.t_ms if (interrupt and stopped) else None
        disagreements = sum(
            1
            for ev in self._events
            if ev.utterance_id == uid and ev.kind == "post_commit_disagreement"
        )
        return UtteranceMetrics(
            utterance_id=uid,
            first_audio_out_ms=first_audio,
            commit_lag_ms=commit_lag,
            barge_in_stop_ms=barge_in,
            retraction_count=self._retractions_by_utt.get(uid, 0),
            post_commit_disagreement=disagreements,
        )

    def report(self) -> MetricsReport:
        """Aggregate per-utterance metrics into a :class:`MetricsReport`."""
        utts = tuple(self.for_utterance(uid) for uid in self._utterance_ids())
        total_disagree = sum(u.post_commit_disagreement for u in utts)
        return MetricsReport(
            utterances=utts,
            total_retractions=self._retractions,
            total_post_commit_disagreement=total_disagree,
        )
