"""Offline benchmark harness: replay a scripted fixture through fake backends.

Shared by the CLI ``bench`` command and the ``examples/`` demo. Builds a
deterministic pipeline over :mod:`interpret_live.backends.fake`, drives it on a
:class:`~interpret_live.clock.ManualClock` via the drain-then-advance protocol,
and returns the derived :class:`~interpret_live.metrics.MetricsReport` plus the
recorded audio so callers can show first-audio-out latency and audio-stage
stability (zero retraction) without any model, network, or real audio.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .audio_io import FakeAudioSink, FakeAudioSource
from .backends.fake import FakeMT, FakeSTT, FakeTTS
from .clock import ManualClock, drain_then_advance
from .config import PipelineConfig
from .metrics import MetricsReport
from .pipeline import Pipeline
from .types import AudioFrame, Hypothesis, Token

__all__ = [
    "FIXTURES",
    "BenchFixture",
    "BenchResult",
    "default_fixture",
    "get_fixture",
    "late_revision_fixture",
    "run_bench",
]


@dataclass(frozen=True, slots=True)
class BenchFixture:
    """A scripted benchmark fixture.

    Attributes:
        name: Human-readable fixture name.
        utterances: One or more utterances, each a list of partial hypotheses
            (the last carrying ``is_final=True``), including ASR revisions.
        translations: Exact segment-text → target-text mapping for the fake MT.
        frame_count: Number of source audio frames to drive STT.
    """

    name: str
    utterances: list[list[Hypothesis]]
    translations: dict[str, str]
    frame_count: int


@dataclass(frozen=True, slots=True)
class BenchResult:
    """Outcome of a benchmark run.

    Attributes:
        report: The derived metrics report.
        played_samples: Concatenated synthesized audio that reached the sink.
        played_segments: Segment indices played, in order (to show ordering).
    """

    report: MetricsReport
    played_samples: np.ndarray
    played_segments: tuple[int, ...]

    @property
    def retraction_count(self) -> int:
        """Audio retractions across the run (``0`` proves audio-stage stability)."""
        return self.report.total_retractions


def _tok(text: str, i: int, step_ms: int = 120) -> Token:
    return Token(text=text, start_ms=i * step_ms, end_ms=(i + 1) * step_ms)


def _hyp(words: list[str], *, is_final: bool = False) -> Hypothesis:
    return Hypothesis(
        tokens=tuple(_tok(w, i) for i, w in enumerate(words)),
        is_final=is_final,
    )


def default_fixture() -> BenchFixture:
    """A two-sentence English fixture with a mid-word ASR revision.

    The ASR first guesses ``"wether"`` then revises to ``"weather"`` on the next
    partial — because LocalAgreement only commits agreed tokens, the wrong guess
    never reaches MT/TTS, so the synthesized audio shows zero retraction.
    """
    utt1 = [
        _hyp(["the"]),
        _hyp(["the", "wether"]),
        _hyp(["the", "weather"]),
        _hyp(["the", "weather", "is"]),
        _hyp(["the", "weather", "is", "nice."]),
        _hyp(["the", "weather", "is", "nice."], is_final=True),
    ]
    utt2 = [
        _hyp(["let's"]),
        _hyp(["let's", "go."]),
        _hyp(["let's", "go."], is_final=True),
    ]
    translations = {
        "the weather is nice.": "el clima es agradable.",
        "let's go.": "vámonos.",
    }
    return BenchFixture(
        name="default-en-2sent",
        utterances=[utt1, utt2],
        translations=translations,
        frame_count=24,
    )


def late_revision_fixture() -> BenchFixture:
    """A one-sentence fixture that makes the LocalAgreement-*n* tradeoff visible.

    The ASR closes a segment on a wrong guess — ``"buck."`` — and only *then*
    revises it to ``"book."`` on the very next partial. The revision therefore
    lands **after** the token would commit at ``n=1`` but **before** it could
    commit at ``n=2`` (which needs the token unchanged across two consecutive
    partials):

    * At ``n=1`` the eager commit ships ``"I read a buck."`` to MT/TTS, and the
      later ``"book."`` contradicts an already-committed token, so
      ``post_commit_disagreement > 0`` — a signal to raise ``n``.
    * At ``n=2`` the wrong guess never commits, the correct ``"I read a book."``
      is what gets spoken, and ``post_commit_disagreement == 0``.

    Retractions stay ``0`` at every ``n``: the committed prefix is monotonic by
    construction (see :mod:`interpret_live.stabilize`), so a late disagreement
    only bumps the tuning counter — it never un-commits already-spoken audio.
    """
    utt = [
        _hyp(["I"]),
        _hyp(["I", "read"]),
        _hyp(["I", "read", "a"]),
        _hyp(["I", "read", "a", "buck."]),
        _hyp(["I", "read", "a", "book."]),
        _hyp(["I", "read", "a", "book."], is_final=True),
    ]
    translations = {
        "I read a book.": "leí un libro.",
        "I read a buck.": "leí un dólar.",  # the eager n=1 misread, translated literally
    }
    return BenchFixture(
        name="late-revision-en",
        utterances=[utt],
        translations=translations,
        frame_count=12,
    )


#: Named, built-in benchmark fixtures. ``get_fixture`` builds a fresh instance
#: per call (each factory returns immutable-by-intent but list-backed data, so a
#: registry of factories keeps callers isolated from one another).
FIXTURES: dict[str, Callable[[], BenchFixture]] = {
    "default-en-2sent": default_fixture,
    "late-revision-en": late_revision_fixture,
}


def get_fixture(name: str) -> BenchFixture:
    """Return the built-in fixture registered under ``name``.

    Raises:
        ValueError: If ``name`` is not a known fixture; the message lists the
            available names so a caller (e.g. the CLI) can surface them.
    """
    try:
        factory = FIXTURES[name]
    except KeyError:
        available = ", ".join(sorted(FIXTURES))
        raise ValueError(f"unknown fixture {name!r}; available: {available}") from None
    return factory()


def _source_frames(clock: ManualClock, count: int) -> FakeAudioSource:
    frames = [
        AudioFrame(
            samples=np.full(320, 0.05, dtype=np.float32),
            sample_rate=16000,
            t_ms=i * 20,
        )
        for i in range(count)
    ]
    return FakeAudioSource(frames, clock=clock, frame_delay_ms=20)


async def run_bench(
    fixture: BenchFixture | None = None,
    *,
    config: PipelineConfig | None = None,
) -> BenchResult:
    """Run ``fixture`` (or the default) through fakes and return a result.

    Deterministic: a :class:`ManualClock` plus drain-then-advance means no real
    sleeps and a reproducible metrics report.
    """
    fixture = fixture or default_fixture()
    cfg = config or PipelineConfig()
    clock = ManualClock()

    stt = FakeSTT(fixture.utterances, clock=clock, partial_delay_ms=40)
    mt = FakeMT(fixture.translations, clock=clock, latency_ms=30)
    tts = FakeTTS(clock=clock, chunks=2, chunk_latency_ms=20)
    sink = FakeAudioSink(clock=clock)
    source = _source_frames(clock, fixture.frame_count)

    pipeline = Pipeline(stt=stt, mt=mt, tts=tts, sink=sink, clock=clock, config=cfg)

    import asyncio

    task = asyncio.ensure_future(pipeline.run(source.frames()))
    await drain_then_advance(clock)
    await task

    return BenchResult(
        report=pipeline.metrics.report(),
        played_samples=sink.concatenated(),
        played_segments=tuple(c.segment_index for c in sink.played),
    )
