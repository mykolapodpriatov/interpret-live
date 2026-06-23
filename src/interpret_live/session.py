"""Session DISPATCHER, capability negotiation, and dual-channel mode.

``Session`` is a *dispatcher*, not a shared state machine: :func:`Session.create`
inspects the chosen :class:`Backend` and returns either a :class:`PipelineSession`
(STT→stabilize→segment→MT→TTS — the LocalAgreement stabilizer is **active**) or
an :class:`S2SSession` (audio→S2S→audio — the stabilizer is honestly
**bypassed**). Both satisfy the small :class:`SessionProto`
(``run`` / ``interrupt`` / ``metrics``), so the CLI and :class:`DualChannel` are
backend-agnostic with no dead code paths.

**Capability negotiation (fail early, clearly):** :func:`Session.create`
validates the backend supports the requested features (e.g. a pipeline-only
feature on an S2S backend, or ``--dual`` on a backend that can't run two
streams) and raises :class:`CapabilityError` at startup — never silently
choosing the wrong path or failing mid-call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .audio_io import tee
from .backends import MT, S2S, STT, TTS
from .clock import Clock
from .config import PipelineConfig
from .metrics import MetricsLog, MetricsReport
from .pipeline import Pipeline
from .s2s import S2SPipeline
from .types import AudioFrame, AudioSink, AudioSource
from .vad import BargeInDetector, EnergyVAD

__all__ = [
    "Backend",
    "Capabilities",
    "CapabilityError",
    "DualChannel",
    "PipelineBackend",
    "PipelineSession",
    "S2SBackend",
    "S2SSession",
    "Session",
    "SessionProto",
]


class CapabilityError(ValueError):
    """Raised at startup when a backend cannot satisfy the requested features."""


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a backend declares it can do (surfaced for fail-early negotiation).

    Attributes:
        interrupt: Supports barge-in / ``interrupt()``.
        metrics: Produces latency / barge-in-stop metrics.
        dual: Can run two concurrent streams (dual-channel).
        stabilizer: Runs the LocalAgreement audio-stage stabilizer (pipeline
            path only — ``False`` on the cloud S2S path, stated honestly).
    """

    interrupt: bool
    metrics: bool
    dual: bool
    stabilizer: bool


@runtime_checkable
class SessionProto(Protocol):
    """The backend-agnostic session surface used by the CLI and DualChannel."""

    async def run(self) -> None:
        """Run this direction end-to-end until the source is exhausted."""
        ...

    def interrupt(self) -> None:
        """Request a barge-in interrupt (no-op if unsupported)."""
        ...

    def metrics(self) -> MetricsReport:
        """Return the derived metrics report for this session."""
        ...


@runtime_checkable
class Backend(Protocol):
    """A backend factory that knows its own capabilities."""

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    def capabilities(self) -> Capabilities:
        """The capabilities this backend supports."""
        ...


@dataclass(slots=True)
class PipelineBackend:
    """A pipeline-path backend: explicit STT + MT + TTS components.

    Attributes:
        stt: Streaming STT.
        mt: Machine translation.
        tts: Streaming TTS.
        name: Backend name (default ``"pipeline"``).
    """

    stt: STT
    mt: MT
    tts: TTS
    name: str = "pipeline"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(interrupt=True, metrics=True, dual=True, stabilizer=True)


@dataclass(slots=True)
class S2SBackend:
    """A unified S2S backend (cloud realtime).

    Attributes:
        s2s: The S2S provider.
        name: Backend name (default ``"s2s"``).
    """

    s2s: S2S
    name: str = "s2s"

    @property
    def capabilities(self) -> Capabilities:
        # The stabilizer is bypassed on the cloud path (provider does S2S).
        return Capabilities(interrupt=True, metrics=True, dual=True, stabilizer=False)


class PipelineSession:
    """A :class:`SessionProto` over the pipeline path (stabilizer ACTIVE)."""

    __slots__ = ("_barge_in", "_clock", "_config", "_pipeline", "_sink", "_source")

    def __init__(
        self,
        *,
        backend: PipelineBackend,
        source: AudioSource,
        sink: AudioSink,
        clock: Clock,
        config: PipelineConfig,
        enable_barge_in: bool,
        metrics: MetricsLog | None = None,
    ) -> None:
        self._source = source
        self._sink = sink
        self._clock = clock
        self._config = config
        self._barge_in: BargeInDetector | None = None
        if enable_barge_in:
            self._barge_in = BargeInDetector(
                EnergyVAD(
                    threshold=config.vad_threshold,
                    hangover_ms=config.vad_hangover_ms,
                ),
                onset_ms=config.barge_in_onset_ms,
                clock=clock,
            )
        self._pipeline = Pipeline(
            stt=backend.stt,
            mt=backend.mt,
            tts=backend.tts,
            sink=sink,
            clock=clock,
            config=config,
            barge_in=self._barge_in,
            metrics=metrics,
        )

    async def run(self) -> None:
        """Run the pipeline, fanning the source to STT (+ barge-in if enabled)."""
        if self._barge_in is None:
            await self._pipeline.run(self._source.frames())
            return
        broadcaster, (stt_sub, barge_sub) = tee(self._source, 2, maxsize=self._config.queue_maxsize)
        import asyncio

        async with asyncio.TaskGroup() as tg:
            tg.create_task(broadcaster.run(), name="mic-broadcaster")
            tg.create_task(
                self._pipeline.run_with_barge_in(stt_sub, barge_sub),
                name="pipeline",
            )

    def interrupt(self) -> None:
        """Manually fire a barge-in interrupt (used by tests/CLI)."""
        self._pipeline._interrupt.fire()

    def metrics(self) -> MetricsReport:
        return self._pipeline.metrics.report()


class S2SSession:
    """A :class:`SessionProto` over the unified S2S path (stabilizer BYPASSED)."""

    __slots__ = ("_barge_in", "_clock", "_config", "_pipeline", "_sink", "_source")

    def __init__(
        self,
        *,
        backend: S2SBackend,
        source: AudioSource,
        sink: AudioSink,
        clock: Clock,
        config: PipelineConfig,
        enable_barge_in: bool,
        metrics: MetricsLog | None = None,
    ) -> None:
        self._source = source
        self._sink = sink
        self._clock = clock
        self._config = config
        self._barge_in: BargeInDetector | None = None
        if enable_barge_in:
            self._barge_in = BargeInDetector(
                EnergyVAD(
                    threshold=config.vad_threshold,
                    hangover_ms=config.vad_hangover_ms,
                ),
                onset_ms=config.barge_in_onset_ms,
                clock=clock,
            )
        self._pipeline = S2SPipeline(
            s2s=backend.s2s,
            sink=sink,
            clock=clock,
            config=config,
            barge_in=self._barge_in,
            metrics=metrics,
        )

    async def run(self) -> None:
        """Run the S2S pipeline, fanning the source if barge-in is enabled."""
        if self._barge_in is None:
            await self._pipeline.run(self._source.frames())
            return
        broadcaster, (s2s_sub, barge_sub) = tee(self._source, 2, maxsize=self._config.queue_maxsize)
        import asyncio

        async with asyncio.TaskGroup() as tg:
            tg.create_task(broadcaster.run(), name="mic-broadcaster")
            tg.create_task(
                self._pipeline.run_with_barge_in(s2s_sub, barge_sub),
                name="s2s-pipeline",
            )

    def interrupt(self) -> None:
        """Manually fire a barge-in interrupt (used by tests/CLI)."""
        self._pipeline._interrupt.set()

    def metrics(self) -> MetricsReport:
        return self._pipeline.metrics.report()


@dataclass(frozen=True, slots=True)
class _Requested:
    """The features a caller asks :func:`Session.create` to satisfy."""

    barge_in: bool = False
    dual: bool = False
    require_stabilizer: bool = False


class Session:
    """Dispatcher: build the right session for a backend, validating features."""

    @staticmethod
    def create(
        *,
        backend: Backend,
        source: AudioSource,
        sink: AudioSink,
        clock: Clock,
        config: PipelineConfig | None = None,
        enable_barge_in: bool = False,
        require_stabilizer: bool = False,
        for_dual: bool = False,
        metrics: MetricsLog | None = None,
    ) -> SessionProto:
        """Return a :class:`PipelineSession` or :class:`S2SSession`.

        Raises:
            CapabilityError: If the backend cannot satisfy a requested feature
                (barge-in, dual-channel, or an explicit stabilizer requirement).
        """
        cfg = config or PipelineConfig()
        Session._negotiate(
            backend,
            _Requested(
                barge_in=enable_barge_in,
                dual=for_dual,
                require_stabilizer=require_stabilizer,
            ),
        )
        if isinstance(backend, PipelineBackend):
            return PipelineSession(
                backend=backend,
                source=source,
                sink=sink,
                clock=clock,
                config=cfg,
                enable_barge_in=enable_barge_in,
                metrics=metrics,
            )
        if isinstance(backend, S2SBackend):
            return S2SSession(
                backend=backend,
                source=source,
                sink=sink,
                clock=clock,
                config=cfg,
                enable_barge_in=enable_barge_in,
                metrics=metrics,
            )
        raise CapabilityError(  # pragma: no cover - guards future backends
            f"unknown backend type: {type(backend).__name__}"
        )

    @staticmethod
    def _negotiate(backend: Backend, requested: _Requested) -> None:
        caps = backend.capabilities
        if requested.barge_in and not caps.interrupt:
            raise CapabilityError(f"backend {backend.name!r} does not support barge-in/interrupt")
        if requested.dual and not caps.dual:
            raise CapabilityError(f"backend {backend.name!r} cannot run dual-channel (two streams)")
        if requested.require_stabilizer and not caps.stabilizer:
            raise CapabilityError(
                f"backend {backend.name!r} bypasses the LocalAgreement stabilizer "
                "(it is the cloud provider's responsibility on the S2S path); "
                "use the pipeline/offline backend if you require audio-stage "
                "stabilization"
            )


@dataclass(slots=True)
class DualChannel:
    """Two independent sessions: A→B and B→A, with separate sources and sinks.

    Takes TWO sources + TWO sinks (no hidden shared input): A→B uses A's mic +
    B's speaker; B→A the reverse. A local demo may pass the same device twice,
    but the constructor is explicit.

    Attributes:
        a_source: Speaker A's microphone.
        a_sink: Speaker A's output.
        b_source: Speaker B's microphone.
        b_sink: Speaker B's output.
    """

    a_to_b: SessionProto
    b_to_a: SessionProto
    _channels: tuple[SessionProto, SessionProto] = field(init=False)

    def __post_init__(self) -> None:
        self._channels = (self.a_to_b, self.b_to_a)

    @classmethod
    def create(
        cls,
        *,
        backend: Backend,
        a_source: AudioSource,
        a_sink: AudioSink,
        b_source: AudioSource,
        b_sink: AudioSink,
        clock: Clock,
        config: PipelineConfig | None = None,
        enable_barge_in: bool = False,
    ) -> DualChannel:
        """Build a dual-channel pair, validating dual capability up front."""
        a_to_b = Session.create(
            backend=backend,
            source=a_source,
            sink=b_sink,
            clock=clock,
            config=config,
            enable_barge_in=enable_barge_in,
            for_dual=True,
        )
        b_to_a = Session.create(
            backend=backend,
            source=b_source,
            sink=a_sink,
            clock=clock,
            config=config,
            enable_barge_in=enable_barge_in,
            for_dual=True,
        )
        return cls(a_to_b=a_to_b, b_to_a=b_to_a)

    async def run(self) -> None:
        """Run both directions concurrently to completion."""
        import asyncio

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.a_to_b.run(), name="dual-a-to-b")
            tg.create_task(self.b_to_a.run(), name="dual-b-to-a")

    def metrics(self) -> tuple[MetricsReport, MetricsReport]:
        """Per-direction metrics reports ``(a_to_b, b_to_a)``."""
        return (self.a_to_b.metrics(), self.b_to_a.metrics())


def _frames(source: AudioSource) -> AsyncIterator[AudioFrame]:
    """Tiny helper kept for symmetry/readability in session wiring."""
    return source.frames()
