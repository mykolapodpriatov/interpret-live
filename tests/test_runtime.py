"""Runtime composition tests with injected factories/resources (no hardware).

Proves the CLI's `run` path builds and awaits a REAL session (audio reaches
the sink through the full pipeline), the construction order (validate ->
models -> workers -> devices -> session), dual independence, Ctrl-C-style
cancellation cleanliness, and bounded shutdown even with a stuck worker.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from helpers import frame, hyp
from interpret_live.audio_io import FakeAudioSink, FakeAudioSource
from interpret_live.backends.fake import FakeMT, FakeS2S, FakeS2STurn, FakeSTT, FakeTTS
from interpret_live.clock import Clock, ManualClock
from interpret_live.config import AudioConfig
from interpret_live.runtime import (
    RuntimeConfigError,
    RuntimeDeps,
    RuntimeOptions,
    run_session,
)


class _Recorder:
    """Tracks construction/lifecycle order across the injected factories."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.sinks: list[FakeAudioSink] = []
        self.started: list[str] = []
        self.closed: list[str] = []


class _LifecycleSTT(FakeSTT):
    def __init__(self, recorder: _Recorder, tag: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._recorder = recorder
        self._tag = tag

    async def start(self) -> None:
        self._recorder.started.append(self._tag)

    async def aclose(self) -> None:
        self._recorder.closed.append(self._tag)


def _deps(recorder: _Recorder, clock: ManualClock, *, frames: int = 8) -> RuntimeDeps:
    def make_stt(opts: RuntimeOptions, resolved: dict, src: str, tgt: str) -> Any:
        recorder.order.append(f"stt:{src}->{tgt}")
        script = [[hyp("hola."), hyp("hola.", is_final=True)]]
        return _LifecycleSTT(
            recorder, f"stt:{src}->{tgt}", script, clock=clock, partial_delay_ms=40
        )

    def make_mt(opts: RuntimeOptions, resolved: dict, src: str, tgt: str) -> Any:
        recorder.order.append(f"mt:{src}->{tgt}")
        return FakeMT({"hola.": f"[{tgt}] hola."}, clock=clock, latency_ms=20)

    def make_tts(opts: RuntimeOptions, resolved: dict, src: str, tgt: str) -> Any:
        recorder.order.append(f"tts:{src}->{tgt}")
        return FakeTTS(clock=clock, chunks=1, chunk_latency_ms=10)

    def make_s2s(opts: RuntimeOptions, src: str, tgt: str, voice: str) -> Any:
        recorder.order.append(f"s2s:{src}->{tgt}:{voice}")
        return FakeS2S(clock=clock, turns=[FakeS2STurn(chunks=2)], chunk_latency_ms=20)

    def make_source(opts: RuntimeOptions, device: int | None, _clock: Clock) -> Any:
        recorder.order.append(f"source:{device}")
        return FakeAudioSource(
            [frame(0.05, t_ms=i * 20, n=320) for i in range(frames)],
            clock=clock,
            frame_delay_ms=20,
        )

    def make_sink(opts: RuntimeOptions, device: int | None, _clock: Clock) -> Any:
        recorder.order.append(f"sink:{device}")
        sink = FakeAudioSink(clock=clock)
        recorder.sinks.append(sink)
        return sink

    async def prefetch(opts: RuntimeOptions, src: str, tgt: str) -> dict:
        recorder.order.append("prefetch")
        return {"_fake": True}

    def validate_devices(opts: RuntimeOptions) -> None:
        recorder.order.append("validate-devices")

    return RuntimeDeps(
        prefetch=prefetch,
        make_stt=make_stt,
        make_mt=make_mt,
        make_tts=make_tts,
        make_s2s=make_s2s,
        make_source=make_source,
        make_sink=make_sink,
        validate_devices=validate_devices,
        clock_factory=lambda: clock,
    )


async def _drive_to_completion(task: asyncio.Task, clock: ManualClock) -> None:
    for _ in range(2000):
        await asyncio.sleep(0)
        if task.done():
            return
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    raise AssertionError("runtime session never completed")


async def test_offline_run_builds_and_awaits_a_real_session() -> None:
    clock = ManualClock()
    recorder = _Recorder()
    opts = RuntimeOptions(backend="offline", source_lang="en", target_lang="es")
    task = asyncio.ensure_future(run_session(opts, deps=_deps(recorder, clock)))
    await _drive_to_completion(task, clock)
    reports = await task

    # Real audio flowed end-to-end into the sink (not a printed hint).
    assert recorder.sinks[0].played, "the session must actually run"
    assert len(reports) == 1
    assert reports[0].utterances
    # Construction order: models -> workers(start) -> devices -> session.
    assert recorder.order[0] == "prefetch"
    assert recorder.order.index("prefetch") < recorder.order.index("stt:en->es")
    assert recorder.order.index("validate-devices") < recorder.order.index("source:None")
    assert "stt:en->es" in recorder.started
    # Everything closed on exit (reverse ownership order includes the adapter).
    assert "stt:en->es" in recorder.closed


async def test_cloud_run_uses_s2s_backend_with_configured_voice() -> None:
    clock = ManualClock()
    recorder = _Recorder()
    opts = RuntimeOptions(backend="cloud", openai_voice="cedar")
    task = asyncio.ensure_future(run_session(opts, deps=_deps(recorder, clock)))
    await _drive_to_completion(task, clock)
    reports = await task
    assert "s2s:en->es:cedar" in recorder.order
    assert "prefetch" not in recorder.order, "cloud keeps no local models"
    assert recorder.sinks[0].played
    assert len(reports) == 1


async def test_dual_builds_two_independent_reversed_backends() -> None:
    clock = ManualClock()
    recorder = _Recorder()
    opts = RuntimeOptions(
        backend="offline",
        dual=True,
        input_device=1,
        output_device=2,
        input_device_b=3,
        output_device_b=4,
    )
    task = asyncio.ensure_future(run_session(opts, deps=_deps(recorder, clock)))
    await _drive_to_completion(task, clock)
    reports = await task

    # Two distinct directional stacks with reversed language directions.
    assert "stt:en->es" in recorder.order and "stt:es->en" in recorder.order
    assert "mt:en->es" in recorder.order and "mt:es->en" in recorder.order
    # Correct sink routing: A(1) mic -> B(4) speaker and B(3) mic -> A(2).
    assert "source:1" in recorder.order and "source:3" in recorder.order
    assert "sink:2" in recorder.order and "sink:4" in recorder.order
    assert len(reports) == 2
    # Both directions produced audio in their own sinks.
    assert all(sink.played for sink in recorder.sinks)


async def test_dual_same_device_reuse_warns() -> None:
    clock = ManualClock()
    recorder = _Recorder()
    warnings: list[str] = []
    opts = RuntimeOptions(
        backend="offline",
        dual=True,
        input_device=1,
        output_device=2,
        input_device_b=1,  # same mic reused across directions
        output_device_b=2,  # same speaker reused across directions
    )
    task = asyncio.ensure_future(
        run_session(opts, deps=_deps(recorder, clock), on_warning=warnings.append)
    )
    await _drive_to_completion(task, clock)
    await task
    assert any("input device" in w for w in warnings)
    assert any("output device" in w for w in warnings)


async def test_cancellation_closes_everything_exactly_once() -> None:
    clock = ManualClock()
    recorder = _Recorder()
    # A long source so the session is definitely mid-flight when cancelled.
    opts = RuntimeOptions(backend="offline")
    task = asyncio.ensure_future(run_session(opts, deps=_deps(recorder, clock, frames=100_000)))
    for _ in range(200):
        await asyncio.sleep(0)
        if recorder.sinks and recorder.sinks[0].played:
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The exit stack still closed the adapters exactly once.
    assert recorder.closed.count("stt:en->es") == 1
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == []


async def test_stuck_adapter_close_is_bounded_by_shutdown_budget() -> None:
    clock = ManualClock()
    recorder = _Recorder()

    class _StuckSTT(_LifecycleSTT):
        async def aclose(self) -> None:  # ignores cooperative shutdown forever
            await asyncio.sleep(3600)

    deps = _deps(recorder, clock, frames=4)
    original_make_stt = deps.make_stt

    def make_stuck_stt(opts: RuntimeOptions, resolved: dict, src: str, tgt: str) -> Any:
        adapter = original_make_stt(opts, resolved, src, tgt)
        adapter.__class__ = _StuckSTT
        return adapter

    deps.make_stt = make_stuck_stt
    opts = RuntimeOptions(backend="offline", audio=AudioConfig(shutdown_timeout_ms=200))
    started = time.monotonic()
    task = asyncio.ensure_future(run_session(opts, deps=deps))
    # The stuck close is bounded by a REAL-time budget (wait_for), so this
    # driver mixes small real sleeps with manual-clock advances.
    for _ in range(600):
        await asyncio.sleep(0.01)
        if task.done():
            break
        nxt = clock.next_wakeup_ms()
        if nxt is not None:
            clock.advance(nxt)
    assert task.done(), "shutdown must not hang on a stuck adapter close"
    await task
    # The wait_for backstop bounded the stuck close (real-time budget 200ms).
    assert time.monotonic() - started < 5.0


async def test_validation_failures_fire_before_any_construction() -> None:
    clock = ManualClock()
    recorder = _Recorder()
    deps = _deps(recorder, clock)
    with pytest.raises(RuntimeConfigError, match="unknown backend"):
        await run_session(RuntimeOptions(backend="bogus"), deps=deps)
    with pytest.raises(RuntimeConfigError, match="must differ"):
        await run_session(RuntimeOptions(source_lang="en", target_lang="en"), deps=deps)
    with pytest.raises(RuntimeConfigError, match="--input-device"):
        await run_session(RuntimeOptions(dual=True), deps=deps)
    with pytest.raises(RuntimeConfigError, match="cloud"):
        await run_session(RuntimeOptions(backend="cloud", offline=True), deps=deps)
    assert recorder.order == [], "nothing may be constructed before validation"


# ----- default (real) dependency constructors --------------------------------


def test_default_deps_wiring_and_voice_selection(tmp_path: Any) -> None:
    import pytest as _pytest

    from interpret_live.backends.guard import MissingExtraError
    from interpret_live.models import ResolvedArtifact
    from interpret_live.runtime import (
        _default_check_extras,
        _default_make_mt,
        _default_make_s2s,
        _default_make_stt,
        _default_make_tts,
        _voice_for_language,
        default_deps,
        duplicate_device_warnings,
    )

    deps = default_deps()
    assert deps.prefetch is not None and deps.check_extras is not None

    opts = RuntimeOptions(piper_voice="explicit-target", piper_voice_source="explicit-source")
    assert _voice_for_language(opts, "es") == "explicit-target"
    assert _voice_for_language(opts, "en") == "explicit-source"
    defaults = RuntimeOptions()
    assert _voice_for_language(defaults, "de") == "de_DE-thorsten-medium"
    with _pytest.raises(RuntimeConfigError, match="no default Piper voice"):
        _voice_for_language(defaults, "xx")

    # The real constructors fail fast with clear install hints in a no-extras
    # environment — after exercising their artifact-resolution logic.
    artifact = ResolvedArtifact(
        name="whisper:small",
        path=str(tmp_path),
        requested_revision="r",
        resolved_revision="r",
        provenance="local",
    )
    with _pytest.raises(MissingExtraError, match="whisper"):
        _default_make_stt(defaults, {"whisper": artifact}, "en", "es")
    with _pytest.raises(MissingExtraError, match=r"\[mt\]"):
        _default_make_mt(defaults, {"nllb": artifact}, "en", "es")
    with _pytest.raises(MissingExtraError, match="openai"):
        _default_make_s2s(defaults, "en", "es", "marin")
    with _pytest.raises(RuntimeConfigError, match="artifacts missing"):
        _default_make_tts(defaults, {}, "en", "es")
    with _pytest.raises(MissingExtraError, match="whisper"):
        _default_check_extras(RuntimeOptions(backend="offline"))
    with _pytest.raises(MissingExtraError, match="openai"):
        _default_check_extras(RuntimeOptions(backend="cloud"))
    assert duplicate_device_warnings(RuntimeOptions()) == []


def test_default_source_sink_and_device_validation_with_fake_sd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from interpret_live.runtime import (
        _default_make_sink,
        _default_make_source,
        _default_validate_devices,
    )
    from test_audio_io_real import make_fake_sd

    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd())
    clock = ManualClock()
    opts = RuntimeOptions(audio=AudioConfig(playback_rate=48000))
    source = _default_make_source(opts, None, clock)
    sink = _default_make_sink(opts, None, clock)
    assert source is not None and sink is not None
    _default_validate_devices(opts)  # both directions validated without error


async def test_default_prefetch_offline_reports_missing(tmp_path: Any) -> None:
    from interpret_live.models import OfflineArtifactsMissingError
    from interpret_live.runtime import _default_prefetch

    opts = RuntimeOptions(cache_dir=str(tmp_path), offline=True)
    with pytest.raises(OfflineArtifactsMissingError):
        await _default_prefetch(opts, "en", "es")
