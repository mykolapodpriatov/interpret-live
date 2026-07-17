"""Runtime composition: build and run real live sessions behind the CLI.

Provider-specific factories construct validated, independently owned
resources in a fixed order (plan Task 8):

1. validate configuration (languages, models, voices, devices),
2. resolve/download models (spawned preflight; ``--offline`` honored),
3. start and health-check model workers / provider clients,
4. open audio devices,
5. enter the :class:`~interpret_live.session.Session`.

Everything lives under an ``AsyncExitStack`` so cancellation or Ctrl-C closes
devices, adapter ``aclose()`` methods, child workers, and clients exactly once
in reverse ownership order, within bounded budgets — leaving no background
task, audio thread, or child process. For ``--dual``, two fully independent
directional backends are built (stateful STT/MT/TTS/provider objects are
never shared between directions).
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .clock import Clock, RealClock
from .config import AudioConfig, PipelineConfig
from .metrics import MetricsReport
from .session import DualChannel, PipelineBackend, S2SBackend, Session
from .types import AudioSink, AudioSource

__all__ = ["RuntimeConfigError", "RuntimeDeps", "RuntimeOptions", "run_session"]


class RuntimeConfigError(ValueError):
    """Incomplete/inconsistent runtime configuration (fails before devices)."""


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Validated options assembled by the CLI for one live run.

    Attributes mirror the CLI surface; credentials never appear here — the
    OpenAI key is read by the SDK from the environment only.
    """

    backend: str = "offline"
    provider: str = "openai"
    source_lang: str = "en"
    target_lang: str = "es"
    whisper_model: str = "small"
    nllb_model: str | None = None
    piper_voice: str | None = None  # target-language manifest default if None
    piper_voice_source: str | None = None  # dual B->A voice (source language)
    openai_model: str | None = None
    openai_voice: str = "marin"
    openai_voice_source: str | None = None  # dual B->A voice (defaults to openai_voice)
    cache_dir: str | None = None
    offline: bool = False
    dual: bool = False
    input_device: int | None = None
    output_device: int | None = None
    input_device_b: int | None = None
    output_device_b: int | None = None
    enable_barge_in: bool = True
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


@dataclass(slots=True)
class RuntimeDeps:
    """Injectable constructors (tests swap in doubles; defaults are real)."""

    prefetch: Callable[[RuntimeOptions, str, str], Awaitable[dict[str, Any]]]
    make_stt: Callable[[RuntimeOptions, dict[str, Any], str, str], Any]
    make_mt: Callable[[RuntimeOptions, dict[str, Any], str, str], Any]
    make_tts: Callable[[RuntimeOptions, dict[str, Any], str, str], Any]
    make_s2s: Callable[[RuntimeOptions, str, str, str], Any]
    make_source: Callable[[RuntimeOptions, int | None, Clock], AudioSource]
    make_sink: Callable[[RuntimeOptions, int | None, Clock], AudioSink]
    validate_devices: Callable[[RuntimeOptions], None]
    check_extras: Callable[[RuntimeOptions], None] = lambda opts: None
    clock_factory: Callable[[], Clock] = RealClock


# ----- default (real) dependency constructors -----------------------------------


def _voice_for_language(opts: RuntimeOptions, language: str) -> str:
    """The Piper voice speaking ``language`` (explicit choice or manifest default)."""
    from .models import load_piper_manifest

    if language == opts.target_lang and opts.piper_voice is not None:
        return opts.piper_voice
    if language == opts.source_lang and opts.piper_voice_source is not None:
        return opts.piper_voice_source
    manifest = load_piper_manifest()
    voice = manifest["defaults"].get(language)
    if voice is None:
        raise RuntimeConfigError(
            f"no default Piper voice for language {language!r}; pass --voice/--voice-b "
            f"(manifest voices: {', '.join(sorted(manifest['voices']))})"
        )
    return str(voice)


async def _default_prefetch(opts: RuntimeOptions, source: str, target: str) -> dict[str, Any]:
    from .models import NLLB_REPO, PrefetchSpec, prefetch_in_worker

    # Each direction speaks its own target language: dual mode needs a voice
    # per direction, resolved (and downloaded) up front.
    voice_languages = [target] + ([source] if opts.dual else [])
    resolved: dict[str, Any] = {}
    for index, language in enumerate(voice_languages):
        voice = _voice_for_language(opts, language)
        spec = PrefetchSpec(
            whisper_model=opts.whisper_model if index == 0 else None,
            nllb_model=(opts.nllb_model or NLLB_REPO) if index == 0 else None,
            piper_voice=voice,
        )
        resolved.update(
            await prefetch_in_worker(spec, cache_dir=opts.cache_dir, offline=opts.offline)
        )
        resolved[f"_piper_voice_for_{language}"] = voice
    return resolved


def _default_make_stt(
    opts: RuntimeOptions, resolved: dict[str, Any], source: str, target: str
) -> Any:
    from .backends.whisper import WhisperSTT

    artifact = resolved.get("whisper")
    model_source = artifact.path if artifact is not None else opts.whisper_model
    return WhisperSTT(
        model_size=model_source,
        language=source,
        vad_threshold=opts.pipeline.vad_threshold,
        vad_hangover_ms=opts.pipeline.vad_hangover_ms,
    )


def _default_make_mt(
    opts: RuntimeOptions, resolved: dict[str, Any], source: str, target: str
) -> Any:
    from .backends.nllb import NllbMT
    from .models import NLLB_REPO

    artifact = resolved.get("nllb")
    model_name = artifact.path if artifact is not None else (opts.nllb_model or NLLB_REPO)
    return NllbMT(source_lang=source, target_lang=target, model_name=model_name)


def _default_make_tts(
    opts: RuntimeOptions, resolved: dict[str, Any], source: str, target: str
) -> Any:
    from .backends.piper import PiperTTS

    voice_id = resolved.get(f"_piper_voice_for_{target}")
    model = resolved.get(f"piper:{voice_id}:model")
    config = resolved.get(f"piper:{voice_id}:config")
    if model is None or config is None:
        raise RuntimeConfigError(f"piper voice artifacts missing for {voice_id!r}")
    return PiperTTS(model_path=model.path, config_path=config.path)


def _default_make_s2s(opts: RuntimeOptions, source: str, target: str, voice: str) -> Any:
    from .backends.realtime import DEFAULT_REALTIME_MODEL, RealtimeS2S

    return RealtimeS2S(
        model=opts.openai_model or DEFAULT_REALTIME_MODEL,
        voice=voice,
        source_lang=source,
        target_lang=target,
    )


def _default_make_source(opts: RuntimeOptions, device: int | None, clock: Clock) -> AudioSource:
    from .audio_io import MicSource

    return MicSource(
        sample_rate=opts.audio.capture_rate,
        frame_ms=opts.audio.frame_ms,
        device=device,
        clock=clock,
        queue_frames=opts.audio.mic_queue_frames,
    )


def _default_make_sink(opts: RuntimeOptions, device: int | None, clock: Clock) -> AudioSink:
    from .audio_io import SpeakerSink

    return SpeakerSink(
        device=device,
        device_rate=opts.audio.playback_rate,
        clock=clock,
        capacity=opts.audio.playback_capacity,
        ring_ms=opts.audio.playback_ring_ms,
    )


def _default_validate_devices(opts: RuntimeOptions) -> None:
    from .audio_io import validate_input_device, validate_output_device

    for device in _input_devices(opts):
        validate_input_device(device, opts.audio.capture_rate)
    for device in _output_devices(opts):
        if opts.audio.playback_rate is not None:
            validate_output_device(device, opts.audio.playback_rate)


def _input_devices(opts: RuntimeOptions) -> list[int | None]:
    return [opts.input_device, opts.input_device_b] if opts.dual else [opts.input_device]


def _output_devices(opts: RuntimeOptions) -> list[int | None]:
    return [opts.output_device, opts.output_device_b] if opts.dual else [opts.output_device]


def _default_check_extras(opts: RuntimeOptions) -> None:
    """Fail fast with a clear install hint before any network/model work."""
    from .backends.guard import require

    if opts.backend == "offline":
        require("faster_whisper", backend="whisper", extra="whisper")
        require("transformers", backend="mt", extra="mt")
        require("piper", backend="piper", extra="piper")
    else:
        require("openai", backend="realtime", extra="openai")
    require("sounddevice", backend="audio", extra="audio")
    require("soxr", backend="audio", extra="audio")


def default_deps() -> RuntimeDeps:
    """The real (import-guarded) constructor set."""
    return RuntimeDeps(
        prefetch=_default_prefetch,
        make_stt=_default_make_stt,
        make_mt=_default_make_mt,
        make_tts=_default_make_tts,
        make_s2s=_default_make_s2s,
        make_source=_default_make_source,
        make_sink=_default_make_sink,
        validate_devices=_default_validate_devices,
        check_extras=_default_check_extras,
    )


# ----- validation -----------------------------------------------------------------


def _validate_options(opts: RuntimeOptions) -> None:
    if opts.backend not in ("offline", "cloud"):
        raise RuntimeConfigError(f"unknown backend {opts.backend!r} (use 'offline' or 'cloud')")
    if opts.backend == "cloud" and opts.provider != "openai":
        raise RuntimeConfigError(f"unsupported cloud provider {opts.provider!r} (only 'openai')")
    if opts.backend == "cloud" and opts.offline:
        raise RuntimeConfigError(
            "--offline cannot be combined with the cloud backend: the flag governs "
            "local model resolution and the cloud path requires a network"
        )
    if not opts.source_lang or not opts.target_lang:
        raise RuntimeConfigError("source and target languages must be non-empty")
    if opts.source_lang == opts.target_lang:
        raise RuntimeConfigError("source and target languages must differ")
    if opts.dual:
        provided = (
            opts.input_device,
            opts.output_device,
            opts.input_device_b,
            opts.output_device_b,
        )
        if any(device is None for device in provided):
            raise RuntimeConfigError(
                "--dual requires explicit A/B device selections: "
                "--input-device, --output-device, --input-device-b, --output-device-b"
            )


def duplicate_device_warnings(opts: RuntimeOptions) -> list[str]:
    """Warnings for physical devices explicitly reused across directions."""
    warnings: list[str] = []
    if not opts.dual:
        return warnings
    if opts.input_device == opts.input_device_b:
        warnings.append(
            f"input device {opts.input_device} is used for BOTH directions; "
            "each speaker should have their own microphone (cross-talk risk)"
        )
    if opts.output_device == opts.output_device_b:
        warnings.append(
            f"output device {opts.output_device} is used for BOTH directions; "
            "translated audio will mix into one output (feedback risk)"
        )
    return warnings


# ----- lifecycle helpers ------------------------------------------------------------


async def _start_adapter(stack: contextlib.AsyncExitStack, adapter: Any, budget_ms: int) -> Any:
    """Start (health-check) an adapter and register its bounded close."""
    start = getattr(adapter, "start", None)
    if callable(start):
        await start()

    aclose = getattr(adapter, "aclose", None)
    if callable(aclose):

        async def _close() -> None:
            # The adapters' own aclose() budgets bound worker teardown; the
            # outer wait_for is a final backstop so shutdown can never hang.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(aclose(), budget_ms / 1000)

        stack.push_async_callback(_close)
    return adapter


# ----- composition ------------------------------------------------------------------


async def _build_offline_backend(
    opts: RuntimeOptions,
    deps: RuntimeDeps,
    stack: contextlib.AsyncExitStack,
    resolved: dict[str, Any],
    source: str,
    target: str,
) -> PipelineBackend:
    budget = opts.audio.shutdown_timeout_ms
    stt = await _start_adapter(stack, deps.make_stt(opts, resolved, source, target), budget)
    mt = await _start_adapter(stack, deps.make_mt(opts, resolved, source, target), budget)
    tts = await _start_adapter(stack, deps.make_tts(opts, resolved, source, target), budget)
    return PipelineBackend(stt=stt, mt=mt, tts=tts, name=f"offline-{source}-{target}")


async def _build_cloud_backend(
    opts: RuntimeOptions,
    deps: RuntimeDeps,
    stack: contextlib.AsyncExitStack,
    source: str,
    target: str,
    voice: str,
) -> S2SBackend:
    budget = opts.audio.shutdown_timeout_ms
    s2s = await _start_adapter(stack, deps.make_s2s(opts, source, target, voice), budget)
    return S2SBackend(s2s=s2s, name=f"cloud-{source}-{target}")


async def run_session(
    opts: RuntimeOptions,
    *,
    deps: RuntimeDeps | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> list[MetricsReport]:
    """Build and run one live session (or a dual pair); return its report(s).

    Construction order: validate -> resolve models -> start workers/clients ->
    open devices -> run the session. Ctrl-C (SIGINT) cancels the session task;
    the exit stack then closes everything exactly once, in reverse order.
    """
    deps = deps or default_deps()
    _validate_options(opts)
    deps.check_extras(opts)
    for warning in duplicate_device_warnings(opts):
        if on_warning is not None:
            on_warning(warning)

    async with contextlib.AsyncExitStack() as stack:
        # 1) Model preflight (offline backend only; cloud keeps no local models).
        resolved: dict[str, Any] = {}
        if opts.backend == "offline":
            resolved = await deps.prefetch(opts, opts.source_lang, opts.target_lang)

        # 2) Backends: model workers / provider clients start and health-check
        #    BEFORE audio devices open.
        directions = [(opts.source_lang, opts.target_lang)]
        if opts.dual:
            directions.append((opts.target_lang, opts.source_lang))
        backends: list[Any] = []
        for src, tgt in directions:
            if opts.backend == "offline":
                backends.append(await _build_offline_backend(opts, deps, stack, resolved, src, tgt))
            else:
                voice = (
                    opts.openai_voice
                    if tgt == opts.target_lang
                    else (opts.openai_voice_source or opts.openai_voice)
                )
                backends.append(await _build_cloud_backend(opts, deps, stack, src, tgt, voice))

        # 3) Devices open last; validation is cheap and already ran, but the
        #    explicit call keeps the fail-before-models contract testable.
        deps.validate_devices(opts)
        clock = deps.clock_factory()

        async def _close_sink(sink: AudioSink) -> None:
            aclose = getattr(sink, "aclose", None)
            if callable(aclose):
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(aclose(), opts.audio.shutdown_timeout_ms / 1000)

        if not opts.dual:
            source = deps.make_source(opts, opts.input_device, clock)
            sink = deps.make_sink(opts, opts.output_device, clock)
            stack.push_async_callback(_close_sink, sink)
            session = Session.create(
                backend=backends[0],
                source=source,
                sink=sink,
                clock=clock,
                config=opts.pipeline,
                enable_barge_in=opts.enable_barge_in,
            )
            await _run_cancellable(session.run())
            return [session.metrics()]

        a_source = deps.make_source(opts, opts.input_device, clock)
        b_source = deps.make_source(opts, opts.input_device_b, clock)
        a_sink = deps.make_sink(opts, opts.output_device, clock)
        b_sink = deps.make_sink(opts, opts.output_device_b, clock)
        stack.push_async_callback(_close_sink, a_sink)
        stack.push_async_callback(_close_sink, b_sink)
        dual = DualChannel.create(
            backend_a_to_b=backends[0],
            backend_b_to_a=backends[1],
            a_source=a_source,
            a_sink=a_sink,
            b_source=b_source,
            b_sink=b_sink,
            clock=clock,
            config=opts.pipeline,
            enable_barge_in=opts.enable_barge_in,
        )
        await _run_cancellable(dual.run())
        reports = dual.metrics()
        return [reports[0], reports[1]]
    raise AssertionError("unreachable")  # pragma: no cover - satisfies mypy


async def _run_cancellable(coro: Awaitable[None]) -> None:
    """Run the session as a task; SIGINT cancels it (a normal user stop).

    External cancellation of the runtime itself still propagates: the inner
    session task is cancelled and awaited first, so nothing keeps running
    behind the exit stack's cleanup.
    """
    task = asyncio.ensure_future(coro)
    loop = asyncio.get_running_loop()
    registered = False
    sigint_fired = False

    def _on_sigint() -> None:
        nonlocal sigint_fired
        sigint_fired = True
        task.cancel()

    with contextlib.suppress(NotImplementedError, RuntimeError):
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        registered = True
    try:
        await task
    except asyncio.CancelledError:
        # Awaiting a task delegates cancellation: an external cancel of the
        # runtime also lands here with the session task cancelled, so the
        # session is already down either way. Only a Ctrl-C we initiated
        # ourselves is a *normal* user stop; anything else propagates.
        if sigint_fired and task.cancelled():
            return
        if not task.done():  # pragma: no cover - defensive
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        raise
    finally:
        if registered:
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(signal.SIGINT)
