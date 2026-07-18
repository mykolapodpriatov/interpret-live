"""``interpret-live`` command-line interface (Typer): ``run`` | ``devices`` | ``bench``.

* ``bench`` — replays a scripted fixture through deterministic fake backends and
  prints latency metrics + audio-stage stability (offline; the default demo).
* ``run`` — runs a live session (offline pipeline or cloud S2S); requires the
  corresponding optional extras and is guarded with clear install hints.
* ``devices`` — lists audio devices (requires the ``audio`` extra).
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .bench import FIXTURES, get_fixture, run_bench
from .config import PipelineConfig

app = typer.Typer(
    name="interpret-live",
    help="Real-time, audio-stage-stabilized simultaneous interpretation.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"interpret-live {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    _version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """interpret-live: streaming speech-to-speech translation."""


@app.command()
def bench(
    fixture_name: str = typer.Option(
        "default-en-2sent",
        "--fixture",
        help=f"Built-in fixture to replay (one of: {', '.join(sorted(FIXTURES))}).",
    ),
    agreement_n: int = typer.Option(2, "--agreement-n", min=1, help="LocalAgreement window."),
    max_segment_tokens: int = typer.Option(
        24, "--max-segment-tokens", min=1, help="Forced-flush segment cap."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit metrics as JSON to stdout (no table) for CI diffing/gating."
    ),
    list_fixtures: bool = typer.Option(
        False,
        "--list-fixtures",
        help="List built-in fixtures with a one-line description and exit.",
    ),
) -> None:
    """Replay a fixture through fake backends and print latency + stability."""
    if list_fixtures:
        # Enumerate the registry (each factory builds a fresh instance to read its
        # description) and exit 0 without running anything.
        for name in sorted(FIXTURES):
            console.print(f"{name}  {FIXTURES[name]().description}", highlight=False)
        raise typer.Exit(code=0)
    cfg = PipelineConfig(agreement_n=agreement_n, max_segment_tokens=max_segment_tokens)
    try:
        fixture = get_fixture(fixture_name)
    except ValueError as exc:
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(code=2) from exc
    result = asyncio.run(run_bench(fixture, config=cfg))
    report = result.report

    if as_json:
        # Deterministic (ManualClock + drain-then-advance), markup-free payload so
        # first-audio-out / commit-lag / retractions can be diffed across commits
        # or gated in CI. json.dumps + a stable key order keep it byte-identical
        # across identical runs; typer.echo bypasses Rich markup/soft-wrapping.
        payload: dict[str, object] = {
            "fixture": fixture.name,
            "config": {"agreement_n": agreement_n, "max_segment_tokens": max_segment_tokens},
            "played_segments": list(result.played_segments),
            **report.to_dict(),
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        table = Table(title=f"interpret-live bench — fixture '{fixture.name}'")
        table.add_column("utterance", style="cyan")
        table.add_column("first-audio-out (ms)", justify="right")
        table.add_column("commit-lag (ms)", justify="right")
        table.add_column("disagreements", justify="right")
        for u in report.utterances:
            table.add_row(
                u.utterance_id,
                _fmt(u.first_audio_out_ms),
                _fmt(u.commit_lag_ms),
                str(u.post_commit_disagreement),
            )
        console.print(table)
        console.print(
            f"audio-stage retractions: [bold green]{result.retraction_count}[/] "
            "(0 = synthesized speech never stuttered)"
        )
        console.print(f"played segments (in order): {list(result.played_segments)}")
        console.print(f"synthesized samples: {result.played_samples.size}")
    # Exit-code contract holds in both modes: a retraction (audio-stage
    # instability) is a hard failure that CI can gate on.
    if result.retraction_count != 0:
        raise typer.Exit(code=1)


@app.command()
def run(
    source: str = typer.Option("en", "--from", help="Source language code."),
    target: str = typer.Option("es", "--to", help="Target language code."),
    backend: str = typer.Option(
        "offline", "--backend", help="Backend: 'offline' (pipeline) or 'cloud' (S2S)."
    ),
    provider: str = typer.Option("openai", "--provider", help="Cloud provider (only 'openai')."),
    dual: bool = typer.Option(False, "--dual", help="Dual-channel meeting mode."),
    whisper_model: str = typer.Option("small", "--whisper-model", help="faster-whisper alias."),
    nllb_model: str | None = typer.Option(None, "--nllb-model", help="NLLB model id override."),
    voice: str | None = typer.Option(
        None, "--voice", help="Piper voice id/path speaking the TARGET language (offline)."
    ),
    voice_b: str | None = typer.Option(
        None, "--voice-b", help="Piper voice speaking the SOURCE language (dual B->A)."
    ),
    openai_voice: str | None = typer.Option(
        None, "--openai-voice", help="OpenAI Realtime output voice (cloud backend only)."
    ),
    openai_model: str | None = typer.Option(
        None, "--openai-model", help="OpenAI Realtime model id override."
    ),
    cache_dir: str | None = typer.Option(
        None, "--cache-dir", help="Model cache root (default: the platform cache dir)."
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never touch the network for local model resolution; fail if artifacts are missing.",
    ),
    input_device: int | None = typer.Option(None, "--input-device", help="Mic device index."),
    output_device: int | None = typer.Option(None, "--output-device", help="Speaker device index."),
    input_device_b: int | None = typer.Option(
        None, "--input-device-b", help="Speaker B's mic device index (dual)."
    ),
    output_device_b: int | None = typer.Option(
        None, "--output-device-b", help="Speaker B's output device index (dual)."
    ),
    barge_in: bool = typer.Option(
        True, "--barge-in/--no-barge-in", help="Interrupt playback when the speaker resumes."
    ),
) -> None:
    """Run a live interpreting session (requires the relevant optional extras)."""
    from .backends.guard import MissingExtraError
    from .runtime import RuntimeConfigError, RuntimeOptions, run_session

    # Provider-specific voice options must match the selected backend.
    if backend == "offline" and openai_voice is not None:
        console.print("[red]--openai-voice applies to the cloud backend only[/]")
        raise typer.Exit(code=2)
    if backend == "cloud" and (voice is not None or voice_b is not None):
        console.print("[red]--voice/--voice-b select Piper voices; use --openai-voice[/]")
        raise typer.Exit(code=2)

    opts = RuntimeOptions(
        backend=backend,
        provider=provider,
        source_lang=source,
        target_lang=target,
        whisper_model=whisper_model,
        nllb_model=nllb_model,
        piper_voice=voice,
        piper_voice_source=voice_b,
        openai_model=openai_model,
        openai_voice=openai_voice or "marin",
        cache_dir=cache_dir,
        offline=offline,
        dual=dual,
        input_device=input_device,
        output_device=output_device,
        input_device_b=input_device_b,
        output_device_b=output_device_b,
        enable_barge_in=barge_in,
    )
    console.print(
        f"[bold]interpret-live run[/] {source} -> {target} (backend={backend}, dual={dual})"
    )
    try:
        reports = asyncio.run(
            run_session(
                opts,
                on_warning=lambda w: console.print(f"[yellow]warning:[/] {w}", highlight=False),
            )
        )
    except RuntimeConfigError as exc:
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(code=2) from exc
    except MissingExtraError as exc:
        console.print(str(exc), markup=False, style="red")
        console.print("Tip: run [bold]interpret-live bench[/] for an offline, no-extras demo.")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("stopped.")
        return
    except Exception as exc:  # pragma: no cover - depends on live resources
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(code=1) from exc
    # Success only after a session actually ran (or the user stopped it).
    for index, report in enumerate(reports):
        _print_report(report, title=f"direction {index + 1}" if len(reports) > 1 else "session")


def _print_report(report: object, *, title: str) -> None:
    """Print the concise final metrics summary after a normal stop."""
    from .metrics import MetricsReport

    assert isinstance(report, MetricsReport)
    table = Table(title=f"metrics — {title}")
    table.add_column("utterance", style="cyan")
    table.add_column("first-audio-out (ms)", justify="right")
    table.add_column("barge-in-stop (ms)", justify="right")
    for u in report.utterances:
        table.add_row(u.utterance_id, _fmt(u.first_audio_out_ms), _fmt(u.barge_in_stop_ms))
    console.print(table)


@app.command()
def devices() -> None:
    """List available audio input/output devices (requires the 'audio' extra)."""
    try:
        from .audio_io import list_devices

        infos = list_devices()
    except ImportError as exc:
        # markup=False so the ``[audio]`` install hint is printed literally.
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title="audio devices")
    table.add_column("index", justify="right")
    table.add_column("name")
    table.add_column("in", justify="right")
    table.add_column("out", justify="right")
    table.add_column("default rate", justify="right")
    table.add_column("default", justify="left")
    for info in infos:
        roles = []
        if info.is_default_input:
            roles.append("input")
        if info.is_default_output:
            roles.append("output")
        table.add_row(
            str(info.index),
            info.name,
            str(info.max_input_channels),
            str(info.max_output_channels),
            f"{info.default_samplerate:g}",
            ", ".join(roles),
        )
    console.print(table)


models_app = typer.Typer(
    name="models",
    help="Manage local model artifacts (explicit prefetch, cache inspection).",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")


@models_app.command("download")
def models_download(
    source: str = typer.Option("en", "--from", help="Source language code."),
    target: str = typer.Option("es", "--to", help="Target language code."),
    backend: str = typer.Option(
        "offline", "--backend", help="Backend: only 'offline' has local artifacts."
    ),
    whisper_model: str = typer.Option("small", "--whisper-model", help="faster-whisper alias."),
    nllb_model: str | None = typer.Option(None, "--nllb-model", help="NLLB model id override."),
    voice: str | None = typer.Option(
        None, "--voice", help="Piper voice id from the manifest (default: per target language)."
    ),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Model cache root."),
    offline: bool = typer.Option(
        False, "--offline", help="Verify cache only; never touch the network."
    ),
) -> None:
    """Prefetch every artifact a run needs, with visible progress."""
    from .models import (
        NLLB_REPO,
        ModelManager,
        ModelResolutionError,
        OfflineArtifactsMissingError,
        PrefetchSpec,
        load_piper_manifest,
    )

    if backend == "cloud":
        console.print("The cloud backend keeps no local models; nothing to download.")
        return
    if backend != "offline":
        console.print(f"[red]unknown backend: {backend!r} (use 'offline' or 'cloud')[/]")
        raise typer.Exit(code=2)
    del source  # language pair only selects the voice today; STT/MT are ids
    manifest = load_piper_manifest()
    voice_id = voice or manifest["defaults"].get(target)
    if voice_id is None:
        console.print(
            f"[red]no default Piper voice for target language {target!r}; "
            f"pass --voice (available: {', '.join(sorted(manifest['voices']))})[/]"
        )
        raise typer.Exit(code=2)
    manager = ModelManager(
        cache_dir=cache_dir,
        offline=offline,
        progress=lambda line: console.print(line, markup=False, highlight=False),
    )
    spec = PrefetchSpec(
        whisper_model=whisper_model,
        nllb_model=nllb_model or NLLB_REPO,
        piper_voice=voice_id,
    )
    try:
        resolved = manager.resolve_all(spec)
    except OfflineArtifactsMissingError as exc:
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(code=1) from exc
    except ModelResolutionError as exc:
        console.print(str(exc), markup=False, style="red")
        raise typer.Exit(code=1) from exc
    table = Table(title=f"resolved model artifacts (cache: {manager.cache_dir})")
    table.add_column("artifact")
    table.add_column("revision")
    table.add_column("path")
    for artifact in resolved.values():
        table.add_row(
            artifact.name,
            (artifact.resolved_revision or "-")[:16],
            artifact.path,
        )
    console.print(table)
    entry = manifest["voices"].get(voice_id)
    if entry is not None:
        console.print(f"voice license/source: {entry['license_url']}", markup=False)


def _fmt(value: int | None) -> str:
    return "-" if value is None else str(value)


if __name__ == "__main__":  # pragma: no cover
    app()
