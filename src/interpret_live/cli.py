"""``interpret-live`` command-line interface (Typer): ``run`` | ``devices`` | ``bench``.

* ``bench`` — replays a scripted fixture through deterministic fake backends and
  prints latency metrics + audio-stage stability (offline; the default demo).
* ``run`` — runs a live session (offline pipeline or cloud S2S); requires the
  corresponding optional extras and is guarded with clear install hints.
* ``devices`` — lists audio devices (requires the ``audio`` extra).
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .bench import default_fixture, run_bench
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
    agreement_n: int = typer.Option(2, "--agreement-n", min=1, help="LocalAgreement window."),
    max_segment_tokens: int = typer.Option(
        24, "--max-segment-tokens", min=1, help="Forced-flush segment cap."
    ),
) -> None:
    """Replay a fixture through fake backends and print latency + stability."""
    cfg = PipelineConfig(agreement_n=agreement_n, max_segment_tokens=max_segment_tokens)
    fixture = default_fixture()
    result = asyncio.run(run_bench(fixture, config=cfg))
    report = result.report

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
    if result.retraction_count != 0:
        raise typer.Exit(code=1)


@app.command()
def run(
    source: str = typer.Option("en", "--from", help="Source language code."),
    target: str = typer.Option("es", "--to", help="Target language code."),
    backend: str = typer.Option(
        "offline", "--backend", help="Backend: 'offline' (pipeline) or 'cloud' (S2S)."
    ),
    dual: bool = typer.Option(False, "--dual", help="Dual-channel meeting mode."),
    cache_dir: str | None = typer.Option(
        None, "--cache-dir", help="Model cache root (default: the platform cache dir)."
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never touch the network for local model resolution; fail if artifacts are missing.",
    ),
) -> None:
    """Run a live interpreting session (requires the relevant optional extras)."""
    if backend == "cloud" and offline:
        # --offline governs LOCAL model resolution; it is not a promise that a
        # cloud provider works without a network. Reject before any client or
        # device is constructed.
        console.print(
            "[red]--offline cannot be combined with --backend cloud: the cloud "
            "path requires network access by definition[/]"
        )
        raise typer.Exit(code=2)
    del cache_dir  # consumed by the runtime factory once Task 8 wires it
    console.print(
        f"[bold]interpret-live run[/] {source} -> {target} (backend={backend}, dual={dual})"
    )
    if backend == "offline":
        hint = (
            "The offline pipeline needs the whisper/mt/piper/audio extras:\n"
            "  pip install 'interpret-live[whisper,mt,piper,audio]'"
        )
    elif backend == "cloud":
        hint = (
            "The cloud S2S path needs a provider extra:\n"
            "  pip install 'interpret-live[openai]'  # or [gemini]"
        )
    else:
        console.print(f"[red]unknown backend: {backend!r} (use 'offline' or 'cloud')[/]")
        raise typer.Exit(code=2)
    console.print(
        "[yellow]Live audio I/O and heavy backends are optional extras.[/] "
        "This build models the audio pipeline and ships deterministic fakes; "
        "install the extras to run live:"
    )
    # Print the hint with markup disabled so the ``[extra]`` brackets are literal
    # (rich would otherwise parse them as style tags and drop them).
    console.print(hint, markup=False)
    console.print("Tip: run [bold]interpret-live bench[/] for an offline, no-extras demo.")


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
