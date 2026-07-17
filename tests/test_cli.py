"""CLI tests: ``bench`` over fakes prints metrics; exit codes; ``run`` hints.

All offline — ``bench`` drives the deterministic fake pipeline; ``run`` prints
install hints (no live audio in CI); ``devices`` reports the missing-extra error.
"""

from __future__ import annotations

import importlib.util

import pytest
from typer.testing import CliRunner

from interpret_live import __version__
from interpret_live.cli import app

runner = CliRunner()

_no_whisper = pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is not None,
    reason="asserts the no-extras install hint; whisper extra is installed",
)
_no_openai = pytest.mark.skipif(
    importlib.util.find_spec("openai") is not None,
    reason="asserts the no-extras install hint; openai extra is installed",
)
_no_audio = pytest.mark.skipif(
    importlib.util.find_spec("sounddevice") is not None,
    reason="asserts the no-extras install hint; audio extra is installed",
)


def test_bench_runs_and_prints_metrics() -> None:
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "bench" in out
    assert "first-audio-out" in out
    assert "retractions: 0" in out  # audio-stage stability proven


def test_bench_accepts_tuning_flags() -> None:
    result = runner.invoke(app, ["bench", "--agreement-n", "3", "--max-segment-tokens", "10"])
    assert result.exit_code == 0, result.output
    assert "retractions: 0" in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@_no_whisper
def test_run_offline_without_extras_fails_with_install_hint() -> None:
    # A real composition attempt now: with no extras installed, the runtime's
    # fail-fast extras check surfaces the clear install hint (exit 1) BEFORE
    # any network/model/device work.
    result = runner.invoke(app, ["run", "--from", "en", "--to", "es", "--backend", "offline"])
    assert result.exit_code == 1
    assert "interpret-live[whisper]" in result.output


@_no_openai
def test_run_cloud_without_extras_fails_with_install_hint() -> None:
    result = runner.invoke(app, ["run", "--backend", "cloud"])
    assert result.exit_code == 1
    assert "interpret-live[openai]" in result.output


def test_run_rejects_openai_voice_on_offline_backend() -> None:
    result = runner.invoke(app, ["run", "--backend", "offline", "--openai-voice", "marin"])
    assert result.exit_code == 2
    assert "--openai-voice" in result.output


def test_run_rejects_piper_voice_on_cloud_backend() -> None:
    result = runner.invoke(app, ["run", "--backend", "cloud", "--voice", "es_ES-davefx-medium"])
    assert result.exit_code == 2
    assert "--voice" in result.output


def test_run_dual_requires_explicit_devices() -> None:
    result = runner.invoke(app, ["run", "--backend", "cloud", "--dual"])
    # Config validation fires before the extras check would matter.
    assert result.exit_code == 2
    assert "--input-device" in result.output


def test_run_unknown_backend_exits_nonzero() -> None:
    result = runner.invoke(app, ["run", "--backend", "bogus"])
    assert result.exit_code == 2
    assert "unknown backend" in result.output


@_no_audio
def test_devices_without_audio_extra_reports_clear_error() -> None:
    # In CI the 'audio' extra is not installed, so this surfaces the install hint
    # and exits non-zero (never a raw traceback).
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 1
    assert "interpret-live[audio]" in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help => usage shown, exit code 0 or 2 depending on typer version.
    assert "Usage" in result.output or "bench" in result.output


def test_run_rejects_cloud_plus_offline_before_anything() -> None:
    result = runner.invoke(app, ["run", "--backend", "cloud", "--offline"])
    assert result.exit_code == 2
    assert "--offline" in result.output and "cloud" in result.output


def test_models_download_cloud_backend_needs_nothing() -> None:
    result = runner.invoke(app, ["models", "download", "--backend", "cloud"])
    assert result.exit_code == 0
    assert "no local models" in result.output


def test_models_download_offline_reports_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(
        app,
        ["models", "download", "--offline", "--cache-dir", str(tmp_path), "--to", "es"],
    )
    assert result.exit_code == 1
    assert "missing from the cache" in result.output


def test_run_success_path_prints_metrics_summary(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import interpret_live.runtime as runtime_mod
    from interpret_live.metrics import MetricsLog
    from interpret_live.types import MetricEvent

    log = MetricsLog()
    log.append(MetricEvent(kind="utterance_start", t_ms=0, utterance_id="u1"))
    log.append(MetricEvent(kind="first_tts_out", t_ms=120, utterance_id="u1"))
    report = log.report()

    async def fake_run_session(opts, *, deps=None, on_warning=None):  # type: ignore[no-untyped-def]
        if on_warning is not None:
            on_warning("synthetic warning")
        return [report]

    monkeypatch.setattr(runtime_mod, "run_session", fake_run_session)
    result = runner.invoke(app, ["run", "--backend", "offline"])
    assert result.exit_code == 0, result.output
    assert "metrics" in result.output
    assert "u1" in result.output and "120" in result.output
    assert "synthetic warning" in result.output


def test_run_keyboard_interrupt_is_a_normal_stop(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import interpret_live.runtime as runtime_mod

    async def fake_run_session(opts, *, deps=None, on_warning=None):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_mod, "run_session", fake_run_session)
    result = runner.invoke(app, ["run", "--backend", "offline"])
    assert result.exit_code == 0
    assert "stopped." in result.output


def test_models_download_success_prints_resolved_table(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import interpret_live.models as models_mod

    artifact = models_mod.ResolvedArtifact(
        name="piper:es_ES-davefx-medium:model",
        path=str(tmp_path / "voice.onnx"),
        requested_revision="v1.0.0",
        resolved_revision="abc123",
        provenance="direct",
    )

    class FakeManager:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.cache_dir = str(tmp_path)

        def resolve_all(self, spec):  # type: ignore[no-untyped-def]
            return {"piper": artifact}

    monkeypatch.setattr(models_mod, "ModelManager", FakeManager)
    result = runner.invoke(app, ["models", "download", "--to", "es"])
    assert result.exit_code == 0, result.output
    assert "resolved model artifacts" in result.output
    assert "license/source" in result.output


def test_devices_lists_table_with_fake_sd(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    from test_audio_io_real import make_fake_sd

    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd())
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 0, result.output
    assert "fake-device" in result.output
    assert "48000" in result.output
