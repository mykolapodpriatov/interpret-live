"""CLI tests: ``bench`` over fakes prints metrics; exit codes; ``run`` hints.

All offline — ``bench`` drives the deterministic fake pipeline; ``run`` prints
install hints (no live audio in CI); ``devices`` reports the missing-extra error.
"""

from __future__ import annotations

from typer.testing import CliRunner

from interpret_live import __version__
from interpret_live.cli import app

runner = CliRunner()


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


def test_run_offline_prints_extra_hint() -> None:
    result = runner.invoke(app, ["run", "--from", "en", "--to", "es", "--backend", "offline"])
    assert result.exit_code == 0
    assert "interpret-live[whisper" in result.output


def test_run_cloud_prints_provider_hint() -> None:
    result = runner.invoke(app, ["run", "--backend", "cloud"])
    assert result.exit_code == 0
    assert "interpret-live[openai]" in result.output


def test_run_unknown_backend_exits_nonzero() -> None:
    result = runner.invoke(app, ["run", "--backend", "bogus"])
    assert result.exit_code == 2
    assert "unknown backend" in result.output


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
