"""ModelManager tests with a fake downloader/cache (no network, no models).

Covers the manifest, atomic verified downloads, checksum failure handling,
cache reuse, retry policy (transient-only), offline aggregation, and the
whisper alias/pinned-revision mapping.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any
from urllib import error as urlerror

import pytest

from interpret_live.models import (
    NLLB_REPO,
    WHISPER_REPOS,
    ChecksumMismatchError,
    ModelManager,
    ModelResolutionError,
    OfflineArtifactsMissingError,
    PrefetchSpec,
    load_piper_manifest,
)

VOICE = "es_ES-davefx-medium"


def _manifest_entry() -> dict[str, Any]:
    return dict(load_piper_manifest()["voices"][VOICE])


class FakeFetch:
    """A scripted stand-in for the direct HTTP downloader."""

    def __init__(self, payloads: dict[str, bytes], *, failures: int = 0) -> None:
        self.payloads = payloads
        self.failures = failures
        self.calls: list[str] = []
        self.progress_lines: list[str] = []

    def __call__(self, url: str, dest: str, progress: Callable[[str], None]) -> None:
        self.calls.append(url)
        progress("50.0% of 1.0 MB")
        self.progress_lines.append(url)
        if self.failures > 0:
            self.failures -= 1
            raise urlerror.URLError("connection reset")
        with open(dest, "wb") as out:
            out.write(self.payloads[url])


def _correct_payloads() -> dict[str, bytes]:
    """Payloads whose SHA-256 matches the shipped manifest? No — inverse:

    We can't invent content matching the manifest's real checksums, so tests
    inject a manager whose expected checksums come from these payloads.
    """
    entry = _manifest_entry()
    return {
        str(entry["model_url"]): b"fake-onnx-bytes",
        str(entry["config_url"]): b'{"audio": {"sample_rate": 22050}}',
    }


@pytest.fixture
def patched_manifest(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Point the manifest checksums at the fake payloads (still real URLs)."""
    payloads = _correct_payloads()
    manifest = load_piper_manifest()
    entry = dict(manifest["voices"][VOICE])
    entry["model_sha256"] = hashlib.sha256(payloads[str(entry["model_url"])]).hexdigest()
    entry["config_sha256"] = hashlib.sha256(payloads[str(entry["config_url"])]).hexdigest()
    manifest["voices"][VOICE] = entry
    monkeypatch.setattr("interpret_live.models.load_piper_manifest", lambda: manifest)
    return payloads


def test_manifest_ships_with_verified_fields_and_defaults() -> None:
    manifest = load_piper_manifest()
    assert manifest["defaults"], "language -> voice defaults must exist"
    for voice_id, entry in manifest["voices"].items():
        for key in (
            "language",
            "sample_rate",
            "model_url",
            "model_sha256",
            "config_url",
            "config_sha256",
            "license_url",
        ):
            assert entry.get(key), f"{voice_id} manifest entry missing {key}"
        assert len(entry["model_sha256"]) == 64
        assert len(entry["config_sha256"]) == 64
    for lang, voice_id in manifest["defaults"].items():
        assert voice_id in manifest["voices"], f"default for {lang} missing"


def test_direct_download_is_atomic_verified_and_cached(
    tmp_path: Any, patched_manifest: dict[str, bytes]
) -> None:
    fetch = FakeFetch(patched_manifest)
    manager = ModelManager(cache_dir=str(tmp_path), fetcher=fetch)
    artifacts = manager.resolve_piper_voice(VOICE)
    assert [a.provenance for a in artifacts] == ["direct", "direct"]
    for artifact in artifacts:
        assert os.path.isfile(artifact.path)
        assert artifact.resolved_revision  # the manifest checksum
    # No stray partial/lock leftovers next to the artifacts.
    voice_dir = os.path.dirname(artifacts[0].path)
    assert not [f for f in os.listdir(voice_dir) if f.endswith(".part")]
    # Second resolve: pure cache hit, zero network calls.
    calls_before = len(fetch.calls)
    again = manager.resolve_piper_voice(VOICE)
    assert len(fetch.calls) == calls_before
    assert [a.path for a in again] == [a.path for a in artifacts]


def test_corrupt_download_is_rejected_and_leaves_no_cache_entry(
    tmp_path: Any, patched_manifest: dict[str, bytes]
) -> None:
    corrupted = dict(patched_manifest)
    entry = _manifest_entry()
    corrupted[str(entry["model_url"])] = b"tampered-bytes"
    fetch = FakeFetch(corrupted)
    manager = ModelManager(cache_dir=str(tmp_path), fetcher=fetch)
    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        manager.resolve_piper_voice(VOICE)
    # Checksum failures never retry and never populate the cache.
    assert len(fetch.calls) == 1
    voice_dir = os.path.join(str(tmp_path), "piper", VOICE)
    if os.path.isdir(voice_dir):
        assert not [f for f in os.listdir(voice_dir) if f.endswith(".onnx")]
        assert not [f for f in os.listdir(voice_dir) if f.endswith(".part")]


def test_transient_failures_retry_then_succeed(
    tmp_path: Any, patched_manifest: dict[str, bytes]
) -> None:
    fetch = FakeFetch(patched_manifest, failures=2)  # two resets, then success
    manager = ModelManager(cache_dir=str(tmp_path), fetcher=fetch)
    artifacts = manager.resolve_piper_voice(VOICE)
    assert artifacts
    entry = _manifest_entry()
    model_calls = [u for u in fetch.calls if u == str(entry["model_url"])]
    assert len(model_calls) == 3  # 2 failures + 1 success


def test_non_transient_http_failure_does_not_retry(
    tmp_path: Any, patched_manifest: dict[str, bytes]
) -> None:
    class Fetch404(FakeFetch):
        def __call__(self, url: str, dest: str, progress: Callable[[str], None]) -> None:
            self.calls.append(url)
            raise urlerror.HTTPError(url, 404, "not found", {}, None)  # type: ignore[arg-type]

    fetch = Fetch404(patched_manifest)
    manager = ModelManager(cache_dir=str(tmp_path), fetcher=fetch)
    with pytest.raises(ModelResolutionError, match="1 attempt"):
        manager.resolve_piper_voice(VOICE)
    assert len(fetch.calls) == 1


def test_offline_reports_every_missing_artifact_in_one_error(tmp_path: Any) -> None:
    def no_snapshot(*args: Any, **kwargs: Any) -> str:
        raise FileNotFoundError("no local snapshot")

    manager = ModelManager(cache_dir=str(tmp_path), offline=True, hf_snapshot=no_snapshot)
    spec = PrefetchSpec(whisper_model="small", nllb_model=NLLB_REPO, piper_voice=VOICE)
    with pytest.raises(OfflineArtifactsMissingError) as excinfo:
        manager.resolve_all(spec)
    missing = excinfo.value.missing
    assert any("whisper" in m for m in missing)
    assert any("nllb" in m for m in missing)
    assert sum("piper" in m for m in missing) == 2  # model + config
    assert "models download" in str(excinfo.value)


def test_offline_with_complete_cache_resolves_without_network(
    tmp_path: Any, patched_manifest: dict[str, bytes]
) -> None:
    warm = ModelManager(cache_dir=str(tmp_path), fetcher=FakeFetch(patched_manifest))
    warm.resolve_piper_voice(VOICE)

    def forbidden_fetch(url: str, dest: str, progress: Callable[[str], None]) -> None:
        raise AssertionError("offline mode must never fetch")

    snapshots: list[tuple[str, str, bool]] = []

    def local_snapshot(repo: str, revision: str, cache_dir: str, *, local_files_only: bool) -> str:
        snapshots.append((repo, revision, local_files_only))
        path = os.path.join(cache_dir, "snapshots", revision)
        os.makedirs(path, exist_ok=True)
        return path

    manager = ModelManager(
        cache_dir=str(tmp_path),
        offline=True,
        fetcher=forbidden_fetch,
        hf_snapshot=local_snapshot,
    )
    resolved = manager.resolve_all(
        PrefetchSpec(whisper_model="small", nllb_model=NLLB_REPO, piper_voice=VOICE)
    )
    assert {a.provenance for a in resolved.values()} == {"huggingface", "direct"}
    # The Hub was consulted strictly locally.
    assert all(local_only for *_x, local_only in snapshots)


def test_whisper_aliases_resolve_to_pinned_repo_revisions(tmp_path: Any) -> None:
    seen: list[tuple[str, str]] = []

    def fake_snapshot(repo: str, revision: str, cache_dir: str, *, local_files_only: bool) -> str:
        seen.append((repo, revision))
        path = os.path.join(cache_dir, "snapshots", revision)
        os.makedirs(path, exist_ok=True)
        return path

    manager = ModelManager(cache_dir=str(tmp_path), hf_snapshot=fake_snapshot)
    artifact = manager.resolve_whisper("small")
    repo, revision = WHISPER_REPOS["small"]
    assert seen == [(repo, revision)]
    assert artifact.provenance == "huggingface"
    assert artifact.requested_revision == revision
    assert artifact.resolved_revision == revision  # snapshot dir basename

    with pytest.raises(ModelResolutionError, match="unsupported faster-whisper"):
        manager.resolve_whisper("gigantic-v9")


def test_local_paths_bypass_the_hub_entirely(tmp_path: Any) -> None:
    local = tmp_path / "my-model"
    local.mkdir()

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("local paths must not consult the Hub")

    manager = ModelManager(cache_dir=str(tmp_path), hf_snapshot=forbidden)
    assert manager.resolve_whisper(str(local)).provenance == "local"
    assert manager.resolve_nllb(str(local)).provenance == "local"


def test_unknown_voice_lists_manifest_options(tmp_path: Any) -> None:
    manager = ModelManager(cache_dir=str(tmp_path))
    with pytest.raises(ModelResolutionError, match="unknown Piper voice"):
        manager.resolve_piper_voice("nope-voice")


def test_default_cache_dir_and_payload_round_trip() -> None:
    from interpret_live.models import default_cache_dir

    assert "interpret-live" in default_cache_dir()
    spec = PrefetchSpec(whisper_model="small", piper_voice=VOICE)
    payload = spec.to_payload()
    assert payload == {"whisper_model": "small", "nllb_model": None, "piper_voice": VOICE}
    assert PrefetchSpec(**payload) == spec


def test_default_fetcher_streams_with_progress(tmp_path: Any) -> None:
    from interpret_live.models import _default_fetcher

    source = tmp_path / "artifact.bin"
    source.write_bytes(b"x" * 4096)
    dest = tmp_path / "dest.bin"
    lines: list[str] = []
    _default_fetcher(source.as_uri(), str(dest), lines.append)
    assert dest.read_bytes() == b"x" * 4096
    assert lines, "progress must be visible"


def test_build_preflight_handler_offline_raises_missing(tmp_path: Any) -> None:
    import threading

    from interpret_live.models import build_preflight_handler

    handler = build_preflight_handler(cache_dir=str(tmp_path), offline=True)
    with pytest.raises(OfflineArtifactsMissingError):
        handler(PrefetchSpec(piper_voice=VOICE).to_payload(), threading.Event())


async def test_prefetch_in_worker_offline_surfaces_typed_error(tmp_path: Any) -> None:
    from interpret_live.models import prefetch_in_worker

    with pytest.raises(OfflineArtifactsMissingError):
        await prefetch_in_worker(
            PrefetchSpec(piper_voice=VOICE),
            cache_dir=str(tmp_path),
            offline=True,
        )


def test_purge_removes_cache_root(tmp_path: Any, patched_manifest: dict[str, bytes]) -> None:
    manager = ModelManager(cache_dir=str(tmp_path / "c"), fetcher=FakeFetch(patched_manifest))
    manager.resolve_piper_voice(VOICE)
    assert os.path.isdir(manager.cache_dir)
    manager.purge()
    assert not os.path.isdir(manager.cache_dir)
