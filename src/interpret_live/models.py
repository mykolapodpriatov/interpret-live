"""Visible, cache-aware model preflight with an explicit offline mode.

The :class:`ModelManager` owns every model artifact the offline pipeline
needs, with the guarantees from plan Task 5:

* A platform-appropriate cache root from ``platformdirs`` (overridable with
  ``--cache-dir``), one ``filelock`` per directly managed artifact,
  temporary-file downloads, and an atomic rename — an interrupted or corrupt
  download can never be mistaken for a valid cache entry.
* Directly managed Piper files are verified against manifest SHA-256 values;
  Hugging Face artifacts are fetched as explicit repository+revision
  snapshots (pinned commit SHAs) through the Hub cache's own integrity
  machinery, and the resolved commit is recorded.
* Offline mode performs no network access and reports every missing artifact
  in one actionable :class:`OfflineArtifactsMissingError`.
* Only *transient* download failures retry (up to three attempts with bounded
  exponential backoff); checksum, authorization, and invalid-manifest
  failures never retry, and partial files are cleaned on final failure.
* For the async runtime, :func:`prefetch_in_worker` runs the whole (blocking)
  preflight inside a short-lived spawned
  :class:`~interpret_live.model_worker.ModelWorker`, so cancellation
  terminates/joins a real process (releasing its file locks via process
  exit) instead of abandoning an uncancellable executor future.

Nothing here is imported by ``bench``, ``devices``, ``--help``, or a
cloud-only run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from .model_worker import ModelWorker, raise_if_cancelled

__all__ = [
    "WHISPER_REPOS",
    "ChecksumMismatchError",
    "ModelManager",
    "ModelResolutionError",
    "OfflineArtifactsMissingError",
    "PrefetchSpec",
    "ResolvedArtifact",
    "build_preflight_handler",
    "default_cache_dir",
    "load_piper_manifest",
    "prefetch_in_worker",
]

#: Supported faster-whisper aliases resolved to explicit (repo, revision).
WHISPER_REPOS: dict[str, tuple[str, str]] = {
    "tiny": ("Systran/faster-whisper-tiny", "d90ca5fe260221311c53c58e660288d3deb8d356"),
    "base": ("Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
    "medium": ("Systran/faster-whisper-medium", "08e178d48790749d25932bbc082711ddcfdfbc4f"),
    "large-v3": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478"),
}

#: Default NLLB repository and its pinned revision.
NLLB_REPO = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

_RETRIES = 3
_BACKOFF_S = 0.5


class ModelResolutionError(RuntimeError):
    """A typed model acquisition/validation failure."""


class ChecksumMismatchError(ModelResolutionError):
    """A downloaded artifact did not match its manifest SHA-256 (never retried)."""


class OfflineArtifactsMissingError(ModelResolutionError):
    """Offline mode: one or more required artifacts are absent from the cache."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        lines = "\n".join(f"  - {item}" for item in missing)
        super().__init__(
            "offline mode: required model artifacts are missing from the cache:\n"
            f"{lines}\n"
            "Run 'interpret-live models download' (without --offline) to fetch them."
        )


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """A resolved local model artifact with provenance.

    Attributes:
        name: Logical artifact name (e.g. ``whisper:small``).
        path: Absolute local snapshot directory or file path.
        requested_revision: The revision/URL revision that was asked for.
        resolved_revision: The resolved commit SHA or content checksum.
        provenance: ``"huggingface"``, ``"direct"``, or ``"local"``.
    """

    name: str
    path: str
    requested_revision: str | None
    resolved_revision: str | None
    provenance: str


@dataclass(frozen=True, slots=True)
class PrefetchSpec:
    """Which artifacts a run needs (serializable for the preflight worker)."""

    whisper_model: str | None = None
    nllb_model: str | None = None
    piper_voice: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def default_cache_dir() -> str:
    """The platform-appropriate cache root for interpret-live models."""
    import platformdirs

    return platformdirs.user_cache_dir("interpret-live")


def load_piper_manifest() -> dict[str, Any]:
    """Load the shipped Piper voice manifest (package data)."""
    data = resources.files("interpret_live").joinpath("data/piper_voices.json").read_text()
    manifest = json.loads(data)
    if "voices" not in manifest or "defaults" not in manifest:
        raise ModelResolutionError("piper_voices.json manifest is invalid (missing keys)")
    return dict(manifest)


def _default_fetcher(url: str, dest: str, progress: Callable[[str], None]) -> None:
    """Stream ``url`` to ``dest`` with visible byte progress."""
    request = urlrequest.Request(url, headers={"User-Agent": "interpret-live"})
    with urlrequest.urlopen(request, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1024 * 256)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                progress(f"{done / total:6.1%} of {total / 1e6:.1f} MB")
            else:
                progress(f"{done / 1e6:.1f} MB")


def _default_hf_snapshot(
    repo_id: str, revision: str, cache_dir: str, *, local_files_only: bool
) -> str:  # pragma: no cover - thin Hub delegation (needs the whisper/mt extras)
    """Fetch/resolve a Hugging Face snapshot; returns the local snapshot path."""
    from huggingface_hub import snapshot_download

    return str(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def _is_transient(exc: BaseException) -> bool:
    """Only network-ish failures retry; auth/not-found/integrity never do."""
    if isinstance(exc, urlerror.HTTPError):
        return exc.code >= 500 or exc.code == 429
    return isinstance(exc, urlerror.URLError | TimeoutError | ConnectionError | OSError)


class ModelManager:
    """Resolve, download, verify, and cache every offline model artifact.

    Blocking by design: it runs inside the CLI (``models download``) or inside
    the short-lived spawned preflight worker — never on the event loop.

    Args:
        cache_dir: Cache root (defaults to the platformdirs location).
        offline: Perform no network access; missing artifacts raise.
        fetcher: Injectable direct-download callable (tests).
        hf_snapshot: Injectable Hugging Face snapshot callable (tests).
        progress: Sink for human-visible progress lines.
    """

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        offline: bool = False,
        fetcher: Callable[[str, str, Callable[[str], None]], None] | None = None,
        hf_snapshot: Callable[..., str] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._cache_dir = os.path.abspath(cache_dir or default_cache_dir())
        self._offline = offline
        self._fetcher = fetcher or _default_fetcher
        self._hf_snapshot = hf_snapshot or _default_hf_snapshot
        self._progress = progress or (lambda _line: None)

    @property
    def cache_dir(self) -> str:
        """The cache root in use."""
        return self._cache_dir

    # ----- aggregate resolution -------------------------------------------------

    def resolve_all(self, spec: PrefetchSpec) -> dict[str, ResolvedArtifact]:
        """Resolve every artifact in ``spec``.

        In offline mode all missing artifacts are collected and reported in a
        single :class:`OfflineArtifactsMissingError` rather than one at a time.
        """
        resolved: dict[str, ResolvedArtifact] = {}
        missing: list[str] = []
        steps: list[tuple[str, Callable[[], ResolvedArtifact | list[ResolvedArtifact]]]] = []
        if spec.whisper_model:
            steps.append(("whisper", lambda: self.resolve_whisper(spec.whisper_model or "")))
        if spec.nllb_model:
            steps.append(("nllb", lambda: self.resolve_nllb(spec.nllb_model or "")))
        if spec.piper_voice:
            steps.append(("piper", lambda: self.resolve_piper_voice(spec.piper_voice or "")))
        for key, step in steps:
            try:
                result = step()
            except OfflineArtifactsMissingError as exc:
                missing.extend(exc.missing)
                continue
            if isinstance(result, list):
                for artifact in result:
                    resolved[artifact.name] = artifact
            else:
                resolved[key] = result
        if missing:
            raise OfflineArtifactsMissingError(missing)
        return resolved

    # ----- Hugging Face snapshots -------------------------------------------------

    def resolve_whisper(self, model: str) -> ResolvedArtifact:
        """Resolve a faster-whisper alias or local path to a snapshot path."""
        if os.path.exists(model):
            return ResolvedArtifact(
                name=f"whisper:{model}",
                path=os.path.abspath(model),
                requested_revision=None,
                resolved_revision=None,
                provenance="local",
            )
        if model not in WHISPER_REPOS:
            raise ModelResolutionError(
                f"unsupported faster-whisper model {model!r}; supported aliases: "
                f"{', '.join(sorted(WHISPER_REPOS))} (or pass a local snapshot path)"
            )
        repo, revision = WHISPER_REPOS[model]
        return self._snapshot(f"whisper:{model}", repo, revision)

    def resolve_nllb(self, model_name: str, revision: str | None = None) -> ResolvedArtifact:
        """Resolve the NLLB repository (or local path) to a snapshot path."""
        if os.path.exists(model_name):
            return ResolvedArtifact(
                name=f"nllb:{model_name}",
                path=os.path.abspath(model_name),
                requested_revision=None,
                resolved_revision=None,
                provenance="local",
            )
        requested = revision or (NLLB_REVISION if model_name == NLLB_REPO else "main")
        return self._snapshot(f"nllb:{model_name}", model_name, requested)

    def _snapshot(self, name: str, repo: str, revision: str) -> ResolvedArtifact:
        hf_cache = os.path.join(self._cache_dir, "hf")
        os.makedirs(hf_cache, exist_ok=True)
        self._progress(f"{name}: resolving {repo}@{revision[:12]}")
        try:
            path = self._hf_snapshot(repo, revision, hf_cache, local_files_only=self._offline)
        except Exception as exc:
            if self._offline:
                raise OfflineArtifactsMissingError(
                    [f"{name}: Hugging Face snapshot {repo}@{revision[:12]}"]
                ) from exc
            raise ModelResolutionError(f"{name}: failed to fetch {repo}@{revision}: {exc}") from exc
        resolved = os.path.basename(os.path.normpath(path))
        return ResolvedArtifact(
            name=name,
            path=path,
            requested_revision=revision,
            resolved_revision=resolved,
            provenance="huggingface",
        )

    # ----- directly managed Piper artifacts ---------------------------------------

    def resolve_piper_voice(self, voice: str) -> list[ResolvedArtifact]:
        """Resolve a manifest voice id (or local ``.onnx`` path) to file paths.

        Returns two artifacts: the model and its config, in that order.
        """
        if voice.endswith(".onnx") or os.path.exists(voice):
            model_path = os.path.abspath(voice)
            config_path = model_path + ".json"
            if not os.path.isfile(model_path):
                raise ModelResolutionError(f"piper voice model not found: {model_path}")
            return [
                ResolvedArtifact(
                    name=f"piper:{os.path.basename(model_path)}",
                    path=model_path,
                    requested_revision=None,
                    resolved_revision=None,
                    provenance="local",
                ),
                ResolvedArtifact(
                    name=f"piper-config:{os.path.basename(config_path)}",
                    path=config_path,
                    requested_revision=None,
                    resolved_revision=None,
                    provenance="local",
                ),
            ]
        manifest = load_piper_manifest()
        entry = manifest["voices"].get(voice)
        if entry is None:
            raise ModelResolutionError(
                f"unknown Piper voice {voice!r}; manifest voices: "
                f"{', '.join(sorted(manifest['voices']))} (or pass a local .onnx path)"
            )
        revision = str(manifest.get("revision", ""))
        voice_dir = os.path.join(self._cache_dir, "piper", voice)
        missing: list[str] = []
        results: list[ResolvedArtifact] = []
        for kind, url_key, sha_key in (
            ("model", "model_url", "model_sha256"),
            ("config", "config_url", "config_sha256"),
        ):
            url = str(entry[url_key])
            sha = str(entry[sha_key])
            dest = os.path.join(voice_dir, os.path.basename(url))
            try:
                self._download_verified(f"piper:{voice}:{kind}", url, sha, dest)
            except OfflineArtifactsMissingError as exc:
                missing.extend(exc.missing)
                continue
            results.append(
                ResolvedArtifact(
                    name=f"piper:{voice}:{kind}",
                    path=dest,
                    requested_revision=revision,
                    resolved_revision=sha,
                    provenance="direct",
                )
            )
        if missing:
            raise OfflineArtifactsMissingError(missing)
        return results

    def _download_verified(self, name: str, url: str, sha256: str, dest: str) -> None:
        """Download ``url`` to ``dest`` atomically, verifying its checksum.

        A completed ``dest`` is trusted (it can only exist via a verified
        atomic rename). Concurrent runs serialize on a per-artifact lock and
        either share the completed artifact or wait for it.
        """
        if os.path.isfile(dest):
            return
        if self._offline:
            raise OfflineArtifactsMissingError([f"{name}: {os.path.basename(dest)} ({url})"])
        from filelock import FileLock

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with FileLock(dest + ".lock"):
            if os.path.isfile(dest):  # another process completed it meanwhile
                return
            self._clean_stale_parts(dest)
            tmp = f"{dest}.{os.getpid()}.part"
            last_error: BaseException | None = None
            for attempt in range(1, _RETRIES + 1):
                try:
                    self._progress(f"{name}: downloading {url} -> {dest}")
                    self._fetcher(url, tmp, lambda line: self._progress(f"{name}: {line}"))
                    digest = _sha256_of(tmp)
                    if digest != sha256:
                        raise ChecksumMismatchError(
                            f"{name}: checksum mismatch for {url}: expected {sha256}, got {digest}"
                        )
                    os.replace(tmp, dest)  # atomic: dest only ever holds verified bytes
                    self._progress(f"{name}: done ({dest})")
                    return
                except ChecksumMismatchError:
                    self._remove_quiet(tmp)
                    raise
                except BaseException as exc:
                    self._remove_quiet(tmp)
                    last_error = exc
                    if not _is_transient(exc) or attempt == _RETRIES:
                        raise ModelResolutionError(
                            f"{name}: download failed ({attempt} attempt(s)): {exc}"
                        ) from exc
                    time.sleep(_BACKOFF_S * (2 ** (attempt - 1)))
            raise ModelResolutionError(f"{name}: download failed: {last_error}")

    @staticmethod
    def _clean_stale_parts(dest: str) -> None:
        directory = os.path.dirname(dest)
        base = os.path.basename(dest)
        for entry in os.listdir(directory):
            if entry.startswith(base) and entry.endswith(".part"):
                with contextlib.suppress(OSError):
                    os.unlink(os.path.join(directory, entry))

    @staticmethod
    def _remove_quiet(path: str) -> None:
        with contextlib.suppress(OSError):
            os.unlink(path)

    def purge(self) -> None:
        """Delete the entire cache root (used by tests/maintenance)."""
        shutil.rmtree(self._cache_dir, ignore_errors=True)


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_preflight_handler(*, cache_dir: str | None, offline: bool) -> Any:
    """Child-process factory for the spawned preflight worker."""
    manager = ModelManager(cache_dir=cache_dir, offline=offline, progress=print)

    def handle(payload: dict[str, Any], cancel_event: Any) -> dict[str, Any]:
        raise_if_cancelled(cancel_event)
        spec = PrefetchSpec(**payload)
        resolved = manager.resolve_all(spec)
        raise_if_cancelled(cancel_event)
        return {key: asdict(artifact) for key, artifact in resolved.items()}

    return handle


async def prefetch_in_worker(
    spec: PrefetchSpec,
    *,
    cache_dir: str | None = None,
    offline: bool = False,
    timeout_s: float = 3600.0,
    grace_s: float = 2.0,
) -> dict[str, ResolvedArtifact]:
    """Run the blocking preflight inside a short-lived spawned process.

    Cancellation terminates and joins that process — its file locks release
    via process exit and its partial files are cleaned by the next run — so
    CLI shutdown can never hang on an uncancellable executor future.
    """
    worker = ModelWorker(
        "interpret_live.models:build_preflight_handler",
        {"cache_dir": cache_dir, "offline": offline},
        name="model-preflight",
        ready_timeout_s=60.0,
        grace_s=grace_s,
    )
    try:
        await worker.start()
        status, value = await worker.request(spec.to_payload())
        if status == "cancelled":
            raise ModelResolutionError("model preflight was cancelled")
        if status == "error":
            text = str(value)
            if "OfflineArtifactsMissingError" in text:
                raise OfflineArtifactsMissingError([text])
            raise ModelResolutionError(f"model preflight failed: {text}")
        return {key: ResolvedArtifact(**artifact) for key, artifact in value.items()}
    finally:
        await worker.aclose()
