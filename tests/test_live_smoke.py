"""Opt-in manual live smoke for maintainers (never runs in ordinary CI).

Requires the relevant extras, local models (or network to fetch them), and —
for the cloud test — ``OPENAI_API_KEY`` in the environment:

    pip install -e '.[whisper,mt,piper,audio,openai,dev]'
    pytest -m live_smoke tests/test_live_smoke.py

The full audible smoke (speak into the microphone, hear the translation,
barge-in, Ctrl-C) stays a manual CLI procedure — see docs/live-adapters.md.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

pytestmark = pytest.mark.live_smoke


def _need(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed (install the matching extra)")


async def test_offline_stack_reaches_readiness() -> None:
    """Prefetch models and bring every offline worker to readiness, then close.

    Proves: visible cached prefetch, worker spawn + model construction from
    the resolved local paths, and clean bounded shutdown (no child PIDs).
    """
    _need("faster_whisper")
    _need("transformers")
    _need("piper")

    from interpret_live.backends.nllb import NllbMT
    from interpret_live.backends.piper import PiperTTS
    from interpret_live.backends.whisper import WhisperSTT
    from interpret_live.models import NLLB_REPO, PrefetchSpec, prefetch_in_worker

    resolved = await prefetch_in_worker(
        PrefetchSpec(
            whisper_model="tiny",  # smallest supported alias for the smoke
            nllb_model=NLLB_REPO,
            piper_voice="es_ES-davefx-medium",
        )
    )
    stt = WhisperSTT(model_size=resolved["whisper"].path, language="en")
    mt = NllbMT(source_lang="en", target_lang="es", model_name=resolved["nllb"].path)
    tts = PiperTTS(
        model_path=resolved["piper:es_ES-davefx-medium:model"].path,
        config_path=resolved["piper:es_ES-davefx-medium:config"].path,
    )
    for adapter in (stt, mt, tts):
        await adapter.start()
    for adapter in (tts, mt, stt):
        await adapter.aclose()


async def test_openai_realtime_session_configures_and_closes() -> None:
    """Open a real Realtime connection, configure the session, close cleanly."""
    _need("openai")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from interpret_live.backends.realtime import RealtimeS2S
    from interpret_live.types import AudioFrame

    async def no_audio() -> object:
        return
        yield AudioFrame  # pragma: no cover - makes this an async generator

    adapter = RealtimeS2S(source_lang="en", target_lang="es")
    events = [event async for event in adapter.stream(no_audio())]  # type: ignore[arg-type]
    # No speech was sent: the EOF machine must close without commit/create and
    # without any error event surfacing as an exception.
    assert events == []
