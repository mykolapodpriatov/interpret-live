"""ElevenLabsTTS tests against a scripted fake SDK client (no network).

Proves: the exact request it builds, PCM decoding across HTTP chunk boundaries
that split a 16-bit sample, the one-block lookahead that marks only the last
chunk final, the voice preflight, deterministic stream closing on barge-in, and
typed failures instead of raw SDK exceptions.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from interpret_live.audio_codec import float32_to_pcm16
from interpret_live.backends import TTS
from interpret_live.backends.elevenlabs import (
    DEFAULT_ELEVENLABS_MODEL,
    ElevenLabsError,
    ElevenLabsTTS,
)
from interpret_live.types import TtsChunk

VOICE = "voice-123"


class FakeTextToSpeech:
    """Replays canned byte blobs and records the request it was given."""

    def __init__(self, blobs: list[bytes], *, error: Exception | None = None) -> None:
        self.blobs = blobs
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = 0
        self.delivered = 0
        self.gate: asyncio.Event | None = None

    def stream(self, voice_id: str, **options: Any) -> Any:
        self.calls.append((voice_id, options))

        async def gen() -> Any:
            try:
                if self.error is not None:
                    raise self.error
                for blob in self.blobs:
                    if self.gate is not None:
                        await self.gate.wait()
                    self.delivered += 1
                    yield blob
            finally:
                self.closed += 1

        return gen()


class FakeClient:
    def __init__(
        self,
        blobs: list[bytes] | None = None,
        *,
        error: Exception | None = None,
        voice_error: Exception | None = None,
        voice_delay: float | None = None,
    ) -> None:
        self.text_to_speech = FakeTextToSpeech(blobs or [], error=error)
        self.voices = _FakeVoices(voice_error, voice_delay)
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class _FakeVoices:
    def __init__(self, error: Exception | None, delay: float | None) -> None:
        self.error = error
        self.delay = delay
        self.requested: list[str] = []

    async def get(self, voice_id: str) -> object:
        self.requested.append(voice_id)
        if self.delay is not None:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return object()


def _tts(client: FakeClient, **overrides: Any) -> ElevenLabsTTS:
    kwargs: dict[str, Any] = {"voice_id": VOICE, "client": client}
    kwargs.update(overrides)
    return ElevenLabsTTS(**kwargs)


def _pcm(values: list[float]) -> bytes:
    return float32_to_pcm16(np.array(values, dtype=np.float32))


async def _collect(adapter: ElevenLabsTTS, text: str = "hola mundo") -> list[TtsChunk]:
    return [chunk async for chunk in adapter.synthesize(text, segment_index=3, utterance_id="u-1")]


# -- construction ---------------------------------------------------------------


def test_satisfies_the_tts_protocol() -> None:
    assert isinstance(_tts(FakeClient()), TTS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"voice_id": ""},
        {"model_id": ""},
        {"sample_rate": 12345},
        {"latency_optimization": 5},
        {"latency_optimization": -1},
    ],
)
def test_invalid_construction_fails_fast(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _tts(FakeClient(), **overrides)


def test_an_unofferable_rate_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="24000"):
        _tts(FakeClient(), sample_rate=11025)


def test_output_format_is_raw_pcm_at_the_configured_rate() -> None:
    assert _tts(FakeClient(), sample_rate=16000).output_format == "pcm_16000"


# -- the request ----------------------------------------------------------------


async def test_the_request_carries_the_voice_model_and_pcm_format() -> None:
    client = FakeClient([_pcm([0.1, 0.2])])
    await _collect(_tts(client))

    voice_id, options = client.text_to_speech.calls[0]
    assert voice_id == VOICE
    assert options["text"] == "hola mundo"
    assert options["model_id"] == DEFAULT_ELEVENLABS_MODEL
    assert options["output_format"] == "pcm_24000"
    # Omitted rather than sent as None: multilingual_v2 rejects language_code.
    assert "language_code" not in options
    assert "optimize_streaming_latency" not in options


async def test_optional_request_fields_are_sent_only_when_configured() -> None:
    client = FakeClient([_pcm([0.1])])
    await _collect(_tts(client, language_code="es", latency_optimization=3, model_id="flash"))

    _voice_id, options = client.text_to_speech.calls[0]
    assert options["language_code"] == "es"
    assert options["optimize_streaming_latency"] == 3
    assert options["model_id"] == "flash"


# -- decoding -------------------------------------------------------------------


async def test_chunks_decode_to_the_original_samples() -> None:
    samples = [0.0, 0.25, -0.5, 0.75]
    client = FakeClient([_pcm(samples)])

    chunks = await _collect(_tts(client))

    assert len(chunks) == 1
    assert np.allclose(chunks[0].samples, samples, atol=1e-4)
    assert chunks[0].sample_rate == 24000
    assert (chunks[0].segment_index, chunks[0].utterance_id) == (3, "u-1")


async def test_a_sample_split_across_two_http_chunks_is_rejoined() -> None:
    """HTTP boundaries fall anywhere, including inside a 16-bit frame."""
    samples = [0.1, -0.2, 0.3, -0.4]
    pcm = _pcm(samples)
    # Split after 3 bytes: one and a half samples in the first blob.
    client = FakeClient([pcm[:3], pcm[3:]])

    chunks = await _collect(_tts(client))

    joined = np.concatenate([c.samples for c in chunks])
    assert np.allclose(joined, samples, atol=1e-4)


async def test_a_stream_ending_mid_sample_is_a_typed_error() -> None:
    pcm = _pcm([0.1, 0.2])
    client = FakeClient([pcm[:-1]])  # truncated response

    with pytest.raises(ElevenLabsError, match="ended mid-sample"):
        await _collect(_tts(client))


async def test_only_the_last_block_is_final() -> None:
    client = FakeClient([_pcm([0.1]), _pcm([0.2]), _pcm([0.3])])

    chunks = await _collect(_tts(client))

    assert [c.final for c in chunks] == [False, False, True]


async def test_a_single_block_is_still_final() -> None:
    chunks = await _collect(_tts(FakeClient([_pcm([0.1])])))
    assert [c.final for c in chunks] == [True]


async def test_an_empty_leading_blob_never_becomes_a_chunk() -> None:
    """A keep-alive or header-only blob is not a block of silence."""
    client = FakeClient([b"", _pcm([0.1, 0.2])])

    chunks = await _collect(_tts(client))

    assert [c.final for c in chunks] == [True]


async def test_blank_text_is_not_sent_at_all() -> None:
    client = FakeClient([_pcm([0.1])])

    assert await _collect(_tts(client), "   ") == []
    assert client.text_to_speech.calls == []


# -- lifecycle ------------------------------------------------------------------


async def test_start_reads_the_voice_back_as_a_preflight() -> None:
    client = FakeClient()
    await _tts(client).start()
    assert client.voices.requested == [VOICE]


async def test_an_unusable_voice_fails_before_any_audio_device_opens() -> None:
    client = FakeClient(voice_error=RuntimeError("voice_not_found"))

    with pytest.raises(ElevenLabsError, match=f"cannot use ElevenLabs voice {VOICE!r}"):
        await _tts(client).start()


async def test_the_preflight_is_bounded() -> None:
    client = FakeClient(voice_delay=5.0)

    with pytest.raises(ElevenLabsError, match="preflight exceeded"):
        await _tts(client, preflight_timeout_s=0.01).start()


async def test_aclose_releases_the_client_once() -> None:
    client = FakeClient()
    adapter = _tts(client)

    await adapter.aclose()
    await adapter.aclose()

    assert client.closed == 1


async def test_aclose_tolerates_a_client_with_nothing_to_close() -> None:
    class _Bare:
        text_to_speech = FakeTextToSpeech([])
        voices = _FakeVoices(None, None)

    await _tts(_Bare()).aclose()  # type: ignore[arg-type]


# -- failures and barge-in ------------------------------------------------------


async def test_an_sdk_failure_becomes_a_typed_error() -> None:
    client = FakeClient(error=ValueError("429 too many requests"))

    with pytest.raises(ElevenLabsError, match="ElevenLabs synthesis failed"):
        await _collect(_tts(client))


async def test_barge_in_closes_the_http_stream_deterministically() -> None:
    """Cancellation must not leave the response open for the collector."""
    client = FakeClient([_pcm([0.1]), _pcm([0.2]), _pcm([0.3])])
    client.text_to_speech.gate = asyncio.Event()
    client.text_to_speech.gate.set()
    adapter = _tts(client)
    stream = adapter.synthesize("hola", segment_index=0, utterance_id="u-1")

    first = await anext(stream)
    assert first.final is False
    await stream.aclose()  # what a cancelled synthesis stage does

    assert client.text_to_speech.closed == 1
    assert client.text_to_speech.delivered < 3, "the stream kept pulling after the abort"


async def test_cancelling_the_consuming_task_stops_and_closes_the_stream() -> None:
    client = FakeClient([_pcm([0.1])] * 5)
    gate = asyncio.Event()
    client.text_to_speech.gate = gate
    adapter = _tts(client)

    async def consume() -> None:
        async for _chunk in adapter.synthesize("hola", segment_index=0, utterance_id="u-1"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.text_to_speech.closed == 1
