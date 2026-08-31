"""Streaming TTS adapter for ElevenLabs, with voice preservation (``elevenlabs``).

A drop-in :class:`~interpret_live.backends.TTS` for the pipeline path: the
LocalAgreement stabilizer and the segmenter keep working exactly as they do with
Piper, and only the final synthesis step moves to ElevenLabs. Point ``voice_id``
at a **cloned voice** and the translation comes back in the speaker's own voice;
point it at a preset and it is an ordinary high-quality voice.

The adapter never creates, uploads or mutates a voice — cloning is an
account-level action with its own consent requirements, and it stays the
operator's to perform. ``start()`` only *reads* the voice back as a preflight,
so a wrong id fails before any audio device opens rather than mid-sentence.

Design notes:

* **PCM only.** ``output_format`` is always ``pcm_<rate>``: the sink consumes
  canonical float32 samples, so an mp3/Opus stream would need a decoder the
  library deliberately does not carry. The rate is validated against the
  formats the API actually offers.
* **Frame-safe chunking.** HTTP chunk boundaries fall anywhere, including
  *inside* a 16-bit sample. A carry byte joins each split frame to the next
  chunk, so no sample is ever decoded from a half frame.
* **One-chunk lookahead**, as in the Piper adapter: the last produced block —
  and only it — carries ``TtsChunk.final=True``, including the single-block
  case.
* **Barge-in** cancels the coroutine; the underlying HTTP stream is closed
  deterministically through :func:`contextlib.aclosing` rather than left to the
  garbage collector.

``ELEVENLABS_API_KEY`` is read by the SDK from the process environment: it is
never accepted as a constructor value, logged, or included in exceptions.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..audio_codec import pcm16_to_float32
from ..types import TtsChunk
from .guard import require

__all__ = [
    "DEFAULT_ELEVENLABS_MODEL",
    "DEFAULT_ELEVENLABS_SAMPLE_RATE",
    "ElevenLabsError",
    "ElevenLabsTTS",
]

#: Default ElevenLabs model id (multilingual, the safe general choice).
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
#: Default output rate; the sink owns any device-rate conversion.
DEFAULT_ELEVENLABS_SAMPLE_RATE = 24000

#: Sample rates the API offers as raw PCM (``pcm_<rate>`` output formats).
_PCM_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


class ElevenLabsError(RuntimeError):
    """A typed ElevenLabs configuration, voice or synthesis failure."""


async def _aclose(stream: AsyncIterator[bytes]) -> None:
    """Close ``stream`` if it supports it (the SDK's is an async generator)."""
    closer = getattr(stream, "aclose", None)
    if callable(closer):
        with contextlib.suppress(Exception):
            await closer()


class ElevenLabsTTS:
    """Streaming TTS over the ElevenLabs API, optionally in a cloned voice.

    Args:
        voice_id: The voice to speak with — a cloned voice for preservation, or
            a preset. It must already exist on the account.
        model_id: ElevenLabs model id. ``eleven_flash_v2_5`` is the
            lower-latency choice for live interpreting.
        sample_rate: Output rate; must be one the API offers as raw PCM.
        language_code: Optional ISO 639-1 language to enforce. Not supported by
            the ``multilingual_v2`` models, so it is omitted unless set.
        latency_optimization: The API's ``optimize_streaming_latency`` (0–4),
            trading audio quality for first-byte latency. Omitted when ``None``.
        client: Injectable async SDK client (tests); the default constructs
            ``elevenlabs.AsyncElevenLabs()`` with the API key resolved by the
            SDK from the environment.
        preflight_timeout_s: Bounded budget for the ``start()`` voice check.

    Raises:
        MissingExtraError: If the ``elevenlabs`` extra is not installed.
        ValueError: If ``voice_id`` is empty, ``sample_rate`` is not an offered
            PCM rate, or ``latency_optimization`` is out of range.
    """

    def __init__(
        self,
        *,
        voice_id: str,
        model_id: str = DEFAULT_ELEVENLABS_MODEL,
        sample_rate: int = DEFAULT_ELEVENLABS_SAMPLE_RATE,
        language_code: str | None = None,
        latency_optimization: int | None = None,
        client: Any | None = None,
        preflight_timeout_s: float = 15.0,
    ) -> None:
        if not voice_id:
            raise ValueError("voice_id must be a non-empty ElevenLabs voice id")
        if not model_id:
            raise ValueError("model_id must be a non-empty ElevenLabs model id")
        if sample_rate not in _PCM_RATES:
            offered = ", ".join(str(rate) for rate in _PCM_RATES)
            raise ValueError(
                f"sample_rate {sample_rate} is not offered as raw PCM; expected one of: {offered}"
            )
        if latency_optimization is not None and not 0 <= latency_optimization <= 4:
            raise ValueError(
                f"latency_optimization must be between 0 and 4, got {latency_optimization}"
            )
        if client is None:
            el = require("elevenlabs", backend="elevenlabs", extra="elevenlabs")
            client = el.AsyncElevenLabs()  # API key from the environment only
        self._client = client
        self._voice_id = voice_id
        self._model_id = model_id
        self._sample_rate = sample_rate
        self._language_code = language_code
        self._latency_optimization = latency_optimization
        self._preflight_timeout_s = preflight_timeout_s
        self._closed = False

    @property
    def sample_rate(self) -> int:
        """The configured output rate, in Hz."""
        return self._sample_rate

    @property
    def output_format(self) -> str:
        """The API output format this adapter requests."""
        return f"pcm_{self._sample_rate}"

    # ----- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Read the configured voice back, so a wrong id fails before devices.

        Raises:
            ElevenLabsError: If the voice cannot be read within the preflight
                budget — a typo, a revoked key, or an unreachable API.
        """
        try:
            async with asyncio.timeout(self._preflight_timeout_s):
                await self._client.voices.get(self._voice_id)
        except TimeoutError as exc:
            raise ElevenLabsError(
                f"ElevenLabs voice preflight exceeded {self._preflight_timeout_s:.1f}s"
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ElevenLabsError(f"cannot use ElevenLabs voice {self._voice_id!r}: {exc}") from exc

    async def aclose(self) -> None:
        """Release the SDK client's resources, if it exposes a way to (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for name in ("aclose", "close"):
            closer = getattr(self._client, name, None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
                return

    # ----- synthesis ---------------------------------------------------------

    def synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        """Return the streamed chunk iterator for ``text``."""
        return self._synthesize(text, segment_index=segment_index, utterance_id=utterance_id)

    def _request(self, text: str) -> AsyncIterator[bytes]:
        """Open the API's streaming synthesis for ``text``."""
        options: dict[str, Any] = {
            "text": text,
            "model_id": self._model_id,
            "output_format": self.output_format,
        }
        if self._language_code is not None:
            options["language_code"] = self._language_code
        if self._latency_optimization is not None:
            options["optimize_streaming_latency"] = self._latency_optimization
        stream: AsyncIterator[bytes] = self._client.text_to_speech.stream(self._voice_id, **options)
        return stream

    async def _synthesize(
        self, text: str, *, segment_index: int, utterance_id: str
    ) -> AsyncIterator[TtsChunk]:
        if not text.strip():
            return  # nothing to say: no chunks, not a silent block
        carry = b""
        pending: NDArray[np.float32] | None = None
        blobs = self._request(text)
        try:
            async for blob in blobs:
                data = carry + bytes(blob)
                if len(data) % 2:
                    # An HTTP chunk boundary fell inside a 16-bit sample: the
                    # odd byte belongs to the next chunk's first frame.
                    data, carry = data[:-1], data[-1:]
                else:
                    carry = b""
                if not data:
                    continue
                samples = pcm16_to_float32(data)
                if pending is not None:
                    # One-block lookahead: only the last block is final.
                    yield self._chunk(pending, segment_index, utterance_id, final=False)
                pending = samples
        except asyncio.CancelledError:
            raise
        except ElevenLabsError:
            raise
        except Exception as exc:
            raise ElevenLabsError(f"ElevenLabs synthesis failed: {exc}") from exc
        finally:
            # On barge-in the coroutine is cancelled here; the HTTP stream must
            # close now, not whenever the generator happens to be collected.
            await _aclose(blobs)
        if carry:
            raise ElevenLabsError(
                "ElevenLabs audio stream ended mid-sample (odd trailing byte); "
                "the response was truncated"
            )
        if pending is not None:
            yield self._chunk(pending, segment_index, utterance_id, final=True)

    def _chunk(
        self,
        samples: NDArray[np.float32],
        segment_index: int,
        utterance_id: str,
        *,
        final: bool,
    ) -> TtsChunk:
        return TtsChunk(
            samples=np.clip(samples, -1.0, 1.0),
            sample_rate=self._sample_rate,
            segment_index=segment_index,
            utterance_id=utterance_id,
            final=final,
        )
