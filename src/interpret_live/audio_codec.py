"""Audio boundary conversions: PCM16 codec and stateful streaming resampling.

The canonical in-process audio type is mono normalized ``float32`` in
``[-1.0, 1.0]`` with an explicit sample rate (see
:class:`~interpret_live.types.AudioFrame`). This module owns the two boundary
conversions the live adapters need:

* **PCM16 wire encoding** — little-endian 16-bit PCM used by model/provider
  boundaries (faster-whisper input files, Piper output, OpenAI Realtime audio).
* **Stateful rate conversion** — a :class:`StreamingResampler` wrapping
  ``soxr`` that preserves filter/phase state across blocks, flushes its tail
  exactly once at a real stream boundary, and resets cleanly.

``soxr`` is a compiled optional dependency; it is imported lazily inside
:class:`StreamingResampler` (and only when the rates actually differ), so the
light default install and ``bench`` never import it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "StreamingResampler",
    "chunk_duration_ms",
    "float32_to_pcm16",
    "pcm16_to_float32",
]

_PCM16_SCALE = 32768.0


def float32_to_pcm16(samples: NDArray[np.float32]) -> bytes:
    """Encode mono float32 samples as little-endian 16-bit PCM bytes.

    Input is clipped to ``[-1.0, 1.0]`` first, so a boundary producing slight
    overshoot (e.g. a resampler ripple) encodes safely instead of wrapping.
    """
    if samples.ndim != 1:
        raise ValueError(f"expected mono (1-D) samples, got {samples.ndim}D")
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    # Symmetric scale with clamping at +full-scale keeps the round-trip within
    # one quantization step (decode divides by the same 32768).
    ints = np.clip(np.round(clipped * _PCM16_SCALE), -32768, 32767).astype("<i2")
    return ints.tobytes()


def pcm16_to_float32(data: bytes) -> NDArray[np.float32]:
    """Decode little-endian 16-bit PCM bytes to normalized mono float32."""
    if len(data) % 2:
        raise ValueError(f"PCM16 byte stream must have even length, got {len(data)}")
    ints = np.frombuffer(data, dtype="<i2")
    return (ints.astype(np.float32) / _PCM16_SCALE).astype(np.float32)


def chunk_duration_ms(sample_count: int, sample_rate: int) -> float:
    """Exact duration of ``sample_count`` samples at ``sample_rate``, in ms.

    Returned as a float so playback accounting can accumulate without
    truncation drift; round only at presentation/metric boundaries.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    if sample_count < 0:
        raise ValueError(f"sample_count must be >= 0, got {sample_count}")
    return 1000.0 * sample_count / sample_rate


class StreamingResampler:
    """A stateful mono resampler with explicit block/flush/reset semantics.

    One instance owns one *continuous* stream: :meth:`process` converts each
    block while preserving the resampler's internal filter/phase state, so
    chunked conversion matches one-shot conversion; :meth:`flush` emits the
    filter tail exactly once at a real stream/utterance boundary; and
    :meth:`reset` restores a fresh stream (used on stop or a validated rate
    change). Processing after a flush without a reset is a programming error
    and raises.

    When ``in_rate == out_rate`` the resampler is a validated pass-through and
    ``soxr`` is never imported.
    """

    __slots__ = ("_flushed", "_in_rate", "_out_rate", "_quality", "_stream")

    def __init__(self, in_rate: int, out_rate: int, *, quality: str = "HQ") -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError(f"rates must be > 0, got in={in_rate} out={out_rate}")
        self._in_rate = in_rate
        self._out_rate = out_rate
        self._quality = quality
        self._flushed = False
        self._stream: object | None = None

    @property
    def in_rate(self) -> int:
        """Input sample rate in Hz."""
        return self._in_rate

    @property
    def out_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._out_rate

    def _ensure_stream(self) -> object:
        if self._stream is None:
            import soxr  # lazy: compiled optional dependency

            self._stream = soxr.ResampleStream(
                self._in_rate,
                self._out_rate,
                num_channels=1,
                dtype="float32",
                quality=self._quality,
            )
        return self._stream

    def process(self, block: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert one block, preserving filter/phase state across calls."""
        if self._flushed:
            raise RuntimeError("StreamingResampler was flushed; call reset() first")
        if block.ndim != 1:
            raise ValueError(f"expected mono (1-D) block, got {block.ndim}D")
        if self._in_rate == self._out_rate:
            return block.astype(np.float32, copy=False)
        stream = self._ensure_stream()
        out = stream.resample_chunk(  # type: ignore[attr-defined]
            np.ascontiguousarray(block, dtype=np.float32), last=False
        )
        return np.asarray(out, dtype=np.float32)

    def flush(self) -> NDArray[np.float32]:
        """Emit the remaining filter tail exactly once (then require reset)."""
        if self._flushed:
            return np.empty(0, dtype=np.float32)
        self._flushed = True
        if self._in_rate == self._out_rate or self._stream is None:
            return np.empty(0, dtype=np.float32)
        out = self._stream.resample_chunk(  # type: ignore[attr-defined]
            np.empty(0, dtype=np.float32), last=True
        )
        return np.asarray(out, dtype=np.float32)

    def reset(self) -> None:
        """Discard all state; the next :meth:`process` starts a fresh stream."""
        self._stream = None
        self._flushed = False
