"""Contract tests for :mod:`interpret_live.audio_codec`.

PCM16 round-trips, duration math, and the stateful :class:`StreamingResampler`
(chunked-vs-one-shot equivalence, single flush, reset, lazy soxr import).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from interpret_live.audio_codec import (
    StreamingResampler,
    chunk_duration_ms,
    float32_to_pcm16,
    pcm16_to_float32,
)


def test_pcm16_round_trip_within_quantization_tolerance() -> None:
    rng = np.random.default_rng(42)
    samples = rng.uniform(-1.0, 1.0, size=4096).astype(np.float32)
    decoded = pcm16_to_float32(float32_to_pcm16(samples))
    assert decoded.dtype == np.float32
    assert decoded.shape == samples.shape
    # One PCM16 step is 1/32768; round-trip error must stay within one step.
    assert float(np.max(np.abs(decoded - samples))) <= 1.0 / 32768.0


def test_pcm16_encoding_clips_overshoot_instead_of_wrapping() -> None:
    overshoot = np.array([1.5, -1.5], dtype=np.float32)
    decoded = pcm16_to_float32(float32_to_pcm16(overshoot))
    assert decoded[0] > 0.99  # clipped to +full-scale, not wrapped negative
    assert decoded[1] < -0.99


def test_pcm16_is_little_endian() -> None:
    data = float32_to_pcm16(np.array([1.0], dtype=np.float32))
    assert data == (32767).to_bytes(2, "little", signed=True)


def test_pcm16_decode_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="even length"):
        pcm16_to_float32(b"\x00")


def test_chunk_duration_ms_exact() -> None:
    assert chunk_duration_ms(16000, 16000) == 1000.0
    assert chunk_duration_ms(240, 24000) == 10.0
    assert chunk_duration_ms(0, 48000) == 0.0
    with pytest.raises(ValueError):
        chunk_duration_ms(10, 0)


@pytest.mark.parametrize("out_rate", [16000, 22050, 24000, 48000])
def test_resample_produces_expected_duration(out_rate: int) -> None:
    pytest.importorskip("soxr")
    in_rate = 16000
    seconds = 0.5
    samples = np.sin(2 * np.pi * 440 * np.arange(int(in_rate * seconds)) / in_rate).astype(
        np.float32
    )
    rs = StreamingResampler(in_rate, out_rate)
    out = np.concatenate([rs.process(samples), rs.flush()])
    expected = round(seconds * out_rate)
    # The output length matches the source duration at the new rate exactly
    # (within one sample of rounding).
    assert abs(len(out) - expected) <= 1


def test_chunked_matches_one_shot_and_has_no_drift() -> None:
    pytest.importorskip("soxr")
    in_rate, out_rate = 24000, 16000
    rng = np.random.default_rng(7)
    signal = rng.uniform(-0.5, 0.5, size=in_rate * 2).astype(np.float32)  # 2 s

    one_shot = StreamingResampler(in_rate, out_rate)
    expected = np.concatenate([one_shot.process(signal), one_shot.flush()])

    chunked = StreamingResampler(in_rate, out_rate)
    pieces = []
    # Deliberately irregular block sizes: state must carry across blocks.
    idx = 0
    for size in (7, 480, 1000, 333, 2048, 1, 5000):
        while idx + size <= len(signal):
            pieces.append(chunked.process(signal[idx : idx + size]))
            idx += size
    pieces.append(chunked.process(signal[idx:]))
    pieces.append(chunked.flush())
    got = np.concatenate(pieces)

    # No cumulative sample-count drift...
    assert len(got) == len(expected)
    # ...and the content matches one-shot conversion within filter tolerance.
    assert float(np.max(np.abs(got - expected))) < 1e-4


def test_flush_is_single_and_process_after_flush_requires_reset() -> None:
    pytest.importorskip("soxr")
    rs = StreamingResampler(48000, 16000)
    rs.process(np.zeros(4800, dtype=np.float32))
    tail = rs.flush()
    assert isinstance(tail, np.ndarray)
    assert rs.flush().size == 0  # the tail is emitted exactly once
    with pytest.raises(RuntimeError, match="reset"):
        rs.process(np.zeros(10, dtype=np.float32))
    rs.reset()
    out = rs.process(np.zeros(4800, dtype=np.float32))
    assert out.dtype == np.float32


def test_identity_resampler_never_imports_soxr(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate soxr being absent: the identity path must not even try.
    monkeypatch.setitem(sys.modules, "soxr", None)
    rs = StreamingResampler(16000, 16000)
    block = np.full(160, 0.25, dtype=np.float32)
    assert np.array_equal(rs.process(block), block)
    assert rs.flush().size == 0
    rs.reset()
    assert np.array_equal(rs.process(block), block)


def test_resampler_validates_rates_and_shape() -> None:
    with pytest.raises(ValueError):
        StreamingResampler(0, 16000)
    rs = StreamingResampler(16000, 16000)
    with pytest.raises(ValueError, match="mono"):
        rs.process(np.zeros((10, 2), dtype=np.float32))
