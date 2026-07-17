"""Canonical-audio contract validation on :class:`AudioFrame`/:class:`TtsChunk`.

Also proves the light default install stays light: importing the package and
running ``bench`` must not pull in any real audio dependency.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from interpret_live.types import AudioFrame, TtsChunk


def _frame(samples: np.ndarray, rate: int = 16000) -> AudioFrame:
    return AudioFrame(samples=samples, sample_rate=rate, t_ms=0)


def test_audio_frame_accepts_canonical_mono_float32() -> None:
    frame = _frame(np.zeros(320, dtype=np.float32))
    assert frame.duration_ms == 20


def test_audio_frame_rejects_non_mono() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        _frame(np.zeros((320, 2), dtype=np.float32))


def test_audio_frame_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="float32"):
        _frame(np.zeros(320, dtype=np.float64))


def test_audio_frame_rejects_non_finite() -> None:
    bad = np.zeros(4, dtype=np.float32)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _frame(bad)


def test_audio_frame_rejects_unnormalized_pcm() -> None:
    with pytest.raises(ValueError, match="normalized"):
        _frame(np.full(4, 1.5, dtype=np.float32))


def test_audio_frame_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        _frame(np.zeros(4, dtype=np.float32), rate=0)


def test_tts_chunk_validates_and_derives_duration() -> None:
    chunk = TtsChunk(
        samples=np.zeros(2400, dtype=np.float32),
        sample_rate=24000,
        segment_index=0,
        utterance_id="u1",
    )
    assert chunk.duration_ms == 100
    with pytest.raises(ValueError, match="normalized"):
        TtsChunk(
            samples=np.full(4, -2.0, dtype=np.float32),
            sample_rate=24000,
            segment_index=0,
            utterance_id="u1",
        )


def test_default_import_and_bench_pull_no_real_audio_dependency() -> None:
    """The default package import and ``bench`` never import soxr/sounddevice."""
    code = (
        "import sys, asyncio\n"
        "import interpret_live\n"
        "from interpret_live.bench import run_bench\n"
        "asyncio.run(run_bench())\n"
        "banned = {'soxr', 'sounddevice'}\n"
        "loaded = banned & set(sys.modules)\n"
        "assert not loaded, f'real audio deps leaked into the default path: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
