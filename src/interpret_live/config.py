"""Runtime configuration models (pydantic) for the pipeline and session.

Centralises the tunable knobs — the LocalAgreement window, segmentation caps,
VAD/barge-in thresholds, queue bounds — with validation, so the CLI and library
share one validated config object. Defaults encode the stability/latency
tradeoff documented in the plan.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = ["AudioConfig", "BackendKind", "Direction", "PipelineConfig", "SessionConfig"]


class BackendKind(StrEnum):
    """Which backend family a session uses."""

    OFFLINE = "offline"  # pipeline path: STT + MT + TTS
    CLOUD = "cloud"  # unified S2S path
    FAKE = "fake"  # deterministic fakes (tests / bench)


class Direction(StrEnum):
    """Translation direction for a single session."""

    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"


class PipelineConfig(BaseModel):
    """Tunable parameters for the streaming pipeline.

    Attributes:
        agreement_n: LocalAgreement window size (higher = more stable, more
            latent).
        max_segment_tokens: Forced-flush cap for an open segment.
        context_tokens: Rolling source-context size (ASR tokens) for MT.
        vad_threshold: RMS speech threshold for the energy VAD.
        vad_hangover_ms: Trailing silence before the VAD flips to silence.
        barge_in_onset_ms: Continuous speech required to confirm a barge-in onset.
        queue_maxsize: Bound on every internal pipeline queue (backpressure).
    """

    agreement_n: int = Field(default=2, ge=1)
    max_segment_tokens: int = Field(default=24, ge=1)
    context_tokens: int = Field(default=50, ge=0)
    vad_threshold: float = Field(default=0.02, ge=0.0)
    vad_hangover_ms: int = Field(default=200, ge=0)
    barge_in_onset_ms: int = Field(default=150, ge=0)
    queue_maxsize: int = Field(default=8, ge=1)


class AudioConfig(BaseModel):
    """Validated real-audio device/edge configuration.

    Attributes:
        input_device: Input (microphone) device index; ``None`` = default.
        output_device: Output (speaker) device index; ``None`` = default.
        capture_rate: Microphone capture rate in Hz.
        playback_rate: Output device rate in Hz; ``None`` = device default.
        frame_ms: Capture frame duration in milliseconds.
        mic_queue_frames: Bounded mic frame queue before drop-oldest applies.
        playback_capacity: Maximum outstanding (scheduled) playback chunks.
        playback_ring_ms: Speaker ring-buffer length in milliseconds.
        playback_timeout_ms: Bound on individual playback operations.
        shutdown_timeout_ms: Grace budget for closing devices/workers.
    """

    input_device: int | None = None
    output_device: int | None = None
    capture_rate: int = Field(default=16000, gt=0)
    playback_rate: int | None = Field(default=None, gt=0)
    frame_ms: int = Field(default=20, gt=0)
    mic_queue_frames: int = Field(default=32, ge=1)
    playback_capacity: int = Field(default=8, ge=1)
    playback_ring_ms: int = Field(default=1000, gt=0)
    playback_timeout_ms: int = Field(default=5000, ge=0)
    shutdown_timeout_ms: int = Field(default=5000, ge=0)


class SessionConfig(BaseModel):
    """Top-level session configuration.

    Attributes:
        source_lang: BCP-47-ish source language code (e.g. ``"en"``).
        target_lang: Target language code (e.g. ``"es"``).
        backend: Which backend family to use.
        pipeline: Nested :class:`PipelineConfig`.
        audio: Nested :class:`AudioConfig` (real device edges).
    """

    source_lang: str = "en"
    target_lang: str = "es"
    backend: BackendKind = BackendKind.FAKE
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
