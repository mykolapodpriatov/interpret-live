"""interpret_live — real-time, audio-stage-stabilized simultaneous interpretation.

A streaming speech-to-speech translator: live STT → LocalAgreement
stabilization → incremental MT → streaming TTS that begins before the sentence
ends, with barge-in/VAD interrupt handling, over one interface that targets both
a cloud-realtime (unified S2S) backend and a fully-offline (faster-whisper +
local MT + Piper) pipeline backend.

The default install is light and offline-testable: the streaming-orchestration
core (stabilizer, segmentation, VAD/barge-in, pipeline) plus deterministic fakes
need only stdlib + numpy + pydantic; the heavy ML/audio/cloud backends are
optional extras behind import-guarded adapters.
"""

from __future__ import annotations

from .clock import Clock, ManualClock, RealClock
from .config import BackendKind, Direction, PipelineConfig, SessionConfig
from .metrics import MetricsLog, MetricsReport, UtteranceMetrics
from .segment import Segmenter
from .session import (
    Backend,
    Capabilities,
    CapabilityError,
    DualChannel,
    PipelineBackend,
    PipelineSession,
    S2SBackend,
    S2SSession,
    Session,
    SessionProto,
)
from .stabilize import LocalAgreementStabilizer, normalize_token
from .types import (
    AudioFrame,
    AudioSink,
    AudioSource,
    CommitResult,
    Hypothesis,
    MetricEvent,
    Segment,
    Token,
    TtsChunk,
)
from .vad import BargeInDetector, EnergyVAD

__version__ = "0.1.0"

__all__ = [
    "AudioFrame",
    "AudioSink",
    "AudioSource",
    "Backend",
    "BackendKind",
    "BargeInDetector",
    "Capabilities",
    "CapabilityError",
    "Clock",
    "CommitResult",
    "Direction",
    "DualChannel",
    "EnergyVAD",
    "Hypothesis",
    "LocalAgreementStabilizer",
    "ManualClock",
    "MetricEvent",
    "MetricsLog",
    "MetricsReport",
    "PipelineBackend",
    "PipelineConfig",
    "PipelineSession",
    "RealClock",
    "S2SBackend",
    "S2SSession",
    "Segment",
    "Segmenter",
    "Session",
    "SessionConfig",
    "SessionProto",
    "Token",
    "TtsChunk",
    "UtteranceMetrics",
    "__version__",
    "normalize_token",
]
