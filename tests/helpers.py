"""Deterministic builders shared by the test modules.

Imported as a bare top-level module (``from helpers import ...``) because pytest
prepends each test directory to ``sys.path``; this avoids depending on a
``tests`` package name, which can collide with other projects on the path.
"""

from __future__ import annotations

import numpy as np

from interpret_live.types import AudioFrame, Hypothesis, Token


def make_tokens(words: list[str], *, step_ms: int = 100) -> tuple[Token, ...]:
    """Build back-to-back word tokens, each ``step_ms`` long, starting at 0."""
    return tuple(
        Token(text=w, start_ms=i * step_ms, end_ms=(i + 1) * step_ms) for i, w in enumerate(words)
    )


def hyp(*words: str, is_final: bool = False) -> Hypothesis:
    """Build a :class:`Hypothesis` from bare word strings."""
    return Hypothesis(tokens=make_tokens(list(words)), is_final=is_final)


def frame(amplitude: float, *, t_ms: int = 0, n: int = 320, sample_rate: int = 16000) -> AudioFrame:
    """Build a constant-amplitude :class:`AudioFrame`."""
    return AudioFrame(
        samples=np.full(n, amplitude, dtype=np.float32),
        sample_rate=sample_rate,
        t_ms=t_ms,
    )
