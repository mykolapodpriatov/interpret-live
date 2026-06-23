"""Static guard: core + fake modules must not call ``asyncio.sleep`` for delay.

Per the determinism strategy, every inter-partial / latency / debounce delay in
the core and fakes must go through the injected ``clock.sleep()``. A real
``asyncio.sleep(ms)`` with a non-zero delay would silently break determinism and
make the suite take real wall-clock time.

``asyncio.sleep(0)`` (a pure cooperative yield, zero real time) is permitted —
it is how the manual-clock drain pump and the mic broadcaster relinquish control.
The real-audio edges (``MicSource``/``SpeakerSink``) and the ``RealClock`` are
production-only and exempt.
"""

from __future__ import annotations

import ast
from pathlib import Path

import interpret_live

_PACKAGE_ROOT = Path(interpret_live.__file__).parent

# Modules that legitimately touch the event loop's real timing (production paths)
# or define the real clock; excluded from the no-real-sleep scan.
_EXEMPT = {
    "clock.py",  # RealClock.sleep delegates to asyncio.sleep (production)
    "audio_io.py",  # MicSource/SpeakerSink are real-audio edges (production)
}

_SCANNED = [
    "stabilize.py",
    "segment.py",
    "vad.py",
    "pipeline.py",
    "s2s.py",
    "session.py",
    "metrics.py",
    "bench.py",
    "backends/fake.py",
]


def _nonzero_asyncio_sleep_calls(source: str) -> list[int]:
    """Return line numbers of ``asyncio.sleep(<non-zero>)`` calls in ``source``."""
    tree = ast.parse(source)
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_asyncio_sleep = (
            isinstance(func, ast.Attribute)
            and func.attr == "sleep"
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
        )
        if not is_asyncio_sleep:
            continue
        # Permit only a literal zero argument (a pure yield).
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
            continue
        offenders.append(node.lineno)
    return offenders


def test_core_and_fakes_have_no_real_asyncio_sleep() -> None:
    problems: dict[str, list[int]] = {}
    for rel in _SCANNED:
        path = _PACKAGE_ROOT / rel
        offenders = _nonzero_asyncio_sleep_calls(path.read_text())
        if offenders:
            problems[rel] = offenders
    assert not problems, (
        f"non-zero asyncio.sleep() found in core/fake modules (use clock.sleep): {problems}"
    )


def test_exempt_set_is_accurate() -> None:
    # Guard against accidentally scanning an exempt file or vice versa.
    assert _EXEMPT.isdisjoint(_SCANNED)
