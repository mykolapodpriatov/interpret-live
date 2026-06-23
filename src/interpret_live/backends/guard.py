"""Import-guard helper for optional backend extras.

Heavy backends (faster-whisper, transformers/torch, Piper, cloud SDKs, audio I/O)
are optional. Each adapter calls :func:`require` for the module(s) it needs so a
missing dependency surfaces as a clear, actionable error::

    interpret_live.backends.guard.MissingExtraError:
        The 'whisper' backend requires the 'whisper' extra.
        Install it with: pip install 'interpret-live[whisper]'
        (missing module: faster_whisper)

instead of an obscure :class:`ImportError` deep inside an adapter.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["MissingExtraError", "require"]


class MissingExtraError(ImportError):
    """Raised when an optional backend's dependency is not installed.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    still catch it, while carrying a precise install hint.
    """

    def __init__(self, *, backend: str, extra: str, module: str) -> None:
        self.backend = backend
        self.extra = extra
        self.module = module
        super().__init__(
            f"The {backend!r} backend requires the {extra!r} extra.\n"
            f"Install it with: pip install 'interpret-live[{extra}]'\n"
            f"(missing module: {module})"
        )


def require(module: str, *, backend: str, extra: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingExtraError` with an install hint.

    Args:
        module: The importable module name the adapter depends on
            (e.g. ``"faster_whisper"``).
        backend: Human-readable backend name for the error (e.g. ``"whisper"``).
        extra: The pip extra that provides the dependency (e.g. ``"whisper"``).

    Returns:
        The imported module.

    Raises:
        MissingExtraError: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise MissingExtraError(backend=backend, extra=extra, module=module) from exc
