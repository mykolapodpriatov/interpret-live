"""Import-guard tests: a missing optional extra raises a clear install hint.

A missing dependency must surface as :class:`MissingExtraError` (a subclass of
:class:`ImportError`) carrying the exact ``pip install 'interpret-live[...]'``
hint — never an obscure raw ``ImportError`` from deep inside an adapter.
"""

from __future__ import annotations

import builtins

import pytest

from interpret_live.backends.guard import MissingExtraError, require


def test_require_returns_module_when_present() -> None:
    mod = require("json", backend="x", extra="x")
    assert mod.__name__ == "json"


def test_missing_module_raises_missing_extra_error_with_hint() -> None:
    with pytest.raises(MissingExtraError) as ei:
        require("definitely_not_installed_xyz", backend="whisper", extra="whisper")
    msg = str(ei.value)
    assert "whisper" in msg
    assert "pip install 'interpret-live[whisper]'" in msg
    assert "definitely_not_installed_xyz" in msg


def test_missing_extra_error_is_import_error() -> None:
    # So existing ``except ImportError`` handlers still catch it.
    assert issubclass(MissingExtraError, ImportError)


@pytest.mark.parametrize(
    ("module", "backend_cls_path", "extra"),
    [
        ("faster_whisper", "interpret_live.backends.whisper.WhisperSTT", "whisper"),
        ("transformers", "interpret_live.backends.nllb.NllbMT", "mt"),
        ("piper", "interpret_live.backends.piper.PiperTTS", "piper"),
        ("openai", "interpret_live.backends.realtime.RealtimeS2S", "openai"),
        ("google.genai", "interpret_live.backends.gemini.GeminiS2S", "gemini"),
        ("elevenlabs", "interpret_live.backends.elevenlabs.ElevenLabsTTS", "elevenlabs"),
        ("sounddevice", "interpret_live.audio_io.MicSource", "audio"),
    ],
)
def test_adapter_construction_without_extra_raises_clear_error(
    module: str,
    backend_cls_path: str,
    extra: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the optional module import to fail, simulating a missing extra.
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == module or name.startswith(module + "."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    import importlib

    mod_path, cls_name = backend_cls_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)

    with pytest.raises(MissingExtraError) as ei:
        # Minimal kwargs; the import guard fires in __init__ before real use.
        if cls_name == "WhisperSTT" or cls_name == "NllbMT":
            cls()
        elif cls_name == "PiperTTS":
            cls(model_path="x.onnx")
        elif cls_name == "RealtimeS2S" or cls_name == "GeminiS2S":
            cls(api_key="k")
        elif cls_name == "ElevenLabsTTS":
            cls(api_key="k", voice_id="v")
        elif cls_name == "MicSource":
            cls()
    assert extra in str(ei.value)
    assert f"interpret-live[{extra}]" in str(ei.value)
