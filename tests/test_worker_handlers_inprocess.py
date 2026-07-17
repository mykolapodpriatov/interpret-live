"""In-process coverage for the pure worker request handlers + child main loop.

The spawned-worker *lifecycle* is covered by real-``spawn`` tests in
``test_model_worker``; here the handler factories run in-process against stub
model modules injected into ``sys.modules``, so their request/branch logic is
measured deterministically instead of hiding inside child processes.
"""

from __future__ import annotations

import queue
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest

from interpret_live.audio_codec import float32_to_pcm16
from interpret_live.model_worker import _child_main


class _Cancel:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False


# ----- whisper handler ----------------------------------------------------------


def _install_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch, words: list[Any]) -> None:
    mod = types.ModuleType("faster_whisper")

    class _Segments:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> Any:
            seg = types.SimpleNamespace(words=words)
            return iter([seg])

        def close(self) -> None:
            self.closed = True

    class WhisperModel:
        def __init__(self, model: str, device: str = "cpu", compute_type: str = "int8") -> None:
            self.model = model

        def transcribe(self, audio: Any, language: str, word_timestamps: bool) -> Any:
            return _Segments(), {"language": language}

    mod.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


def test_whisper_handler_converts_words_and_honors_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interpret_live.backends.whisper import build_whisper_handler

    words = [
        types.SimpleNamespace(word=" hello", start=0.0, end=0.5),
        types.SimpleNamespace(word="world ", start=0.5, end=1.0),
        types.SimpleNamespace(word="  ", start=1.0, end=1.1),  # empty after strip
    ]
    _install_fake_faster_whisper(monkeypatch, words)
    handler = build_whisper_handler(
        model_source="small", language="en", device="cpu", compute_type="int8"
    )
    pcm = np.zeros(1600, dtype=np.float32).tobytes()
    tokens = handler({"pcm": pcm, "final": True, "turn": "turn-1"}, _Cancel())
    assert tokens == [("hello", 0, 500), ("world", 500, 1000)]

    cancel = _Cancel()
    cancel.set()
    from interpret_live.model_worker import WorkerCancelledError

    with pytest.raises(WorkerCancelledError):
        handler({"pcm": pcm, "final": False, "turn": "turn-1"}, cancel)


# ----- nllb handler --------------------------------------------------------------


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch, *, n_input_tokens: int = 3
) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    torch_mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    torch_mod.no_grad = _NoGrad  # type: ignore[attr-defined]
    torch_mod.float32 = "float32"  # type: ignore[attr-defined]

    tf_mod = types.ModuleType("transformers")

    class _Tensor:
        def __init__(self, shape: tuple[int, int]) -> None:
            self.shape = shape

        def to(self, device: str) -> _Tensor:
            return self

    class _Tokenizer:
        def __call__(self, text: str, return_tensors: str) -> dict[str, _Tensor]:
            calls["text"] = text
            return {"input_ids": _Tensor((1, n_input_tokens))}

        def convert_tokens_to_ids(self, token: str) -> int:
            return 7

        def batch_decode(self, generated: Any, skip_special_tokens: bool) -> list[str]:
            return [f"<{calls['text']}>"]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, src_lang: str) -> _Tokenizer:
            calls["src_lang"] = src_lang
            return _Tokenizer()

    class _Model:
        def to(self, device: str) -> None:
            calls["device"] = device

        def generate(self, **kwargs: Any) -> list[int]:
            calls["generate"] = kwargs
            criteria = kwargs["stopping_criteria"]
            # Model checks the stopping criteria between steps.
            calls["stopped_early"] = bool(criteria[0](None, None))
            return [1, 2, 3]

    class AutoModelForSeq2SeqLM:
        @staticmethod
        def from_pretrained(model: str, torch_dtype: Any = None) -> _Model:
            return _Model()

    class StoppingCriteria:
        pass

    class StoppingCriteriaList(list):
        pass

    tf_mod.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    tf_mod.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM  # type: ignore[attr-defined]
    tf_mod.StoppingCriteria = StoppingCriteria  # type: ignore[attr-defined]
    tf_mod.StoppingCriteriaList = StoppingCriteriaList  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", tf_mod)
    return calls


def test_nllb_handler_translates_and_bounds_input(monkeypatch: pytest.MonkeyPatch) -> None:
    from interpret_live.backends.nllb import MtInputTooLongError, build_nllb_handler

    calls = _install_fake_transformers(monkeypatch, n_input_tokens=3)
    handler = build_nllb_handler(
        model_name="m",
        src_lang="eng_Latn",
        tgt_lang="spa_Latn",
        device="cpu",
        dtype="auto",
        max_input_tokens=8,
        max_new_tokens=32,
    )
    out = handler({"text": "hola mundo."}, _Cancel())
    assert out == "<hola mundo.>"
    assert calls["generate"]["forced_bos_token_id"] == 7
    assert calls["generate"]["max_new_tokens"] == 32
    assert calls["stopped_early"] is False

    # Over the input bound: typed refusal instead of silent truncation.
    calls2 = _install_fake_transformers(monkeypatch, n_input_tokens=99)
    del calls2
    handler2 = build_nllb_handler(
        model_name="m",
        src_lang="eng_Latn",
        tgt_lang="spa_Latn",
        device="cpu",
        dtype="float32",
        max_input_tokens=8,
        max_new_tokens=32,
    )
    with pytest.raises(MtInputTooLongError):
        handler2({"text": "way too long"}, _Cancel())


def test_nllb_handler_cancel_mid_generation_never_returns_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from interpret_live.backends.nllb import build_nllb_handler
    from interpret_live.model_worker import WorkerCancelledError

    _install_fake_transformers(monkeypatch)
    handler = build_nllb_handler(
        model_name="m",
        src_lang="eng_Latn",
        tgt_lang="spa_Latn",
        device="cpu",
        dtype="auto",
        max_input_tokens=8,
        max_new_tokens=32,
    )
    cancel = _Cancel()
    cancel.set()  # set before the call: caught at entry
    with pytest.raises(WorkerCancelledError):
        handler({"text": "x"}, cancel)


# ----- piper handler --------------------------------------------------------------


def _install_fake_piper(monkeypatch: pytest.MonkeyPatch, blocks: list[bytes]) -> dict[str, Any]:
    state: dict[str, Any] = {"closed_generators": 0}
    mod = types.ModuleType("piper")

    class _Voice:
        config = types.SimpleNamespace(sample_rate=22050)

        def synthesize_stream_raw(self, text: str) -> Any:
            state["text"] = text

            def gen() -> Any:
                try:
                    yield from blocks
                finally:
                    state["closed_generators"] += 1

            return gen()

    class PiperVoice:
        @staticmethod
        def load(model_path: str, config_path: str | None = None) -> _Voice:
            return _Voice()

    mod.PiperVoice = PiperVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", mod)
    return state


def test_piper_handler_state_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from interpret_live.backends.piper import build_piper_handler

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    blocks = [float32_to_pcm16(np.full(10, 0.1, dtype=np.float32)) for _ in range(2)]
    state = _install_fake_piper(monkeypatch, blocks)
    handler = build_piper_handler(model_path=str(model), config_path=None)

    cancel = _Cancel()
    assert handler({"op": "start", "text": "hola."}, cancel) == {"rate": 22050}
    first = handler({"op": "next"}, cancel)
    assert first["end"] is False and first["pcm"] == blocks[0]
    # A new utterance closes the stale generator in the child.
    handler({"op": "start", "text": "otra."}, cancel)
    assert state["closed_generators"] == 1
    handler({"op": "next"}, cancel)
    handler({"op": "next"}, cancel)
    done = handler({"op": "next"}, cancel)
    assert done == {"end": True}
    # Explicit stop is idempotent and closes any open generator.
    handler({"op": "start", "text": "tres."}, cancel)
    handler({"op": "stop"}, cancel)
    assert handler({"op": "next"}, cancel) == {"end": True}
    with pytest.raises(ValueError, match="unknown Piper worker op"):
        handler({"op": "bogus"}, cancel)


def test_piper_handler_missing_files_raise_typed(tmp_path: Any) -> None:
    from interpret_live.backends.piper import TtsVoiceError, build_piper_handler

    with pytest.raises(TtsVoiceError, match="not found"):
        build_piper_handler(model_path=str(tmp_path / "nope.onnx"), config_path=None)
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    with pytest.raises(TtsVoiceError, match="config not found"):
        build_piper_handler(model_path=str(model), config_path=str(tmp_path / "nope.json"))


# ----- the child main loop, in-process ---------------------------------------------


def test_child_main_loop_processes_requests_in_process() -> None:
    req_q: queue.Queue[Any] = queue.Queue()
    res_q: queue.Queue[Any] = queue.Queue()
    cancel = threading.Event()

    req_q.put((1, "ping"))
    req_q.put((2, "boom"))
    req_q.put(None)  # cooperative shutdown sentinel

    _child_main(
        "worker_handlers:echo_or_error_factory",
        {},
        req_q,
        res_q,
        cancel,
    )

    assert res_q.get_nowait() == ("ready", None, None)
    assert res_q.get_nowait() == ("ok", 1, "pong:ping")
    kind, req_id, err = res_q.get_nowait()
    assert (kind, req_id) == ("error", 2)
    assert "RuntimeError" in err
    assert res_q.get_nowait() == ("closed", None, None)


def test_child_main_reports_fatal_startup() -> None:
    req_q: queue.Queue[Any] = queue.Queue()
    res_q: queue.Queue[Any] = queue.Queue()
    _child_main("worker_handlers:failing_startup_factory", {}, req_q, res_q, threading.Event())
    kind, _rid, message = res_q.get_nowait()
    assert kind == "fatal"
    assert "boom at load" in message
