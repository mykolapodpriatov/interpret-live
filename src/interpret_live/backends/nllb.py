"""Offline MT adapter backed by an NLLB seq2seq model (``mt`` extra).

Import-guarded: constructing :class:`NllbMT` without the ``mt`` extra raises a
clear :class:`~interpret_live.backends.guard.MissingExtraError` (the spawned
worker shares this environment, so the parent-side import check is a valid
preflight).

Design (plan Task 3):

* Tokenization, ``generate()``, and decoding all run inside a dedicated
  long-lived spawned :class:`~interpret_live.model_worker.ModelWorker` — never
  on the event loop.
* **Only the current segment is translated.** The protocol's ``context``
  argument is accepted for compatibility but intentionally ignored: prepending
  rolling context to a stateless seq2seq model returns the translation of the
  *combined* input, and no context/output-alignment strategy has been proven
  that reliably strips the context back out — so already-spoken text would be
  repeated to the listener. Until such a strategy exists, non-repetition wins
  over cross-segment coherence.
* Cancellation is cooperative through a ``transformers`` ``StoppingCriteria``
  polling the worker's shared cancel event: a cancelled ``translate()``
  propagates cancellation without producing a value, and a stale result never
  reaches TTS. A worker that ignores cancellation is hard-reaped by
  :meth:`~interpret_live.model_worker.ModelWorker.aclose`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol

from ..model_worker import ModelWorker, raise_if_cancelled
from ..types import Segment
from .guard import require

__all__ = ["NllbMT", "build_nllb_handler"]

#: Supported NLLB BCP-47 -> FLORES-200 language code mappings.
_FLORES: Mapping[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
}

_VALID_DTYPES = {"auto", "float32", "float16", "bfloat16"}


class MtInputTooLongError(RuntimeError):
    """The segment exceeds ``max_input_tokens``; silent truncation is refused."""


def build_nllb_handler(
    *,
    model_name: str,
    src_lang: str,
    tgt_lang: str,
    device: str,
    dtype: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Any:
    """Child-process factory: construct tokenizer/model, return the handler.

    Runs inside the spawned worker; the heavy transformers/torch imports and
    model load never touch the event loop. ``from_pretrained`` already returns
    the model in inference mode.
    """
    import torch
    import transformers

    torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)

    class _CancelCriteria(transformers.StoppingCriteria):  # type: ignore[misc]
        def __init__(self, cancel_event: Any) -> None:
            self._cancel_event = cancel_event

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return bool(self._cancel_event.is_set())

    def handle(payload: dict[str, Any], cancel_event: Any) -> str:
        raise_if_cancelled(cancel_event)
        inputs = tokenizer(payload["text"], return_tensors="pt")
        n_tokens = int(inputs["input_ids"].shape[1])
        if n_tokens > max_input_tokens:
            raise MtInputTooLongError(
                f"segment tokenizes to {n_tokens} tokens > max_input_tokens="
                f"{max_input_tokens}; refusing to silently truncate the segment"
            )
        forced = tokenizer.convert_tokens_to_ids(tgt_lang)
        criteria = transformers.StoppingCriteriaList([_CancelCriteria(cancel_event)])
        with torch.no_grad():
            generated = model.generate(
                **{k: v.to(device) for k, v in inputs.items()},
                forced_bos_token_id=forced,
                max_new_tokens=max_new_tokens,
                stopping_criteria=criteria,
            )
        # Generation stopped early because of the cancel signal: the partial
        # output is a stale value and must never surface as a translation.
        raise_if_cancelled(cancel_event)
        out: str = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return out

    return handle


class MtWorker(Protocol):
    """The worker surface :class:`NllbMT` needs (injectable in tests)."""

    async def start(self) -> None:
        """Spawn/await model readiness."""
        ...

    async def request(self, payload: Any) -> tuple[str, Any]:
        """Send one translate request; return ``(status, value)``."""
        ...

    def signal_cancel(self) -> None:
        """Set the cooperative cancel signal."""
        ...

    def clear_cancel(self) -> None:
        """Re-arm the cancel signal."""
        ...

    async def aclose(self) -> None:
        """Shut the worker down within its bounded budget."""
        ...


class NllbMT:
    """Sentence-level MT over a Hugging Face seq2seq NLLB model.

    Args:
        source_lang: Source BCP-47 code (validated against FLORES-200).
        target_lang: Target BCP-47 code (validated against FLORES-200).
        model_name: Hugging Face model id or resolved local snapshot path.
        device: torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...).
        dtype: ``"auto"`` or an explicit torch dtype name.
        max_input_tokens: Hard input bound; longer segments raise a typed
            error instead of being silently truncated.
        max_new_tokens: Generation cap per segment.
        worker: Injectable translate worker (tests); defaults to a spawned
            :class:`~interpret_live.model_worker.ModelWorker` running
            :func:`build_nllb_handler`.
        ready_timeout_s: Model-load readiness budget for the default worker.
        grace_s: Per-stage shutdown budget for the default worker.
    """

    def __init__(
        self,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        dtype: str = "auto",
        max_input_tokens: int = 512,
        max_new_tokens: int = 256,
        worker: MtWorker | None = None,
        ready_timeout_s: float = 300.0,
        grace_s: float = 2.0,
    ) -> None:
        for role, code in (("source_lang", source_lang), ("target_lang", target_lang)):
            if code not in _FLORES:
                raise ValueError(
                    f"{role}={code!r} is not a supported NLLB language; "
                    f"supported codes: {', '.join(sorted(_FLORES))}"
                )
        if not model_name:
            raise ValueError("model_name must be a non-empty model id or local path")
        if dtype not in _VALID_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_VALID_DTYPES)}, got {dtype!r}")
        if not device or device.split(":")[0] not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"device must be cpu/cuda[:N]/mps, got {device!r}")
        if max_input_tokens < 1:
            raise ValueError(f"max_input_tokens must be >= 1, got {max_input_tokens}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        self._src = _FLORES[source_lang]
        self._tgt = _FLORES[target_lang]
        if worker is None:
            # The spawned child shares this interpreter's environment, so a
            # parent-side import check is a valid (and fail-fast) preflight.
            require("transformers", backend="mt", extra="mt")
            worker = ModelWorker(
                "interpret_live.backends.nllb:build_nllb_handler",
                {
                    "model_name": model_name,
                    "src_lang": self._src,
                    "tgt_lang": self._tgt,
                    "device": device,
                    "dtype": dtype,
                    "max_input_tokens": max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                },
                name="nllb-worker",
                ready_timeout_s=ready_timeout_s,
                grace_s=grace_s,
            )
        self._worker: MtWorker = worker
        self._closed = False

    async def start(self) -> None:
        """Start the model worker and wait for readiness (preflight)."""
        await self._worker.start()

    async def aclose(self) -> None:
        """Cancel in-flight generation and reap the worker (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._worker.signal_cancel()
        await self._worker.aclose()

    async def translate(self, segment: Segment, context: tuple[str, ...] = ()) -> str:
        """Translate exactly ``segment.text`` (CLOSED segments only).

        ``context`` is accepted for protocol compatibility but intentionally
        ignored (see the module docstring): translating context + segment
        returns the combined translation, and re-speaking already-delivered
        context is worse than losing cross-segment coherence.
        """
        if not segment.closed:
            raise AssertionError("NllbMT must only translate closed segments")
        try:
            status, value = await self._worker.request({"text": segment.text})
        except asyncio.CancelledError:
            # Barge-in cancelled us mid-generation: stop the worker's current
            # generate() cooperatively; its stale result is already discarded.
            self._worker.signal_cancel()
            raise
        if status == "cancelled":
            raise asyncio.CancelledError("translation was cancelled mid-generation")
        if status == "error":
            if "MtInputTooLongError" in str(value):
                raise MtInputTooLongError(str(value))
            raise RuntimeError(f"NLLB translation failed: {value}")
        return str(value)
