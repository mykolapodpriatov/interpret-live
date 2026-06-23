"""Offline MT adapter backed by an NLLB/Madlad seq2seq model (``mt`` extra).

Import-guarded: constructing :class:`NllbMT` without the ``mt`` extra raises a
clear :class:`~interpret_live.backends.guard.MissingExtraError`. Translates one
CLOSED :class:`~interpret_live.types.Segment` at a time, prepending the rolling
source context so a stateless model keeps coherence.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..types import Segment
from .guard import require

__all__ = ["NllbMT"]

#: A small subset of NLLB BCP-47 → FLORES-200 language code mappings.
_FLORES: Mapping[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
}


class NllbMT:
    """Sentence-level MT over a Hugging Face seq2seq NLLB model.

    Args:
        source_lang: Source BCP-47 code (mapped to FLORES-200).
        target_lang: Target BCP-47 code (mapped to FLORES-200).
        model_name: Hugging Face model id.
        max_new_tokens: Generation cap per segment.
    """

    def __init__(
        self,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
        model_name: str = "facebook/nllb-200-distilled-600M",
        max_new_tokens: int = 256,
    ) -> None:
        transformers = require("transformers", backend="mt", extra="mt")
        self._src = _FLORES.get(source_lang, source_lang)
        self._tgt = _FLORES.get(target_lang, target_lang)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, src_lang=self._src)
        self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._max_new_tokens = max_new_tokens

    async def translate(self, segment: Segment, context: tuple[str, ...] = ()) -> str:
        """Translate a CLOSED ``segment`` with optional rolling source context."""
        if not segment.closed:
            raise AssertionError("NllbMT must only translate closed segments")
        text = (" ".join(context) + " " + segment.text).strip() if context else segment.text
        inputs = self._tokenizer(text, return_tensors="pt")
        forced = self._tokenizer.convert_tokens_to_ids(self._tgt)
        generated = self._model.generate(
            **inputs,
            forced_bos_token_id=forced,
            max_new_tokens=self._max_new_tokens,
        )
        out: str = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return out
