"""Segmentation & incremental-MT policy — translate stable units, never mid-word.

The committed token stream is split into :class:`~interpret_live.types.Segment`
objects at **sentence boundaries** — the pinned terminal-punctuation set ``.``
``!`` ``?`` (ASCII) for M1–M2 — or a ``max_segment_tokens`` cap to bound latency.

* **MT translates only CLOSED segments** (a completed sentence or a max-token
  flush) — never a partial clause — because translating a partial clause yields
  wrong word order.
* **"Incremental" means per-closed-segment, not per-token:** simultaneity comes
  from translating + speaking each segment *as soon as it closes*, before the
  whole utterance ends. A long multi-clause utterance therefore yields several
  MT→TTS bursts during the utterance.
* **Rolling context (bounded):** :meth:`Segmenter.context_for` returns the last
  ``context_tokens`` of prior committed *source* text, truncated at an ASR
  ``Token`` boundary (never mid-word). ``context_tokens`` counts word-level
  ``Token`` objects, not MT subword pieces.
"""

from __future__ import annotations

from .types import Segment, Token

__all__ = ["TERMINAL_PUNCT", "Segmenter", "ends_segment"]

#: Pinned terminal-punctuation set for M1–M2 (ASCII sentence terminators).
#: CJK ``。！？`` and clause marks (``;`` / ``:``) are a documented future.
TERMINAL_PUNCT: frozenset[str] = frozenset({".", "!", "?"})


def ends_segment(token: Token) -> bool:
    """Return ``True`` if ``token`` ends with a terminal punctuation mark.

    A token like ``"end."`` or ``"really?"`` closes a segment; trailing quotes
    or brackets after the terminator (e.g. ``'done."'``) still count.
    """
    text = token.text.rstrip("\"')]}»”’")
    return bool(text) and text[-1] in TERMINAL_PUNCT


class Segmenter:
    """Accumulate committed tokens and emit closed, translatable segments.

    A segment closes when a token ends in terminal punctuation **or** when the
    open segment reaches ``max_segment_tokens`` (a forced flush so latency stays
    bounded). Only closed segments are returned by :meth:`feed`; the trailing
    open buffer is held until it closes.

    Args:
        max_segment_tokens: Hard cap on tokens in an open segment before a forced
            flush. Must be ``>= 1``.
        context_tokens: How many prior committed *source* tokens to expose via
            :meth:`context_for` as rolling MT context. Must be ``>= 0``.
    """

    __slots__ = (
        "_committed_history",
        "_consumed",
        "_context_tokens",
        "_max_tokens",
        "_next_index",
        "_open",
    )

    def __init__(self, max_segment_tokens: int = 24, context_tokens: int = 50) -> None:
        if max_segment_tokens < 1:
            raise ValueError(f"max_segment_tokens must be >= 1, got {max_segment_tokens}")
        if context_tokens < 0:
            raise ValueError(f"context_tokens must be >= 0, got {context_tokens}")
        self._max_tokens = max_segment_tokens
        self._context_tokens = context_tokens
        self._open: list[Token] = []
        # All tokens fed so far, in order, for rolling-context lookups.
        self._committed_history: list[Token] = []
        self._next_index = 0
        # Absolute index of the first token currently in ``_open``.
        self._consumed = 0

    @property
    def pending(self) -> tuple[Token, ...]:
        """Tokens currently buffered in the open (not-yet-closed) segment."""
        return tuple(self._open)

    def _make_segment(self, *, closed: bool) -> Segment:
        tokens = tuple(self._open)
        start = self._consumed
        end = start + len(tokens)
        seg = Segment(
            text=" ".join(t.text for t in tokens),
            tokens=tokens,
            token_span=(start, end),
            closed=closed,
            index=self._next_index,
        )
        self._next_index += 1
        self._consumed = end
        self._open = []
        return seg

    def feed(self, tokens: tuple[Token, ...]) -> list[Segment]:
        """Add newly committed tokens; return any segments that closed.

        Tokens are appended to the open segment in order. Each token that ends a
        sentence closes the current segment; if the open segment grows to
        ``max_segment_tokens`` it is force-flushed as a closed segment.
        """
        closed: list[Segment] = []
        for tok in tokens:
            self._committed_history.append(tok)
            self._open.append(tok)
            if ends_segment(tok) or len(self._open) >= self._max_tokens:
                closed.append(self._make_segment(closed=True))
        return closed

    def flush(self) -> Segment | None:
        """Close and return the open segment at end-of-utterance, if non-empty.

        Used when a final hypothesis force-commits a tail that did not end in
        terminal punctuation, so the trailing clause is still translated.
        """
        if not self._open:
            return None
        return self._make_segment(closed=True)

    def context_for(self, segment: Segment) -> tuple[Token, ...]:
        """Return up to ``context_tokens`` prior source tokens before ``segment``.

        Truncated at an ASR ``Token`` boundary (whole words only) and never
        includes the segment's own tokens — pure left-context for a stateless MT.
        """
        if self._context_tokens == 0:
            return ()
        start_index = segment.token_span[0]
        lo = max(0, start_index - self._context_tokens)
        return tuple(self._committed_history[lo:start_index])

    def reset(self) -> None:
        """Reset all buffers and counters for a fresh utterance."""
        self._open = []
        self._committed_history = []
        self._next_index = 0
        self._consumed = 0
