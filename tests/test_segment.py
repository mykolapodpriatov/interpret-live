"""Segmentation tests: terminal-punctuation close, max-token flush, closed-only.

Covers segment closing on ``.`` / ``!`` / ``?``, the ``max_segment_tokens``
forced flush, the guarantee that only CLOSED segments are produced (so MT never
sees a partial clause), and the rolling ASR-token context.
"""

from __future__ import annotations

import pytest

from helpers import make_tokens
from interpret_live.segment import Segmenter, ends_segment
from interpret_live.types import Token


def _toks(*words: str) -> tuple[Token, ...]:
    return make_tokens(list(words))


# ----- Terminal-punctuation close ---------------------------------------------


def test_closes_on_period() -> None:
    seg = Segmenter(max_segment_tokens=10)
    out = seg.feed(_toks("hello", "world."))
    assert len(out) == 1
    assert out[0].closed is True
    assert out[0].text == "hello world."


@pytest.mark.parametrize("terminator", [".", "!", "?"])
def test_closes_on_each_terminal_mark(terminator: str) -> None:
    seg = Segmenter(max_segment_tokens=10)
    out = seg.feed(_toks("go", f"now{terminator}"))
    assert len(out) == 1
    assert out[0].text == f"go now{terminator}"


def test_does_not_close_without_terminal_punctuation() -> None:
    seg = Segmenter(max_segment_tokens=10)
    out = seg.feed(_toks("still", "going"))
    assert out == []
    assert [t.text for t in seg.pending] == ["still", "going"]


@pytest.mark.parametrize("terminator", ["。", "！", "？"])
def test_closes_on_each_cjk_terminal_mark(terminator: str) -> None:
    seg = Segmenter(max_segment_tokens=10)
    out = seg.feed(_toks("これは", f"テスト{terminator}"))
    assert len(out) == 1
    assert out[0].text == f"これは テスト{terminator}"


def test_cjk_terminator_alone_ends_segment() -> None:
    assert ends_segment(Token("終わり。", 0, 1)) is True
    assert ends_segment(Token("元気ですか？", 0, 1)) is True
    assert ends_segment(Token("すごい！", 0, 1)) is True
    # A CJK token without a terminal mark stays open.
    assert ends_segment(Token("こんにちは", 0, 1)) is False


def test_terminator_with_trailing_quote_still_closes() -> None:
    assert ends_segment(Token('done."', 0, 1)) is True
    assert ends_segment(Token("word", 0, 1)) is False
    # CJK terminator followed by a CJK closing corner bracket still closes.
    assert ends_segment(Token("終わり。」", 0, 1)) is True
    assert ends_segment(Token("「はい」", 0, 1)) is False


def test_multiple_sentences_in_one_feed_produce_multiple_segments() -> None:
    seg = Segmenter(max_segment_tokens=20)
    out = seg.feed(_toks("one.", "two.", "three."))
    assert [s.text for s in out] == ["one.", "two.", "three."]
    assert all(s.closed for s in out)
    assert [s.index for s in out] == [0, 1, 2]


# ----- Max-token forced flush -------------------------------------------------


def test_max_segment_tokens_forces_flush() -> None:
    seg = Segmenter(max_segment_tokens=3)
    out = seg.feed(_toks("a", "b", "c"))  # no terminal punctuation
    assert len(out) == 1
    assert out[0].closed is True  # forced flush is still a closed segment
    assert out[0].text == "a b c"


def test_max_token_flush_then_continues_new_segment() -> None:
    seg = Segmenter(max_segment_tokens=2)
    out = seg.feed(_toks("a", "b", "c"))
    # First two flush at the cap; "c" remains open.
    assert [s.text for s in out] == ["a b"]
    assert [t.text for t in seg.pending] == ["c"]


# ----- Only-closed-to-MT invariant --------------------------------------------


def test_only_closed_segments_are_emitted() -> None:
    seg = Segmenter(max_segment_tokens=10)
    # Feed a partial clause: nothing emitted (it would be a partial-clause MT).
    assert seg.feed(_toks("the", "quick", "brown")) == []
    # Close it: now a single closed segment emerges.
    out = seg.feed(_toks("fox."))
    assert len(out) == 1
    assert out[0].closed is True
    assert out[0].text == "the quick brown fox."


def test_flush_closes_trailing_open_segment_at_end_of_utterance() -> None:
    seg = Segmenter(max_segment_tokens=10)
    seg.feed(_toks("no", "terminator", "here"))
    tail = seg.flush()
    assert tail is not None
    assert tail.closed is True
    assert tail.text == "no terminator here"
    # A second flush with nothing open returns None.
    assert seg.flush() is None


# ----- Rolling ASR-token context ----------------------------------------------


def test_rolling_context_returns_prior_source_tokens() -> None:
    seg = Segmenter(max_segment_tokens=10, context_tokens=50)
    seg.feed(_toks("first", "sentence."))
    out = seg.feed(_toks("second", "sentence."))
    ctx = seg.context_for(out[0])
    assert [t.text for t in ctx] == ["first", "sentence."]


def test_rolling_context_truncates_at_token_boundary_and_bounds_size() -> None:
    seg = Segmenter(max_segment_tokens=100, context_tokens=2)
    seg.feed(_toks("w1", "w2", "w3", "w4."))
    out = seg.feed(_toks("target."))
    ctx = seg.context_for(out[0])
    # Only the last 2 prior ASR tokens, whole words.
    assert [t.text for t in ctx] == ["w3", "w4."]


def test_rolling_context_excludes_segments_own_tokens() -> None:
    seg = Segmenter(max_segment_tokens=10, context_tokens=50)
    out = seg.feed(_toks("only", "sentence."))
    # No prior context for the very first segment.
    assert seg.context_for(out[0]) == ()


def test_context_tokens_zero_returns_empty() -> None:
    seg = Segmenter(max_segment_tokens=10, context_tokens=0)
    seg.feed(_toks("a", "b."))
    out = seg.feed(_toks("c", "d."))
    assert seg.context_for(out[0]) == ()


# ----- Token spans + reset ----------------------------------------------------


def test_token_spans_are_contiguous_and_half_open() -> None:
    seg = Segmenter(max_segment_tokens=10)
    out1 = seg.feed(_toks("a", "b."))
    out2 = seg.feed(_toks("c", "d", "e."))
    assert out1[0].token_span == (0, 2)
    assert out2[0].token_span == (2, 5)


def test_reset_clears_history_and_indices() -> None:
    seg = Segmenter(max_segment_tokens=10, context_tokens=50)
    seg.feed(_toks("a", "b."))
    seg.reset()
    out = seg.feed(_toks("c", "d."))
    assert out[0].index == 0
    assert out[0].token_span == (0, 2)
    assert seg.context_for(out[0]) == ()


# ----- Construction guards ----------------------------------------------------


def test_invalid_max_segment_tokens_raises() -> None:
    with pytest.raises(ValueError, match="max_segment_tokens must be >= 1"):
        Segmenter(max_segment_tokens=0)


def test_invalid_context_tokens_raises() -> None:
    with pytest.raises(ValueError, match="context_tokens must be >= 0"):
        Segmenter(context_tokens=-1)
