"""LocalAgreement-*n* partial-hypothesis stabilizer — the algorithmic heart.

A streaming ASR keeps revising its partial transcript. Feeding those raw
partials straight into MT→TTS makes the synthesized speech stutter as words are
rewritten. The :class:`LocalAgreementStabilizer` commits only the prefix that
has **agreed across the last *n* partial hypotheses**, so only *stable* text
flows downstream and the audio never has to retract.

Precise commit rule (so two implementers write the same thing)::

    c = committed_end                        # tokens already committed
    tails = [normalized(h.tokens[c:]) for h in last_n_hypotheses]
    L = len( LCP(tails) )                     # prefix shared by ALL n tails
    committed_end = max(c, c + L)             # monotonic; never retracts

* ``LCP`` is the longest prefix shared by **all** *n* tails simultaneously — the
  count stops at the first index where *any* two tails disagree (not a
  pairwise-rolling comparison).
* The LCP is computed over the **uncommitted tail only** (tokens after the
  current committed prefix).
* A hypothesis **shorter than ``c``** contributes an **empty tail**, so ``L``
  becomes 0 that step and nothing un-commits.
* Because we only ever assign ``max(c, …)`` and extend, the committed prefix is
  **monotonic — never retracted**. A later hypothesis that disagrees with an
  already-committed token does *not* un-commit; instead the
  ``post_commit_disagreement`` counter is incremented so the user can tune *n*.

Normalization (pinned)::

    norm(t) = t.lower().strip(string.punctuation)

so ``"Hello,"`` ≡ ``"hello"`` and ``"Mr."`` ≡ ``"Mr"``, but ``"don't"`` stays
``"don't"`` (distinct from ``"dont"``) because only *leading/trailing*
punctuation is stripped.

On ``is_final`` the remaining tail is force-committed (the latency floor) and the
window + ``committed_end`` are reset before the next utterance, so a new
utterance's LCP is never computed against stale tokens.
"""

from __future__ import annotations

import string
from collections import deque

from .types import CommitResult, Hypothesis, Token

__all__ = ["LocalAgreementStabilizer", "normalize_token"]

_PUNCT = string.punctuation


def normalize_token(text: str) -> str:
    """Pinned token-equality normalizer: ``text.lower().strip(punctuation)``.

    Strips only *leading and trailing* ASCII punctuation and lowercases, so
    internal punctuation is preserved. Two tokens *agree* iff their normalized
    forms are equal.

    Examples:
        >>> normalize_token("Hello,")
        'hello'
        >>> normalize_token("Mr.")
        'mr'
        >>> normalize_token("don't")
        "don't"
    """
    return text.lower().strip(_PUNCT)


def _lcp_length(tails: list[list[Token]]) -> int:
    """Length of the longest common prefix shared by **all** token tails.

    Agreement uses :func:`normalize_token`. The count stops at the first index
    where any two tails disagree (or any tail ends). An empty tail forces 0.
    """
    if not tails:
        return 0
    shortest = min(len(t) for t in tails)
    if shortest == 0:
        return 0
    first = tails[0]
    for i in range(shortest):
        ref = normalize_token(first[i].text)
        for other in tails[1:]:
            if normalize_token(other[i].text) != ref:
                return i
    return shortest


class LocalAgreementStabilizer:
    """Commit the ASR prefix agreed across the last *n* partial hypotheses.

    Args:
        n: Window size — a token is committed only once it appears, unchanged,
            in the uncommitted tail of all ``n`` most recent hypotheses. Higher
            ``n`` = more stable, more latent. Must be ``>= 1``.

    Attributes are intentionally read-only via properties; mutate only through
    :meth:`commit` and :meth:`reset`.
    """

    __slots__ = ("_committed", "_committed_end", "_n", "_post_commit_disagreement", "_window")

    def __init__(self, n: int = 2) -> None:
        if n < 1:
            raise ValueError(f"LocalAgreement window n must be >= 1, got {n}")
        self._n = n
        self._committed: list[Token] = []
        self._committed_end = 0
        self._window: deque[Hypothesis] = deque(maxlen=n)
        self._post_commit_disagreement = 0

    @property
    def n(self) -> int:
        """The LocalAgreement window size."""
        return self._n

    @property
    def committed_end(self) -> int:
        """Count of committed tokens (``c`` in the commit rule)."""
        return self._committed_end

    @property
    def committed_prefix(self) -> tuple[Token, ...]:
        """The full stable prefix committed so far this utterance."""
        return tuple(self._committed)

    @property
    def post_commit_disagreement(self) -> int:
        """Count of later hypotheses that disagreed with committed tokens.

        Incremented (never decremented) whenever a fresh hypothesis is long
        enough to cover the committed region yet diverges from it. Surfaced as a
        metric so the user can raise ``n`` to trade latency for fewer
        disagreements.
        """
        return self._post_commit_disagreement

    def reset(self) -> None:
        """Clear the window and committed prefix for a fresh utterance.

        Called automatically after an ``is_final`` force-commit, but also exposed
        so the pipeline can start a brand-new utterance after barge-in.
        """
        self._committed = []
        self._committed_end = 0
        self._window.clear()
        # post_commit_disagreement is cumulative across the session by design
        # (a tuning signal), so it is NOT reset here.

    def _count_disagreement(self, hypothesis: Hypothesis) -> None:
        """Bump the disagreement counter if ``hypothesis`` contradicts commits."""
        c = self._committed_end
        if len(hypothesis.tokens) < c:
            # Shorter than the committed prefix: an empty tail, not a
            # contradiction of specific committed tokens.
            return
        for i in range(c):
            if normalize_token(hypothesis.tokens[i].text) != normalize_token(
                self._committed[i].text
            ):
                self._post_commit_disagreement += 1
                return

    def commit(self, hypothesis: Hypothesis) -> CommitResult:
        """Feed one partial hypothesis and return what was newly committed.

        On a non-final hypothesis the LocalAgreement rule extends the committed
        prefix by the LCP shared across the last ``n`` uncommitted tails. On a
        final hypothesis the whole remaining tail is force-committed and the
        stabilizer resets for the next utterance.
        """
        self._count_disagreement(hypothesis)

        if hypothesis.is_final:
            return self._commit_final(hypothesis)

        self._window.append(hypothesis)
        c = self._committed_end

        newly: tuple[Token, ...] = ()
        if len(self._window) == self._n:
            tails = [list(h.tokens[c:]) for h in self._window]
            length = _lcp_length(tails)
            if length > 0:
                # Use the newest hypothesis as the source of the committed tokens
                # (all n agree on these positions by construction).
                newest_tail = self._window[-1].tokens[c : c + length]
                self._committed.extend(newest_tail)
                self._committed_end = c + length
                newly = newest_tail

        tentative_tail = hypothesis.tokens[self._committed_end :]
        return CommitResult(
            newly_committed=newly,
            committed_prefix=tuple(self._committed),
            tentative_tail=tuple(tentative_tail),
        )

    def _commit_final(self, hypothesis: Hypothesis) -> CommitResult:
        """Force-commit the remaining tail of a final hypothesis, then reset."""
        c = self._committed_end
        if len(hypothesis.tokens) > c:
            newly = hypothesis.tokens[c:]
            self._committed.extend(newly)
        else:
            # Final hypothesis is shorter than (or equal to) the committed
            # prefix: nothing new, and we never un-commit.
            newly = ()
        committed_prefix = tuple(self._committed)
        self.reset()
        return CommitResult(
            newly_committed=newly,
            committed_prefix=committed_prefix,
            tentative_tail=(),
        )
