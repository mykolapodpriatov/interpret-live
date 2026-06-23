"""LocalAgreement-n stabilizer tests — the correctness heart (exhaustive).

Covers the exact ``new_committed_end = max(c, c + LCP(tails))`` rule with the LCP
shared by ALL n tails, never-retract + post_commit_disagreement, fewer-than-n,
shrinking, shorter-than-c, the pinned normalization, is_final force-commit, token
granularity, and multi-utterance window reset.
"""

from __future__ import annotations

import string

import pytest

from helpers import hyp
from interpret_live.stabilize import LocalAgreementStabilizer, normalize_token


def _texts(tokens: tuple) -> list[str]:
    return [t.text for t in tokens]


# ----- Normalization (pinned exactness) ---------------------------------------


def test_normalize_strips_leading_trailing_punctuation_and_lowercases() -> None:
    assert normalize_token("Hello,") == "hello"
    assert normalize_token("Mr.") == "mr"
    assert normalize_token("(world)") == "world"
    assert normalize_token("...wait...") == "wait"


def test_normalize_preserves_internal_punctuation() -> None:
    # "don't" keeps its apostrophe and is therefore distinct from "dont".
    assert normalize_token("don't") == "don't"
    assert normalize_token("don't") != normalize_token("dont")


def test_normalize_equivalences_drive_agreement() -> None:
    assert normalize_token("Hello,") == normalize_token("hello")
    assert normalize_token("Mr.") == normalize_token("Mr")


def test_norm_is_exactly_lower_strip_punctuation() -> None:
    # Pin the exact definition so the contract cannot silently drift.
    sample = "Hello,World.don't"
    assert normalize_token(sample) == sample.lower().strip(string.punctuation)


# ----- Basic agreement / LCP across ALL n -------------------------------------


def test_commits_prefix_agreed_across_n_hypotheses() -> None:
    s = LocalAgreementStabilizer(n=2)
    r1 = s.commit(hyp("the", "weather"))
    assert r1.newly_committed == ()  # fewer than n hypotheses considered
    r2 = s.commit(hyp("the", "weather", "is"))
    # Both tails agree on "the", "weather"; "is" present only in the newest.
    assert _texts(r2.newly_committed) == ["the", "weather"]
    assert _texts(s.committed_prefix) == ["the", "weather"]


def test_lcp_is_shared_by_all_n_not_pairwise() -> None:
    # With n=3, the LCP stops at the FIRST position where ANY tail disagrees.
    s = LocalAgreementStabilizer(n=3)
    s.commit(hyp("a", "b", "c"))
    s.commit(hyp("a", "b", "x"))  # disagrees at index 2
    r = s.commit(hyp("a", "b", "c"))
    # Tails: [a,b,c], [a,b,x], [a,b,c]. All share "a","b"; index 2 differs in
    # the middle tail, so the shared LCP is exactly 2 (not a pairwise 3 between
    # the 1st and 3rd).
    assert _texts(r.newly_committed) == ["a", "b"]


def test_fewer_than_n_commits_nothing() -> None:
    s = LocalAgreementStabilizer(n=3)
    assert s.commit(hyp("a", "b")).newly_committed == ()
    assert s.commit(hyp("a", "b")).newly_committed == ()
    # Only on the 3rd hypothesis does the window fill and a commit become possible.
    r = s.commit(hyp("a", "b"))
    assert _texts(r.newly_committed) == ["a", "b"]


def test_identical_repeated_hypotheses_advance_commit() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("hello", "world"))
    r = s.commit(hyp("hello", "world"))
    assert _texts(r.newly_committed) == ["hello", "world"]


# ----- Never-retract + post_commit_disagreement -------------------------------


def test_never_retracts_when_later_hypothesis_shortens() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("the", "weather", "is"))
    s.commit(hyp("the", "weather", "is"))  # commits the,weather,is
    assert _texts(s.committed_prefix) == ["the", "weather", "is"]
    # A later, shorter hypothesis must NOT un-commit anything.
    r = s.commit(hyp("the", "weather"))
    assert _texts(s.committed_prefix) == ["the", "weather", "is"]
    assert r.newly_committed == ()


def test_disagreement_with_committed_tokens_increments_counter_not_retract() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("i", "scream"))
    s.commit(hyp("i", "scream"))  # commits "i","scream"
    assert _texts(s.committed_prefix) == ["i", "scream"]
    assert s.post_commit_disagreement == 0
    # A revision contradicting a committed token ("scream" -> "cream").
    s.commit(hyp("i", "cream", "now"))
    assert s.post_commit_disagreement == 1
    # Committed prefix is unchanged (monotonic).
    assert _texts(s.committed_prefix) == ["i", "scream"]


def test_disagreement_counter_does_not_increment_for_shorter_hypothesis() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("alpha", "beta", "gamma"))
    s.commit(hyp("alpha", "beta", "gamma"))
    assert _texts(s.committed_prefix) == ["alpha", "beta", "gamma"]
    # A hypothesis SHORTER than the committed prefix contributes an empty tail,
    # not a contradiction of specific committed tokens.
    s.commit(hyp("alpha", "beta"))
    assert s.post_commit_disagreement == 0


# ----- Shorter-than-c edge ----------------------------------------------------


def test_hypothesis_shorter_than_committed_end_empty_tail_no_uncommit() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("one", "two", "three"))
    s.commit(hyp("one", "two", "three"))
    assert s.committed_end == 3
    # Shorter than committed_end => empty tail => L == 0 => no change.
    r = s.commit(hyp("one"))
    assert r.newly_committed == ()
    assert s.committed_end == 3
    assert _texts(s.committed_prefix) == ["one", "two", "three"]


# ----- is_final force-commit + reset ------------------------------------------


def test_is_final_force_commits_remaining_tail() -> None:
    s = LocalAgreementStabilizer(n=3)
    s.commit(hyp("hold", "the"))  # < n hypotheses, nothing committed
    assert s.commit(hyp("hold", "the", "line")).newly_committed == ()
    third = s.commit(hyp("hold", "the", "line"))  # window now full (n=3)
    # The oldest tail ("hold","the") bounds the shared LCP, so only those two
    # commit; "line" stays tentative because one tail in the window lacks it.
    assert _texts(third.committed_prefix) == ["hold", "the"]
    # A final hypothesis force-commits the remaining tail ("line","now").
    fr = s.commit(hyp("hold", "the", "line", "now", is_final=True))
    assert _texts(fr.committed_prefix) == ["hold", "the", "line", "now"]
    assert _texts(fr.newly_committed) == ["line", "now"]


def test_is_final_resets_window_and_committed_end() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("first", "utterance"))
    s.commit(hyp("first", "utterance", is_final=True))
    assert s.committed_end == 0
    assert s.committed_prefix == ()


def test_is_final_shorter_than_committed_does_not_uncommit() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("keep", "this", "text"))
    s.commit(hyp("keep", "this", "text"))  # commit all three
    # A final hypothesis shorter than committed prefix: nothing new, no un-commit,
    # then reset.
    fr = s.commit(hyp("keep", is_final=True))
    assert _texts(fr.committed_prefix) == ["keep", "this", "text"]
    assert s.committed_end == 0


# ----- Multi-utterance: final -> reset -> fresh -------------------------------


def test_multi_utterance_commits_fresh_after_reset_no_stale_bleed() -> None:
    s = LocalAgreementStabilizer(n=2)
    # Utterance 1.
    s.commit(hyp("good", "morning"))
    s.commit(hyp("good", "morning", is_final=True))
    assert s.committed_end == 0
    # Utterance 2 — different words. Must commit fresh, with no influence from
    # the prior utterance's tokens.
    r1 = s.commit(hyp("good", "night"))
    assert r1.newly_committed == ()  # only one hypothesis in the fresh window
    r2 = s.commit(hyp("good", "night"))
    assert _texts(r2.newly_committed) == ["good", "night"]


# ----- Token (not char) granularity -------------------------------------------


def test_token_granularity_not_character() -> None:
    s = LocalAgreementStabilizer(n=2)
    s.commit(hyp("internationalization"))
    r = s.commit(hyp("internationalization"))
    # The whole word commits as one token; partial-character agreement is N/A.
    assert _texts(r.newly_committed) == ["internationalization"]


# ----- Empty hypotheses -------------------------------------------------------


def test_empty_hypotheses_commit_nothing() -> None:
    s = LocalAgreementStabilizer(n=2)
    assert s.commit(hyp()).newly_committed == ()
    assert s.commit(hyp()).newly_committed == ()
    assert s.committed_prefix == ()


# ----- Construction guards ----------------------------------------------------


def test_window_n_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        LocalAgreementStabilizer(n=0)


def test_normalization_drives_agreement_with_punctuation_variation() -> None:
    s = LocalAgreementStabilizer(n=2)
    # "Hello," and "hello" normalize equal, so they agree and commit. The
    # committed token text is taken from the newest hypothesis ("hello").
    s.commit(hyp("Hello,", "world"))
    r = s.commit(hyp("hello", "world"))
    assert _texts(r.newly_committed) == ["hello", "world"]
