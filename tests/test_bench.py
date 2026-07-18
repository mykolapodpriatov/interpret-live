"""Benchmark harness tests: the fixture registry and the LocalAgreement tradeoff.

Every run is offline and deterministic (scripted fakes + a ManualClock), so the
assertions below are exact rather than probabilistic.
"""

from __future__ import annotations

import asyncio

import pytest

from interpret_live.bench import (
    FIXTURES,
    default_fixture,
    get_fixture,
    late_revision_fixture,
    run_bench,
)
from interpret_live.config import PipelineConfig


def _disagreements_and_retractions(agreement_n: int) -> tuple[int, int]:
    """Run ``late-revision-en`` at ``agreement_n`` and return (disagreements, retractions)."""
    fixture = get_fixture("late-revision-en")
    cfg = PipelineConfig(agreement_n=agreement_n)
    result = asyncio.run(run_bench(fixture, config=cfg))
    return result.report.total_post_commit_disagreement, result.retraction_count


def test_registry_lists_both_builtin_fixtures() -> None:
    assert set(FIXTURES) == {"default-en-2sent", "late-revision-en"}


def test_get_fixture_returns_named_fixture() -> None:
    assert get_fixture("default-en-2sent").name == "default-en-2sent"
    assert get_fixture("late-revision-en").name == "late-revision-en"


def test_get_fixture_builds_a_fresh_instance_each_call() -> None:
    # A registry of factories (not shared singletons) keeps callers isolated.
    assert get_fixture("late-revision-en") is not get_fixture("late-revision-en")


def test_get_fixture_unknown_name_lists_available() -> None:
    with pytest.raises(ValueError, match="unknown fixture 'nope'") as exc:
        get_fixture("nope")
    message = str(exc.value)
    assert "default-en-2sent" in message and "late-revision-en" in message


def test_builtin_factories_are_registered() -> None:
    assert FIXTURES["default-en-2sent"] is default_fixture
    assert FIXTURES["late-revision-en"] is late_revision_fixture


def test_late_revision_makes_the_agreement_n_tradeoff_visible() -> None:
    """n=1 commits the wrong guess (disagreement fires); n=2 waits and stays clean.

    Retractions are 0 at both n: the committed prefix is monotonic, so a late
    disagreement bumps the tuning counter but never un-commits spoken audio.
    """
    disagree_n1, retractions_n1 = _disagreements_and_retractions(agreement_n=1)
    disagree_n2, retractions_n2 = _disagreements_and_retractions(agreement_n=2)

    assert disagree_n1 > 0  # eager commit shipped "buck.", later contradicted by "book."
    assert disagree_n2 == 0  # the wrong guess never committed, so nothing to contradict
    assert retractions_n1 == 0
    assert retractions_n2 == 0


def test_late_revision_synthesizes_one_segment_at_both_n() -> None:
    """The disagreement is not cosmetic: a real segment is spoken at both n."""
    fixture = get_fixture("late-revision-en")

    async def _run(agreement_n: int) -> tuple[set[int], int]:
        result = await run_bench(fixture, config=PipelineConfig(agreement_n=agreement_n))
        return set(result.played_segments), int(result.played_samples.size)

    for agreement_n in (1, 2):
        segments, samples = asyncio.run(_run(agreement_n))
        assert segments == {0}  # exactly the one closed sentence reached the sink
        assert samples > 0  # and it actually synthesized audio
