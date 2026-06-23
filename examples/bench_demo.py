#!/usr/bin/env python3
"""Offline demo: run a scripted fixture through the fake pipeline and show metrics.

No models, no network, no real audio — this replays a two-sentence English
fixture (with a deliberate mid-word ASR revision) through the deterministic fake
STT/MT/TTS on the manual clock and prints:

* per-utterance **first-audio-out latency** (how soon target audio begins —
  i.e. simultaneity), and
* the **audio-stage retraction count**, which is ``0`` because LocalAgreement
  never lets the wrong guess reach MT/TTS, so the synthesized speech never
  stutters.

Run it with::

    python examples/bench_demo.py

or, equivalently, via the installed CLI::

    interpret-live bench
"""

from __future__ import annotations

import asyncio

from interpret_live.bench import default_fixture, run_bench


async def _main() -> None:
    fixture = default_fixture()
    result = await run_bench(fixture)
    report = result.report

    print(f"interpret-live offline demo — fixture {fixture.name!r}")
    print("-" * 60)
    for utt in report.utterances:
        first = "n/a" if utt.first_audio_out_ms is None else f"{utt.first_audio_out_ms} ms"
        lag = "n/a" if utt.commit_lag_ms is None else f"{utt.commit_lag_ms} ms"
        print(
            f"  {utt.utterance_id}: first-audio-out={first:<8} "
            f"commit-lag={lag:<8} disagreements={utt.post_commit_disagreement}"
        )
    print("-" * 60)
    print(f"audio-stage retractions : {result.retraction_count}  (0 = no stutter)")
    print(f"played segments (order) : {list(result.played_segments)}")
    print(f"synthesized samples     : {result.played_samples.size}")
    print()
    print(
        "The ASR revised 'wether' -> 'weather' mid-stream, yet the synthesized "
        "audio shows zero retraction: LocalAgreement committed only the agreed "
        "prefix, so the wrong guess never reached MT/TTS."
    )

    # The demo's contract: audio-stage stability is provable (no retraction).
    assert result.retraction_count == 0


if __name__ == "__main__":
    asyncio.run(_main())
