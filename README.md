# interpret-live

> A real-time speech-to-speech translator CLI — you speak one language, the other side hears another, with audio-stage stabilization so the synthesized speech never stutters, barge-in interrupt handling, and one interface over cloud-realtime or fully-offline backends.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

`interpret-live` performs **simultaneous interpretation**: it streams live speech-to-text → incremental machine translation → streaming text-to-speech in the target language, emitting audio **before the speaker finishes the sentence**. Its marquee idea is **LocalAgreement stabilization pushed into the audio stage** — only the ASR prefix that has *agreed across the last n partial hypotheses* is ever translated and spoken, so the synthesized voice does not stutter or retract as the recognizer revises words.

## What's novel here

Streaming speech-to-speech is fragmented across text-only tools, research models, and vendor-locked demos. The hard, reusable part is the **streaming orchestration**: stabilizing partial hypotheses, segmenting into translatable units, translating only complete clauses, and halting instantly on barge-in — all without leaking tasks or stuttering audio. That orchestration core is what this project builds, tests deterministically, and ships light. The heavy ASR/MT/TTS/cloud backends are pluggable adapters behind optional extras.

- **Audio-stage stabilization (the marquee).** A **LocalAgreement-*n*** stabilizer commits only tokens that have appeared, unchanged, in the last *n* partial hypotheses. The committed prefix is **monotonic — it never retracts** — so a mid-word ASR revision (e.g. `wether` → `weather`) is corrected *before* it ever reaches MT/TTS, and the spoken audio shows **zero retraction**.
- **Simultaneity.** Each sentence/clause is translated and spoken **as soon as it closes** (terminal punctuation or a max-token cap), before the whole utterance ends — so the listener hears the translation begin while the speaker is still talking.
- **Barge-in / VAD.** A debounced energy-VAD onset detector watches the source mic and, on interrupt, **cancels in-flight MT/TTS, discards queued audio, and stops the sink** promptly (a measured `barge-in-stop` metric) — leaking no async tasks.
- **One interface, two backends.** A `Session` dispatcher runs either a **pipeline path** (separate STT + MT + TTS — where the stabilizer is active) or a **unified cloud S2S path** (OpenAI Realtime / Gemini Live — which does speech-to-speech internally). Both satisfy one small interface, so the CLI and dual-channel logic are backend-agnostic.
- **Dual-channel meeting mode** (two independent directions, no cross-talk) and **optional speaker voice-preservation** (ElevenLabs) as a drop-in TTS.

## How it works

```
mic ─┬─► STT.stream ─► LocalAgreement stabilizer ─► segmenter ─► MT (closed segments only) ─► TTS.stream ─► speaker
     └─► VAD / BargeInDetector ───────────────────────────────────────────► interrupt (cancel + discard + stop)
```

The pipeline runs STT, MT, and TTS concurrently with **bounded queues** (real backpressure), driven by an injected `Clock`. The microphone is fanned out (a `tee`/broadcast) so STT and the barge-in detector consume it independently. On barge-in, the current utterance's MT/TTS are cancelled, queued chunks are discarded, the sink is stopped, and a **new utterance starts with a fresh stabilizer** — already-spoken segments are kept and never re-translated.

### Capability matrix (honest — what's active per path)

| Component | Pipeline path (incl. fully offline) | Unified cloud S2S path |
|---|---|---|
| LocalAgreement audio-stage stabilizer | ✅ active (the marquee) | ⛔ N/A — the cloud does S2S internally; the harness never sees ASR partials, so audio-stage stabilization is the provider's responsibility |
| Segmentation / incremental MT | ✅ | ⛔ (cloud-internal) |
| VAD + barge-in (`interrupt()`) | ✅ | ✅ (detect onset on the source mic, send the provider's cancel) |
| Latency / barge-in-stop metrics | ✅ | ✅ |

The novel audio-stage stabilization is delivered on the **pipeline (including the fully-offline)** path. The cloud S2S path trades our stabilizer for cloud-quality end-to-end translation — stated plainly rather than implied.

## Scope & status

🚧 **Early development.** This repository is built in the open. The current build delivers and **deterministically tests** the streaming-orchestration core (stabilizer, segmentation, VAD/barge-in, the interruptible asyncio pipeline, metrics, the `Session` dispatcher, dual-channel) end-to-end over deterministic fakes. The **audio pipeline is fully modeled and tested**; the heavy ML/audio/cloud backends are **optional extras** behind import-guarded adapters (a missing extra raises a clear `install interpret-live[...]` error, never an obscure `ImportError`).

- [x] LocalAgreement audio-stage stabilizer + segmentation + VAD/barge-in (pure, deterministic)
- [x] Interruptible asyncio pipeline + metrics + `Session`/dual-channel + CLI `bench` (offline)
- [x] Offline adapters wired live: faster-whisper → NLLB → Piper (`[whisper]`, `[mt]`, `[piper]`, `[audio]`) — spawned model workers, endpointing, model prefetch/cache/offline mode ([guide](docs/live-adapters.md))
- [x] Cloud S2S live: OpenAI Realtime (`[openai]` + `[audio]`) — persistent session, response-scoped barge-in cancel/truncate
- [ ] Gemini Live (`[gemini]`); ElevenLabs voice preservation (`[elevenlabs]`)

## Install

The default install is light (stdlib + numpy + pydantic + typer/rich) and runs the offline demo with no models or network:

```bash
pip install interpret-live          # core + deterministic fakes
```

Optional backends are extras (install only what you need):

```bash
pip install 'interpret-live[whisper,mt,piper,audio]'   # fully-offline live pipeline
pip install 'interpret-live[openai,audio]'             # OpenAI Realtime (mic/speaker need [audio])
pip install 'interpret-live[elevenlabs]'               # voice-preserving TTS (planned)
```

Requires Python **3.11+** (validated on 3.11, 3.12, 3.13).

## Quick start

Run the offline, no-extras demo — it replays a scripted fixture through the fake backends and prints latency metrics plus audio-stage stability:

```bash
interpret-live bench
```

```
          interpret-live bench — fixture 'default-en-2sent'
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ utterance ┃ first-audio-out (ms) ┃ commit-lag (ms) ┃ disagreements ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ utt-1     │                  350 │              60 │             0 │
│ utt-2     │                  170 │              60 │             0 │
└───────────┴──────────────────────┴─────────────────┴───────────────┘
audio-stage retractions: 0 (0 = synthesized speech never stuttered)
```

### Why `n` is tunable (the LocalAgreement tradeoff)

The built-in `late-revision-en` fixture makes the stability/latency tradeoff a
one-liner. Its ASR closes a segment on a wrong guess (`buck.`) and only *then*
revises it to `book.` — a correction that lands **after** the token would commit
at `n=1` but **before** it could commit at `n=2`:

```bash
interpret-live bench --fixture late-revision-en --agreement-n 1   # disagreements > 0 (spoke "buck.")
interpret-live bench --fixture late-revision-en --agreement-n 2   # disagreements 0 (waited, spoke "book.")
```

At `n=1` the eager commit ships the misread to MT/TTS and the later `book.`
contradicts an already-committed token, so the **disagreements** column is
non-zero — the signal to raise `n`. At `n=2` the wrong guess never commits, the
correct sentence is spoken, and the column is `0`. **Retractions stay `0` at
both** because the committed prefix is monotonic: a late disagreement only bumps
the tuning counter, it never un-commits already-spoken audio.

The same demo as a script lives at [`examples/bench_demo.py`](examples/bench_demo.py):

```bash
python examples/bench_demo.py
```

Other commands:

```bash
interpret-live models download --backend offline --from en --to es   # visible, cached prefetch
interpret-live run --from en --to es --backend offline               # live offline session
interpret-live run --from en --to es --backend cloud --openai-voice marin
interpret-live devices                                   # list audio devices (needs [audio])
```

Live setup, model cache/offline mode, device selection, dual mode, and
troubleshooting are documented in [docs/live-adapters.md](docs/live-adapters.md).

## Library

```python
import asyncio
from interpret_live import Session, PipelineConfig, ManualClock
from interpret_live.session import PipelineBackend
from interpret_live.backends.fake import FakeSTT, FakeMT, FakeTTS
from interpret_live.audio_io import FakeAudioSource, FakeAudioSink

clock = ManualClock()
backend = PipelineBackend(
    stt=FakeSTT([...], clock=clock),
    mt=FakeMT({...}, clock=clock),
    tts=FakeTTS(clock=clock),
)
session = Session.create(
    backend=backend,
    source=FakeAudioSource([...], clock=clock),
    sink=FakeAudioSink(clock=clock),
    clock=clock,
    config=PipelineConfig(agreement_n=2),
    enable_barge_in=True,
)
asyncio.run(session.run())
```

Swap the fakes for `WhisperSTT` / `NllbMT` / `PiperTTS` (offline) or an `S2SBackend` wrapping `RealtimeS2S` / `GeminiS2S` (cloud) — the `Session`/CLI code is unchanged.

## Determinism

The entire core is tested **offline and deterministically**: scripted fake STT/MT/TTS/VAD, fake audio source/sink, and a **manual `Clock` with a drain-then-advance pump**. `asyncio.sleep()` is forbidden in the core and fakes (everything pacing-related uses the injected clock), and a test asserts the whole suite runs in well under a wall-clock second — catching any stray real sleep that would silently break reproducibility.

## Development

```bash
pip install -e '.[dev]'
ruff check && ruff format --check
mypy src
pytest -q --cov=interpret_live --cov-report=term-missing --cov-fail-under=90
```

PR CI covers the live adapters through mocked hardware/model/transport
boundaries (plus a dedicated all-extras contract job); the opt-in
`pytest -m live_smoke` suite and the manual audible procedure in
[docs/live-adapters.md](docs/live-adapters.md) exercise real models.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
