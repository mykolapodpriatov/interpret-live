# Live adapters: setup, models, devices, and troubleshooting

This guide covers running `interpret-live` **live** — real microphone and
speaker, real models — on the two supported paths:

1. **Offline pipeline**: microphone → faster-whisper → LocalAgreement → NLLB →
   Piper → speaker. The LocalAgreement audio-stage stabilizer is **active**.
2. **Cloud S2S**: microphone → one persistent provider connection →
   translated audio → speaker. The provider does S2S internally. Two providers
   are supported: **OpenAI Realtime** (`--provider openai`) and **Gemini Live**
   (`--provider gemini`).

Everything here degrades safely: a missing extra prints an actionable
`pip install 'interpret-live[...]'` hint, never a raw traceback.

## Install

```bash
# fully-offline live pipeline
pip install 'interpret-live[whisper,mt,piper,audio]'

# OpenAI Realtime (the microphone/speaker still need the audio extra)
pip install 'interpret-live[openai,audio]'

# Gemini Live
pip install 'interpret-live[gemini,audio]'
```

Python 3.11–3.13. The `[openai]` extra installs `openai[realtime]` (the SDK's
WebSocket transport) and `soxr`; `[gemini]` installs `google-genai`; `[audio]`
installs `sounddevice` + `soxr`.

## Credentials

Each provider's key is read by **its own SDK, from the environment only** —
never a CLI flag, never logged, never part of an error message:

```bash
export OPENAI_API_KEY=...   # --provider openai
export GEMINI_API_KEY=...   # --provider gemini (GOOGLE_API_KEY also works)
```

## Models: prefetch, cache, offline

Model downloads are **visible, atomic, cached, and checksum-verified** (for
directly managed Piper files; Hugging Face snapshots are pinned to explicit
revisions). Prefetch everything a run needs, once:

```bash
interpret-live models download --backend offline --from en --to es
```

- The cache root defaults to the platform cache dir (e.g.
  `~/Library/Caches/interpret-live` on macOS, `~/.cache/interpret-live` on
  Linux); override with `--cache-dir`.
- A second download run performs **no network requests**.
- Interrupted/corrupt downloads are never treated as valid cache entries
  (temporary file + SHA-256 verification + atomic rename).
- Every shipped Piper voice's license/source is printed after download and
  listed in the manifest (`interpret_live/data/piper_voices.json`).
- `run` performs the same non-interactive preflight automatically when
  artifacts are missing.

Inspect or reclaim the cache without touching the network:

```bash
interpret-live models list                 # cached artifacts + on-disk size
interpret-live models clear                 # prompts, then deletes the whole cache root
interpret-live models clear --yes           # skip the confirmation prompt
```

- `models list` walks the cache root and prints every resolved whisper/NLLB
  snapshot and Piper voice it finds, with size and path; both commands accept
  `--cache-dir` to target a non-default cache root.
- `models clear` reports how many artifacts and bytes it would remove, asks
  for confirmation (unless `--yes`), then calls the same purge used by
  maintenance/tests and prints how much space was freed.

**Offline mode** (`--offline`) performs no network access at all; if anything
is missing it lists **every** missing artifact in one error:

```bash
interpret-live run --backend offline --offline --from en --to es
```

`--offline` governs local model resolution only — combining it with
`--backend cloud` is rejected up front.

## Devices

```bash
interpret-live devices
```

lists every device with its index, input/output channel counts, default sample
rate, and whether it is the system default. Pass indexes explicitly:

```bash
interpret-live run --backend offline --from en --to es \
  --input-device 1 --output-device 2
```

Device capabilities are validated **before** models load: a device that cannot
capture/play mono at the configured rate fails fast with a typed error.

## Single direction

```bash
# offline (stabilizer active)
interpret-live models download --backend offline --from en --to es
interpret-live run --backend offline --from en --to es \
  --voice es_ES-davefx-medium --input-device 1 --output-device 2

# OpenAI Realtime (OPENAI_API_KEY exported)
interpret-live run --backend cloud --provider openai --from en --to es \
  --openai-voice marin --input-device 1 --output-device 2

# Gemini Live (GEMINI_API_KEY exported)
interpret-live run --backend cloud --provider gemini --from en --to es \
  --gemini-voice Puck --input-device 1 --output-device 2
```

Expected behavior:

- translated audio begins **before you finish the sentence** (the summary
  table printed on exit shows per-turn `first-audio-out` latency, measured
  from actual speech onset to the first audible sample);
- no already-spoken segment is ever repeated;
- **barge-in**: speaking again while translation audio is playing stops the
  old audio promptly (the `barge-in-stop` metric) and your new speech is
  translated as a fresh turn. On OpenAI Realtime exactly one response-scoped
  cancel + heard-audio truncation is sent to the provider; the Gemini Live API
  exposes no such cancel, so there the barge-in is enforced locally — playback
  stops and the abandoned turn's remaining audio is dropped at the adapter
  boundary while the provider finishes generating it unheard;
- **Ctrl-C** stops the session cleanly: devices close, model workers are
  cooperatively stopped (then terminated within a bounded budget if stuck),
  and the metrics summary prints. No child processes survive — check with
  `pgrep -fl interpret-live` after exit.

## Dual direction (meeting mode)

Dual mode runs two **fully independent** directional stacks (nothing stateful
is shared; the cloud providers use two connections). Explicit A/B devices are
required:

```bash
interpret-live run --backend offline --from en --to es --dual \
  --input-device 1 --output-device 2 \
  --input-device-b 3 --output-device-b 4 \
  --voice es_ES-davefx-medium --voice-b en_US-lessac-medium
```

- A→B translates `--from`→`--to` and speaks with the target-language voice;
  B→A is reversed and speaks with the source-language voice (`--voice-b`).
- Reusing the same physical device for both directions prints a
  feedback/cross-talk warning.

## Tuning

| Option | Meaning |
|---|---|
| `--whisper-model` | faster-whisper alias (`tiny`/`base`/`small`/`medium`/`large-v3`) or a local snapshot path |
| `--nllb-model` | NLLB model id / local path override |
| `--voice` / `--voice-b` | Piper voice id from the manifest, or a local `.onnx` path |
| `--openai-voice` / `--openai-model` | Realtime output voice / model id |
| `--gemini-voice` / `--gemini-model` | Gemini Live prebuilt voice / model id |
| `--gemini-native-translation` | send Gemini's own `translation_config` (needs a translation-capable Live model) |
| `--no-barge-in` | disable the barge-in interrupt |
| `--agreement-n` | LocalAgreement window (higher = more stable, more latent); default 2 |
| `--max-segment-tokens` | Forced-flush cap for an open segment; default 24 |
| `--vad-threshold` | RMS speech threshold for the energy VAD; default 0.02 |
| `--vad-hangover-ms` | Trailing silence before the VAD flips to silence; default 200 |

## Troubleshooting

- **`The 'whisper' backend requires the 'whisper' extra`** — install the
  printed extra; `interpret-live bench` always works with no extras.
- **`offline mode: required model artifacts are missing`** — run
  `interpret-live models download` once without `--offline`.
- **`input device N cannot capture mono float32 at 16000 Hz`** — pick another
  index from `interpret-live devices` (the default rate column helps).
- **Choppy/robotic output audio** — the speaker sink counts ring-buffer
  underruns; a loaded machine or an aggressive DAW grabbing the device are the
  usual causes. Try the device's default rate (omit overrides).
- **Nothing is transcribed** — the endpointing VAD needs speech above its RMS
  threshold; check the mic input level, or lower it with `--vad-threshold`
  (or `PipelineConfig.vad_threshold` on the library API).
- **Cloud session ends with `realtime connection failed`** — the transport
  never reconnects silently after audio has been sent (replay could duplicate
  or lose speech); restart the run.
- **A model worker seems stuck on exit** — shutdown escalates automatically:
  cooperative stop → terminate → kill, each within the configured grace
  budget (`AudioConfig.shutdown_timeout_ms`).

## Maintainer smoke (opt-in)

Automated CI never touches real models, hardware, or the network. With local
models + an API key, maintainers can run the opt-in smoke:

```bash
pip install -e '.[whisper,mt,piper,audio,openai,dev]'
pytest -m live_smoke tests/test_live_smoke.py
```

and the full audible procedure:

```bash
interpret-live models download --backend offline --from en --to es
interpret-live run --backend offline --from en --to es --voice es_ES-davefx-medium \
  --input-device <id> --output-device <id>
# with OPENAI_API_KEY already exported in the environment
interpret-live run --backend cloud --provider openai --from en --to es \
  --openai-voice marin --input-device <id> --output-device <id>
```

For each live smoke, verify: audible translated output, no repeated prior
segment, visible first-audio latency in the exit summary, barge-in stops old
audio, speech resumes after the interrupt, Ctrl-C closes devices promptly, and
no traceback or background process remains (`pgrep -fl interpret-live`). For
the OpenAI smoke also verify the configured output voice is audibly selected
and no API key or authorization header appears in any debug output.
