# Implementation Plan: Live Offline Adapters and OpenAI Realtime

**Status:** Revised after plan review on 2026-07-17; re-reviewed against the codebase on 2026-07-17 (r2) — baseline claims verified, per-task test-maintenance rule added.

**Scope:** Make the existing offline adapters, real audio edges, and OpenAI Realtime path usable end to end from the CLI without weakening the deterministic fake/core path.

## Objective

Deliver two genuinely runnable live modes behind the existing `Session` surface:

1. **Offline pipeline:** microphone -> faster-whisper -> LocalAgreement -> NLLB -> Piper -> speaker.
2. **OpenAI Realtime:** microphone -> persistent Realtime connection -> translated audio -> speaker.

Both modes must remain interruptible, bounded, cleanly shut down, and testable without real hardware, model downloads, or network access in the normal PR test suite.

## Current Baseline

The live adapters are not empty; the work starts from these partial implementations:

- `src/interpret_live/backends/whisper.py` constructs `WhisperModel` and performs synchronous re-decodes from an async generator. It emits a final hypothesis only when its input iterator ends.
- `src/interpret_live/backends/nllb.py` loads Transformers and generates a translation synchronously inside `async translate()`. It currently prepends rolling context and returns the translation of that entire combined input.
- `src/interpret_live/backends/piper.py` calls `synthesize_stream_raw()` but materializes all chunks with `list()` before yielding.
- `src/interpret_live/audio_io.py` has real `MicSource` and `SpeakerSink` classes, but lacks robust overflow/status handling, persistent output-stream ownership, format conversion, and explicit resource cleanup.
- `src/interpret_live/backends/realtime.py` is the only adapter whose primary stream/interrupt methods remain `NotImplemented`.
- `src/interpret_live/session.py` already owns `PipelineBackend` and `S2SBackend`; provider adapters must satisfy their protocols rather than bypassing the session/pipeline.
- `interpret-live run` currently prints install hints and exits without constructing or running a session.
- CI exercises the deterministic core only, and coverage excludes every live adapter plus `audio_io.py`.

The existing fake-backed `bench`, core protocols, LocalAgreement behavior, and deterministic test suite are regression constraints throughout this plan.

## Scope Boundaries

### In scope

- Live offline STT/MT/TTS and real microphone/speaker operation.
- OpenAI Realtime over the official asynchronous Python SDK WebSocket surface.
- Visible, cache-aware model preflight and an explicit offline mode.
- Single-direction and dual-direction runtime composition with independent stateful backends.
- Mocked contract/integration tests, optional-extra CI, documentation, and manual smoke procedures.

### Out of scope

- Gemini Live and ElevenLabs implementation.
- Silero VAD in the first live release; use the existing `EnergyVAD` plus a deterministic endpointing state machine.
- Token-level MT-to-TTS streaming. The current TTS protocol receives one completed translated segment; Piper must stream audio chunks for that segment without buffering the entire result.
- A context-aware NLLB alignment algorithm. Until one is proven, NLLB translates only the current segment so already-spoken context is never repeated.
- Model training, quantization, automatic device benchmarking, or production hosting.

## Required Architecture Decisions

These decisions are fixed before adapter work begins so each task implements the same contracts.

1. **Canonical in-process audio:** mono normalized `numpy.float32` in `[-1.0, 1.0]`, with the actual sample rate carried by every `AudioFrame`/audio chunk. PCM16 is a wire/model-boundary encoding, not the core type.
2. **Explicit, stateful rate conversion:** Whisper receives 16 kHz audio; OpenAI PCM receives 24 kHz mono little-endian PCM16; Piper output uses the voice model's declared native rate; `SpeakerSink` owns one configured device rate. Each continuous stream owns a stateful resampler that preserves filter/phase state across blocks, flushes at a real stream/utterance boundary, and resets on stop or a validated rate change.
3. **Killable model isolation:** no model construction, decode, generation, or synthesis runs on the event loop. Each stateful offline adapter owns a long-lived `multiprocessing` worker created with the cross-platform `spawn` context; models are constructed inside that worker and only serializable requests/results cross bounded IPC queues. Cooperative cancellation is attempted first, but a worker that does not acknowledge shutdown within the configured grace period is terminated, joined, and killed as a final fallback. Cancelling an `asyncio` future or discarding a late thread result is not considered sufficient shutdown.
4. **Bounded live flow:** microphone, worker-request, worker-result, provider-output, and playback queues are bounded. Live capture uses an explicit drop-oldest policy with counters rather than allowing callback `QueueFull` exceptions.
5. **Offline turn ownership:** the STT adapter owns endpointing and emits full-prefix partial `Hypothesis` values plus exactly one `is_final=True` value per detected utterance while continuing to consume the same live source.
6. **Barge-in ownership:** local `BargeInDetector` is authoritative for immediate playback stop and is armed only while target work for the current turn is in flight, queued, or playing; ordinary speech after silence with no active target work is a new turn, not an interrupt. OpenAI server VAD creates responses but does not independently auto-cancel them; the local interrupt path performs provider cancellation and conversation truncation once.
7. **Resource ownership:** runtime composition owns real devices, model-worker processes, provider clients, and sessions through `AsyncExitStack`. Cancellation or Ctrl-C closes them exactly once and leaves no background tasks, audio threads, or child processes.
8. **Secrets:** `OPENAI_API_KEY` is resolved by the SDK from the process environment. It is never accepted as a visible CLI value, logged, written to project config, or included in exceptions.
9. **One timing domain:** source onset, provider lifecycle, playback presentation, and metrics use the injected monotonic `Clock` domain. PortAudio ADC/DAC timestamps are calibrated to that clock when streams start; raw device-stream time is never subtracted directly from application monotonic time.

## Implementation Sequence

Tasks are ordered by dependency. Each task must preserve `interpret-live bench`, the default no-extras import path, and the existing unit suite before the next task begins.

**Test maintenance is part of every task, not only Task 9.** Tasks 1-8 change public contracts the existing suite and `bench` exercise (`AudioSink.play` -> `schedule`, the `S2S.stream` signature, new `Hypothesis` fields, `DualChannel.create`). Each task therefore updates the affected fakes, existing tests, and `bench.py` in the same change so `ruff`/`mypy`/`pytest --cov-fail-under=90` stay green at every merge; the per-task acceptance criteria land as tests inside that task's change. Task 9 consolidates the remaining new contract/integration suites, CI jobs, coverage-omission removal, and documentation.

### Task 1: Audio Contract, Conversion, and Real Device Lifecycle (`[audio]`)

**Files:**

- `src/interpret_live/types.py`
- `src/interpret_live/audio_io.py`
- `src/interpret_live/audio_codec.py` (new)
- `src/interpret_live/pipeline.py`
- `src/interpret_live/s2s.py`
- `src/interpret_live/config.py`
- `pyproject.toml`

**Changes:**

1. Keep `AudioFrame.samples` and `TtsChunk.samples` as mono `float32`; add constructor/runtime validation for one-dimensional finite samples, positive sample rates, and normalized/clipped PCM boundaries.
2. Add conversion helpers:
   - float32 <-> little-endian PCM16 bytes;
   - a stateful mono `StreamingResampler` around `soxr`, with deterministic `process()`, `flush()`, and `reset()` semantics;
   - chunk-duration calculation used by playback accounting.
3. Add `soxr>=0.5` to each extra that directly performs resampling (`[audio]`, `[whisper]`, and `[openai]`), keep the import lazy inside the resampling helper, and keep the compiled dependency outside the light default install.
4. Refactor `MicSource`:
   - validate the selected input device supports mono capture at the requested rate;
   - never let the PortAudio callback raise on a full queue;
   - drop the oldest frame on overflow and expose/log a drop counter;
   - surface non-empty callback status as a typed audio error on the async consumer;
   - close the input stream when iteration is cancelled or the runtime exits;
   - calibrate PortAudio ADC timestamps to the injected monotonic clock, use frame-derived timestamps in that shared domain, and reject sample-rate changes mid-stream.
5. Refine `AudioSink` with an explicit playback contract and update the fake sink with the same semantics:
   - replace sequential `await sink.play(chunk)` consumption with `await sink.schedule(playback_generation, chunk) -> PlaybackHandle`; scheduling waits only for bounded sink capacity, not for audible completion, so the next chunk can be buffered before the current one ends;
   - issue an opaque monotonic `PlaybackGeneration` for each pipeline utterance/provider response. Only one generation may own a sink at a time; the next remains in the pipeline's bounded queue until the previous generation drains or is stopped. Within the active generation, multiple chunks use bounded lookahead. `schedule()` validates the token both before waiting and again under the sink lock immediately before enqueue, so invalidation cannot race capacity release;
   - each handle exposes async `started` and `completed` notifications plus immutable `PlaybackProgress`/`PlaybackReceipt` values containing handle/chunk ownership, source-equivalent presented samples/source rate, device-presented frames/device rate, first-audible timestamp, and completion/interrupted state;
   - `drain()` awaits all scheduled handles at normal EOF; playback stages retain only bounded outstanding handles, reap completed receipts in order, and never create an unbounded list of completion tasks;
   - `stop(generation)` invalidates that generation under the same lock before releasing capacity, atomically snapshots every affected handle's presentation position, aborts/clears queued output through an independent control path, resolves completion as interrupted, and wakes blocked schedules with a typed rejection; a stale schedule can never enqueue after the snapshot.
6. Refactor `SpeakerSink` to own a persistent callback-mode `sounddevice.OutputStream` instead of invoking global `sd.play()` or blocking `write()` per chunk:
   - resample every chunk to the configured output-device rate;
   - feed a bounded ring buffer in order while the PortAudio callback remains non-blocking;
   - zero-fill and surface a typed underrun/status error plus counter when the ring buffer cannot satisfy a callback, never replay stale memory;
   - avoid opening a new stream between chunks;
   - calibrate PortAudio stream/DAC time to the injected monotonic clock and maintain a per-chunk scheduled-frame ledger in that shared domain; map presented device frames back to source-content duration across the stateful resampler and clamp progress to current calibrated stream time;
   - make `stop()` use a direct, lock-safe abort path that can never wait behind playback work on the same worker, clear only this sink, and return the frozen progress snapshot;
   - flush and reset the output resampler at a final chunk, stop, or input-rate change rather than independently resampling every chunk;
   - expose an idempotent async close operation for runtime cleanup.
7. Update both offline and S2S playback stages to schedule bounded lookahead, await handle notifications without serializing chunk scheduling, drain on normal completion, and track every pending schedule/notification task by playback generation. On stop/failure, invalidate the generation in the sink first, then cancel and await every pending task before a fresh generation is issued. Emit `first_tts_out` from the first handle's `started` receipt. This preserves the existing metric name while making its documented meaning “first audible target audio,” not “first chunk received.”
8. Extend device enumeration to report default status, supported directions, channel counts, and default rates. Fail before model loading if a requested device cannot satisfy its role.
9. Add audio/device fields to validated configuration: input/output device IDs, capture rate, playback rate, frame duration, queue size, and bounded playback/shutdown timeouts.

**Acceptance:**

- PCM16 round-trips within quantization tolerance and 16/22.05/24/48 kHz conversions produce the expected duration.
- Chunked stateful resampling matches one-shot resampling within tolerance, has no cumulative sample-count drift, flushes its tail once, and resets cleanly after stop.
- A mocked microphone callback can overflow repeatedly without an uncaught loop exception; drops are counted.
- With resampling plus simulated device buffering/output latency, `SpeakerSink.stop()` returns only source-equivalent samples whose DAC presentation time has passed; it interrupts promptly without affecting another sink or waiting behind playback work.
- Adjacent short chunks are scheduled ahead and have no zero-fill gap at their boundary; outstanding handle/notification counts remain bounded and `drain()` waits for the final presentation.
- In both playback paths, a schedule blocked on full capacity is rejected when its generation is stopped and cannot enqueue after capacity is cleared; all pending schedule/notification tasks are awaited before the next generation starts.
- A new generation cannot enter the device/ring buffer before the old one drains or is invalidated, so aborting the old generation never erases already-accepted new-turn audio.
- Simulated ADC/DAC clocks with a non-zero offset still produce correct source-onset/first-audible latency in the injected clock domain.
- Cancellation closes and joins mocked input/output streams and any audio helper thread.
- No real audio dependency is imported by the default package import or `bench`.

### Task 2: Offline STT Endpointing and Non-Blocking faster-whisper (`[whisper]`)

**Files:**

- `src/interpret_live/backends/whisper.py`
- `src/interpret_live/backends/fake.py`
- `src/interpret_live/types.py`
- `src/interpret_live/vad.py`
- `src/interpret_live/pipeline.py`
- `src/interpret_live/config.py`
- `src/interpret_live/model_worker.py` (new)

**Changes:**

1. Add a deterministic `UtteranceEndpointDetector` around the existing `EnergyVAD` with validated defaults:
   - `pre_roll_ms=200`;
   - `partial_interval_ms=500`;
   - `end_silence_ms=500`;
   - `max_utterance_ms=30_000`.
2. Keep consuming the live frame iterator across turns. Start an utterance after speech begins, retain pre-roll, finalize after trailing silence or the maximum duration, then reset only the utterance buffer/state. A maximum-duration split followed by continued speech starts the next turn immediately without requiring an artificial silence gap or dropping the boundary frame.
3. Add backward-compatible optional `source_turn_id` and `speech_started_at_ms` fields to `Hypothesis`; legacy test construction may omit them, but every live STT turn must populate both. Capture the immutable onset from the first VAD-positive input frame, repeat the same values on every partial/final, and make `Pipeline` emit `utterance_start` at that source timestamp rather than first-decode arrival. `FakeSTT` can script both fields for deterministic latency tests, and the fallback for legacy fakes retains current-clock behavior.
4. Resample the continuous microphone stream to 16 kHz with one stateful resampler before endpoint buffering/decoding while preserving duration and monotonic turn-relative timestamps. Flush it only at source EOF and do not reset its phase at ordinary utterance boundaries.
5. Replace the cropped rolling decode window with a bounded full-current-utterance buffer. Every partial must contain the complete current utterance token prefix from index zero so it satisfies `LocalAgreementStabilizer`; word timestamps remain relative to that utterance.
6. Decode no more often than `partial_interval_ms`, plus one final decode at the endpoint. Do not invoke `transcribe()` once per 20 ms audio frame.
7. Run frame ingestion/endpointing and decode-result emission as separate adapter tasks. Ingestion must continue buffering the next turn while a previous final decode is running; results remain ordered so the previous final reaches the pipeline before any next-turn partial. Bound pending turn buffers and surface an explicit overrun metric/error rather than silently losing whole turns when inference cannot keep up.
8. Implement the shared spawned-process worker protocol and construct `WhisperModel` inside a long-lived worker. The child does not independently handle terminal SIGINT; the parent owns cancellation and teardown. Use bounded request/result channels and a one-slot latest-wins partial request policy so slow inference cannot build an unbounded backlog; a final request clears/replaces a queued partial and remains ordered ahead of the next turn.
9. Tag decode requests with the same turn ID plus a generation ID. On cancellation, endpoint reset, or shutdown, set the worker's cooperative cancellation signal, discard stale results, and close the lazy segment generator inside the child process between returned segments. If a native call never returns, `aclose()` terminates and joins the process after the configured grace period.
10. Add explicit async adapter lifecycle: `start()` waits for worker model-load/readiness before audio devices open; `aclose()` is idempotent and follows cooperative cancel -> timed join -> terminate -> timed join -> kill/join, then closes IPC endpoints and joins any queue-feeder/bridge thread within the same budget.
11. Emit zero or more non-final hypotheses and exactly one final hypothesis for every detected non-empty utterance. Do not emit empty/silence-only turns.
12. Make pipeline roll/discard logic turn-aware. A barge-in reset may discard a stale final only when its upstream turn ID matches the interrupted source turn; the first hypothesis/final from the next detected turn must always be processed.
13. Gate the existing pipeline barge-in callback on in-flight MT/TTS or queued/playing target audio. A new source utterance after silence must not abandon itself when there is no old target work to interrupt.
14. Validate language/model/device/compute configuration and give a clear error for unsupported combinations before live capture begins.

**Acceptance:**

- Synthetic silence -> speech -> silence -> speech produces two final hypotheses without ending the source iterator.
- Input consumption continues while a slow final decode runs; the next turn is retained and emitted only after the preceding final.
- Sustained speech across `max_utterance_ms` becomes ordered adjacent turns without losing the split frame.
- An utterance longer than the old `window_ms` retains a stable full prefix and never re-indexes LocalAgreement against a cropped tail.
- Slow mocked transcription does not block an event-loop heartbeat, does not create more than one queued partial request, and yields no stale hypothesis after cancellation.
- Known source-onset timestamps produce latency that includes the partial interval and decode delay instead of starting at first-hypothesis arrival.
- A fake worker whose decode never returns is terminated and reaped within the shutdown budget; no child PID survives adapter close.
- Finalization flushes an unpunctuated trailing clause through the existing pipeline.
- Speech onset with no active target work does not trigger a destructive barge-in roll; onset during in-flight, queued, or playing output still does.
- After barge-in, a new turn that emits only a final hypothesis is translated rather than mistaken for the interrupted turn's stale final.

### Task 3: Non-Blocking NLLB with Non-Repeating Output (`[mt]`)

**Files:**

- `src/interpret_live/backends/nllb.py`
- `src/interpret_live/config.py`

**Changes:**

1. Construct the tokenizer/model and run tokenization, `generate()`, and decoding inside a dedicated long-lived spawned worker process using the shared bounded worker protocol.
2. Translate `segment.text` only. Accept the protocol's `context` argument for compatibility, but document that it is intentionally ignored until a context/output-alignment strategy can prove that only the current segment is returned.
3. Add a cross-process cancellation signal checked through a Transformers `StoppingCriteria` implementation. A cancelled coroutine propagates cancellation without producing a value, and its eventual/stale result never reaches TTS; `aclose()` hard-terminates and reaps the worker if cooperative stopping exceeds the grace period.
4. Validate source/target language codes against the supported FLORES mapping and fail clearly rather than passing an arbitrary incompatible string to the tokenizer.
5. Add validated model ID, device, dtype, maximum input tokens, and maximum output tokens. Reject truncation silently changing a segment; report a typed configuration/runtime error instead.

**Acceptance:**

- Translating segment 2 with segment 1 as context returns only segment 2's translation and TTS never repeats segment 1.
- A slow mocked `generate()` leaves the event loop responsive and honors cancellation without emitting a stale translation.
- Unsupported language/model configuration fails during preflight, before opening the microphone.

### Task 4: Incremental, Interruptible Piper Audio (`[piper]`)

**Files:**

- `src/interpret_live/backends/piper.py`
- `src/interpret_live/config.py`

**Changes:**

1. Derive the native output rate from the loaded Piper voice configuration; remove the unsafe independent `sample_rate=22050` assumption.
2. Load `PiperVoice`, create `synthesize_stream_raw()`, and advance it only inside its dedicated spawned worker process. The parent sends one bounded `NEXT` request at a time and awaits one block result, which provides natural downstream backpressure without a cross-thread producer queue or event-loop model load.
3. Remove `list(...)`. Use one-chunk lookahead so the final produced block alone has `TtsChunk.final=True` while buffering at most one block.
4. Convert Piper PCM16 to the canonical float32 representation while preserving the voice's native rate; `SpeakerSink` performs any device-rate conversion.
5. On coroutine cancellation/barge-in, signal the worker, stop advancing, close the generator inside the child process, and discard any late block tagged with the cancelled utterance ID. If a native Piper/ONNX call does not return within the grace period, terminate and reap the process.
6. Validate voice/model/config compatibility and surface a clear missing/corrupt voice error during preflight.

**Acceptance:**

- The first mocked Piper block is yielded before the mocked generator finishes producing later blocks.
- Cancelling after the first block yields no later block for that utterance and leaves no worker/result task alive.
- A mocked permanently blocked synthesis/load is hard-stopped within the configured shutdown budget and leaves no child process.
- Exactly one block is final, including the single-block case, and its sample rate comes from voice metadata.

### Task 5: Visible Model Preflight, Cache, and Offline Mode

**Files:**

- `src/interpret_live/models.py` (new)
- `src/interpret_live/data/piper_voices.json` (new)
- `src/interpret_live/cli.py`
- `src/interpret_live/config.py`
- `pyproject.toml`

**Changes:**

1. Add a model manager with a platform-appropriate cache root from `platformdirs`, per-artifact `filelock`, temporary-file downloads, and atomic rename. Directly managed Piper files require manifest SHA-256 verification; Hugging Face snapshots use an explicit revision plus the Hub cache's integrity checks. Its output is a typed resolved artifact containing an absolute local snapshot/file path, requested revision, resolved commit/checksum, and provenance.
2. Add direct dependencies only where imported: `platformdirs` and `filelock` in core because the shared model manager imports them, and `huggingface-hub` in the `[whisper]`/`[mt]` extras.
3. Resolve supported faster-whisper aliases to explicit repository plus revision pairs, prefetch them with visible Hugging Face progress, record the resolved commit SHA, and pass only the resolved local snapshot path into the worker's `WhisperModel` construction.
4. Prefetch the configured NLLB repository/revision with visible progress before starting its worker, record the resolved commit SHA, and construct tokenizer/model from that exact local path with `local_files_only=True` (or the library's equivalent). Pass the selected cache directory/revision explicitly so no adapter silently falls back to the global cache or network.
5. Ship a small Piper voice manifest containing the supported language/voice ID, model URL, config URL, SHA-256 values, and license URL. Resolve CLI language/voice choices through this manifest; custom local model/config paths remain supported without download.
6. Add `interpret-live models download` for explicit prefetch and make `run` invoke the same non-interactive preflight automatically when artifacts are missing.
7. Add `--cache-dir` and `--offline`. Offline mode performs no network access and lists every missing artifact in one actionable error. Reject `run --backend cloud --offline` during CLI validation before client construction; the flag governs local model resolution and is not a promise that a cloud provider works without a network.
8. Retry only transient download failures up to three times with bounded exponential backoff. Never retry checksum, authorization, or invalid-manifest failures. Clean temporary/partial files on final failure; concurrent runs either share a completed artifact or wait on its lock.
9. Never load/download models for `bench`, `devices`, `--help`, or a cloud-only run.
10. Complete preflight before starting model workers or opening audio devices. Run blocking Hub/HTTP/cache-library calls in a short-lived spawned preflight process with a bounded result channel. On cancellation, terminate/join that process, release its file lock through process exit, and remove its owned temporary artifacts; do not leave an uncancellable executor future that can hold CLI shutdown open.
11. Include `piper_voices.json` as package data, build a wheel in CI, and prove a clean wheel install can discover the manifest rather than relying on the source tree.

**Acceptance:**

- First-run downloads show artifact name, size/progress, destination, and completion; a second run performs no network request.
- Interrupted/corrupt downloads are not treated as valid cache entries.
- Offline mode is verified with network calls forbidden and produces one complete missing-artifact report.
- With network access forbidden, the complete offline runtime factory resolves cached paths, starts all adapter workers, constructs their model/tokenizer/voice objects from those paths, and reaches readiness without a hidden Hub/HTTP request.
- The cloud/offline CLI combination fails validation before opening a connection or device.
- Piper voices are checksum-verified and their license/source is discoverable from CLI output/documentation.

### Task 6: Persistent S2S Protocol and Post-Interrupt Recovery

**Files:**

- `src/interpret_live/types.py`
- `src/interpret_live/backends/__init__.py`
- `src/interpret_live/s2s.py`
- `src/interpret_live/backends/fake.py`
- `src/interpret_live/backends/gemini.py`
- `src/interpret_live/session.py`
- `src/interpret_live/vad.py`
- `src/interpret_live/metrics.py`

**Changes:**

1. Keep `S2SBackend` in `session.py`; do not create a second wrapper in `realtime.py`.
2. Refine the provider protocol for a session-long connection:
   - `stream(audio)` no longer receives a fixed pipeline utterance ID;
   - it yields a typed `S2SEvent` union rather than audio alone: input speech started/committed, response started, `S2SAudioChunk`, content-audio done, and response done with status/reason;
   - `S2SAudioChunk` contains float32 samples, rate, provider response/item IDs, output/content indexes, and content-final status;
   - `PlaybackCursor(response_id, item_id, content_index, audio_end_ms)` always scopes heard audio to one response;
   - `interrupt(S2SInterruptTarget(response_id, cursor | None))` always names the snapshotted response even when no audio has yet been heard.
3. Make input speech-start a required provider control event carrying provider input-item ID plus `source_started_at_ms` already translated into the injected application `Clock` domain. `S2SPipeline` creates the local utterance and records `utterance_start` from that event. Local `EnergyVAD` remains authoritative only for immediate barge-in; its thresholds/turn count are never required to match provider VAD segmentation.
4. Let `S2SPipeline` maintain explicit provider input-item/response-ID -> local-utterance-ID maps from provider speech-start/commit/response-start events. Stamp audio when it enters the playback queue. A completed response closes generation state without losing queued/playing chunk ownership, and two later responses retain independent start/first-audio metrics. Duplicate provider events are idempotent; an unknown response/input reference surfaces a protocol error, but ordinary local/provider VAD boundary disagreement does not.
5. Define lifecycle semantics precisely: content-audio done closes only that response content stream. Only `response.done` with `status=completed` is a natural response completion. Expected `cancelled` for an abandoned response is an interrupt acknowledgement; `failed`/unexpected `cancelled`/`incomplete` surface typed outcomes and never emit a natural final or roll a fresh turn.
6. Split provider receive and speaker scheduling/receipt collection into separate bounded tasks. Keep bounded scheduled lookahead for gapless audio; an interrupt must remain observable while handles are queued or presenting, not only while waiting for the next provider event. Emit `first_tts_out` from the first handle's `started` receipt for the response's mapped local turn, never on queue insertion.
7. Protect response/playback ownership with one pipeline state lock. On local barge-in:
   - under the lock, atomically mark the current response interrupting/abandoned, pause the scheduling task, and snapshot its response/item/content identity before any await; the receiver may continue filling its bounded event queue, but no newer response reaches the sink or replaces the target until cleanup completes;
   - call `sink.stop(old_playback_generation)` through its independent abort path so the generation is invalid before capacity is released; combine the returned presented-sample snapshot with the saved identity into a clamped `PlaybackCursor`;
   - cancel and await every pending schedule/receipt task for that generation; typed generation rejections are expected, and no fresh generation is issued until they have settled;
   - drain queued audio for only the interrupted response/utterance;
   - cancel/truncate exactly the saved response once via `S2SInterruptTarget`, even if that response finishes and a newer response starts during sink shutdown;
   - emit interrupt/sink metrics;
   - clear/re-arm local interrupt state;
   - do not allocate or roll a speculative new turn from the local barge-in onset; the provider's next mapped speech-start event creates the fresh utterance exactly once, so local/provider VAD disagreement cannot skip or double-roll speech;
   - resume scheduling so queued events for later non-abandoned responses can proceed;
   - continue the same provider/source session so later speech produces audio.
8. Build `audio_end_ms` only from completed handle receipts plus the atomic stop snapshot, accumulated at the source chunk rate per response item/content index; include neither scheduled-but-unpresented frames nor device-buffer frames whose DAC time is still in the future.
9. Discard late delta/content-done/response-done events for an abandoned response, and ensure its eventual cancelled `response.done` cannot roll the fresh local utterance a second time.
10. Gate cloud barge-in on an in-progress response or queued/playing output just like the pipeline path; source speech with nothing to interrupt remains an ordinary provider turn.
11. Update `FakeS2S` to model persistent multiple input turns/responses, all control events/statuses, provider metadata, targeted cancellation/truncation calls, late old-response events, and post-interrupt output.
12. Migrate the out-of-scope `GeminiS2S` scaffold to the new method signatures/event types (still raising its clear not-implemented error) and add a static protocol-conformance check so it cannot silently drift behind `S2S`.

**Acceptance:**

- Barge-in while a scheduled playback handle is active stops promptly, resolves queued handles as interrupted, sends exactly one provider interrupt, and discards all queued old-utterance chunks.
- With a partially presented chunk and simulated device latency, the cursor includes exactly the sink-reported presented samples and excludes queued/device-buffered samples.
- If an old response finishes and a new response starts while `sink.stop()` runs, cancellation still names only the old response ID.
- A provider `response.done` received before local playback completes preserves the queued chunks' utterance/item ownership and still permits correct later truncation.
- Two source turns with known onset and first-presentation timestamps produce two independent, non-zero first-audio latency values.
- Deliberately different local-barge-in and provider-VAD segmentation still maps provider turns and metrics without terminating the session or inventing extra turns.
- Completed, cancelled, failed, and incomplete response statuses follow the defined roll/error behavior; audio-done alone never closes a response naturally.
- Speech after interruption produces audio under a new utterance ID without reconnecting or exhausting the source.
- Late audio/final events from the cancelled response are ignored and do not cause a second utterance roll.
- Speech with no in-progress response or queued/playing provider output does not invoke provider cancellation.
- Cancellation, source EOF, provider failure, and normal completion leave no receiver/playback tasks behind.

### Task 7: OpenAI Realtime Transport (`[openai]`)

**Files:**

- `src/interpret_live/backends/realtime.py`
- `src/interpret_live/config.py`
- `pyproject.toml`

**Changes:**

1. Implement `RealtimeS2S`, not `S2SBackend`, using `AsyncOpenAI.realtime.connect()` and the refined S2S protocol.
2. At implementation time, select a currently supported Realtime model as the tested default, keep it configurable, and set the project `[openai]` extra to `openai[realtime]>=<lowest-version-exercised-by-adapter-CI>` plus `soxr`. Commit a concrete lower bound proven by CI; do not leave a placeholder, retain the old preview model, or use plain `openai>=1.40`. The SDK's `realtime` extra is required because it installs the WebSocket transport used by `realtime.connect()`.
3. Configure the Realtime session with:
   - translation-only instructions containing source and target language;
   - audio output modality;
   - `session.audio.output.voice` from validated provider-specific CLI/configuration before the first response;
   - 24 kHz PCM input/output;
   - server VAD response creation enabled;
   - server-VAD idle timeout/unsolicited empty-turn response disabled so every response maps to a tracked speech turn or explicit EOF commit;
   - server automatic response interruption disabled so local barge-in owns cancellation.
4. Run concurrent connection tasks:
   - input encoder: use one stateful 24 kHz resampler for the continuous source, convert to little-endian PCM16/base64, and enqueue bounded append commands; maintain a cumulative sent-audio ledger mapping provider-buffer sample/millisecond ranges back to original `AudioFrame` timestamps in the injected `Clock` domain, prune fully resolved committed ranges while retaining the cumulative offset/base anchor, bound it by unresolved audio, and flush exactly once at EOF;
   - one serialized outbound command pump: it is the only task allowed to write to the WebSocket, applies a bounded timeout to each small send, preserves append order, and sends cancel+truncate as one prioritized, non-interleavable interrupt group immediately after any already-started send while new input remains bounded; EOF commit/create follows the acknowledged state machine below rather than an unsafe blind pair;
   - receiver: process server events and put typed input-turn/response lifecycle events plus decoded `S2SAudioChunk`s from `response.output_audio.delta` into a bounded internal queue;
   - public async generator: yield from that queue, applying bounded backpressure to the receiver while input encoding remains independently cancellable; cancellation closes the encoder, writer, receiver, and generator paths.
5. Map `input_audio_buffer.speech_started.audio_start_ms` through the sent-audio ledger to the corresponding original source timestamp and emit that mapped value with the provider input-item ID. Map input commit, response created, output-audio delta/done, and response done into the remaining protocol events from Task 6. Preserve response/item/output/content indexes for every delta. Treat output-audio done only as content completion; inspect `response.done.status`/status details to distinguish completed, cancelled, failed, and incomplete outcomes.
6. Treat server `error` events as typed failures even though the SDK keeps the socket open. Give every cancel/truncate client event a unique `event_id`, retain a pending-control map, and tolerate a no-active/already-completed response error only when the server error references the matching targeted cancel event. Unrelated errors remain fatal. Handle rate limits, disconnects, and malformed/base64-invalid audio without leaking tasks.
7. Do not transparently reconnect after audio has been sent because replay could duplicate or lose speech; stop playback and fail the session clearly. A bounded reconnect is allowed only if the initial connection fails before the first input frame is sent.
8. Implement `interrupt(target)` through the serialized command pump:
   - send `response.cancel(response_id=target.response_id)` for exactly the snapshotted response, never an unqualified cancel;
   - when `target.cursor` exists, send `conversation.item.truncate` for its exact item/content/heard duration immediately after that cancel in the same prioritized command group;
   - correlate the documented no-active-response error by the cancel's client `event_id` without closing an otherwise healthy session or swallowing other errors;
   - keep the connection available for the next server-VAD turn.
9. Implement an EOF state machine under the same connection-state lock: `empty`, `speech_pending`, `auto_committed`, `manual_commit_sent`, `manual_committed`, `response_started`, `response_done`. After the final resampler flush:
   - if no speech/uncommitted audio exists, close without commit/create;
   - if an automatic commit/response is already observed, never send a manual duplicate;
   - if speech is pending, allow a bounded VAD-settle window while the receiver continues processing; then send one client-ID-tagged manual commit only if still pending;
   - await the commit acknowledgement before enqueueing exactly one `response.create`; if an automatic commit/response wins the race, suppress the create and treat only the matching empty-buffer error for the superseded manual commit as benign;
   - await the single final response or a bounded timeout, then close the connection/client cleanly.
10. Validate implementation behavior against the current [official Realtime event reference](https://developers.openai.com/api/reference/resources/realtime) and pin the tested dependency shape from the [official Python SDK metadata](https://github.com/openai/openai-python/blob/main/pyproject.toml) in the same change.

**Acceptance:**

- A scripted fake SDK connection proves exact session configuration (including selected voice), stateful input PCM conversion, event/status mapping, error propagation, response-ID-scoped cancel/truncate ordering, and continued output after interruption.
- When an interrupt races an in-progress append, release of that append is followed immediately by the contiguous cancel/truncate group with no newer append interleaved; a send that exceeds its timeout fails/closes the session without a concurrent WebSocket send.
- A control error referring to the expected already-finished cancel event is tolerated, while an unrelated/error-mismatched `event_id` fails the session.
- Scripted EOF cases for no speech, auto-commit before EOF, auto-commit racing the settle/manual commit, and genuinely uncommitted speech produce respectively zero or exactly one final response with no fatal empty-buffer race.
- Server speech-start offsets map through the sent-audio ledger to original source-clock timestamps even when local EnergyVAD deliberately segments the same frames differently.
- A response cancelled before any audio was heard sends cancel without an invalid truncate.
- API keys never appear in logs, exceptions, snapshots, or test fixtures.
- No code writes directly to `AudioSink`; all output remains under `S2SPipeline` control.

### Task 8: Runtime Composition, CLI, and Dual Direction

**Files:**

- `src/interpret_live/runtime.py` (new)
- `src/interpret_live/cli.py`
- `src/interpret_live/config.py`
- `src/interpret_live/session.py`

**Changes:**

1. Add provider-specific runtime factories that construct validated, independently owned resources in this order: validate configuration -> resolve/download models -> start and health-check model workers/provider clients -> open audio devices -> enter the `Session`:
   - offline: `WhisperSTT` + `NllbMT` + `PiperTTS` -> `PipelineBackend`;
   - OpenAI: `RealtimeS2S` -> existing `S2SBackend`;
   - real `MicSource`, `SpeakerSink`, `RealClock`, configuration, and `Session`.
2. Replace the placeholder `run` command with `asyncio.run()` over the built session. Preserve clear missing-extra errors, but return success only after a session actually ran or the user stopped it normally.
3. Expose validated options for backend/provider, source/target language, offline model/voice IDs, OpenAI output voice, cache/offline mode, input/output devices, and relevant VAD/audio settings. Keep credentials environment-only, reject cloud plus `--offline`, and reject provider-specific voice choices that the selected backend cannot consume.
4. Use `AsyncExitStack` and signal/Ctrl-C handling so devices, adapter `aclose()` methods, child workers, and clients close in reverse ownership order. For every worker, request cooperative shutdown, wait the bounded grace period, then terminate/join and kill/join if necessary; never call an executor shutdown that can wait forever. Print a concise final metrics summary after normal stop.
5. For `--dual`, require/accept explicit A/B input and output device selections and build two independent directional backends:
   - A -> B uses source -> target and the target voice;
   - B -> A uses target -> source and the source voice;
   - OpenAI uses two independent Realtime connections;
   - stateful STT/MT/TTS/provider objects are never shared between directions.
6. Refactor `DualChannel.create()` to accept two already-built backends or a per-direction factory instead of reusing one stateful backend instance. Retain capability negotiation for both directions.
7. Fail before opening devices when model/voice/language/provider configuration is incomplete. If the same physical device is explicitly reused across directions, warn about likely feedback/cross-talk.

**Acceptance:**

- A CLI composition test proves `run --backend offline` builds and awaits a real `Session` using injected adapter/device doubles rather than printing hints.
- The cloud CLI path builds the existing `S2SBackend` around `RealtimeS2S` and reads its key only from the environment.
- Dual construction creates distinct backend/provider identities with reversed language directions and correct sink routing.
- Ctrl-C during model inference, provider receive, or speaker playback exits cleanly with no leaked tasks or open mocked resources.
- Ctrl-C with a model worker intentionally stuck forever exits within the configured budget and leaves no child PID, audio helper thread, or open mocked resource.

### Task 9: Tests, CI, Documentation, and Release Gate

**Files:**

- `tests/test_audio_codec.py` (new)
- `tests/test_audio_io_real.py` (new, mocked `sounddevice`)
- `tests/test_model_worker.py` (new, spawned cooperative/stuck workers)
- `tests/test_whisper_adapter.py` (new, mocked model-process boundary)
- `tests/test_nllb_adapter.py` (new)
- `tests/test_piper_adapter.py` (new)
- `tests/test_models.py` (new, fake downloader/cache)
- `tests/test_realtime_adapter.py` (new, scripted fake SDK connection)
- `tests/test_runtime.py` (new, injected factories/resources)
- `tests/test_s2s.py`
- `tests/test_metrics.py`
- `tests/test_cli.py`
- `tests/test_session_dual.py`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `README.md`
- `docs/live-adapters.md` (new)

**Changes:**

1. Add the contract/integration tests listed in Tasks 1-8. All PR tests mock hardware, model computation, downloads, and OpenAI transport; no secret or network is required.
2. Replace the CLI tests that expect placeholder hints with composition, validation, missing-extra, offline-cache, and cleanup assertions.
3. Remove only `whisper.py`, `nllb.py`, `piper.py`, `realtime.py`, and `audio_io.py` from `coverage.run.omit` as their mocked tests land. Remove obsolete class/method-level `# pragma: no cover` exclusions from the mocked in-scope `MicSource`, `SpeakerSink`, and Realtime methods as well; retain exclusions only on genuinely unreachable hardware/platform branches with an inline reason. Leave the out-of-scope Gemini/ElevenLabs file omissions unchanged, verify the coverage report now lists every in-scope adapter class/method, and keep the threshold at or above 90%.
4. Keep the current Python 3.11/3.12/3.13 core matrix and add:
   - one Python 3.12 adapter-contract job installing `[whisper,mt,piper,audio,openai,dev]` without downloading models, asserting `openai`, `websockets`, and every declared optional transport import is present before tests;
   - a scheduled/manual 3.11/3.13 optional-extra compatibility matrix if the full dependency install is too expensive for every PR;
   - dependency caching, with all network/model download calls disabled during tests.
5. Add an opt-in manual/live smoke marker for maintainers with local models/hardware and an OpenAI key; never run it in ordinary CI.
6. Document exact setup, model prefetch, cache/offline behavior, device selection, single/dual commands, expected output, Ctrl-C behavior, and troubleshooting.
7. Update the README capability matrix and status checkboxes only after both live paths pass their contract tests. Correct install examples so a microphone-based OpenAI run includes both `[openai]` and `[audio]`.
8. Add deterministic multi-turn metric coverage for both paths: scripted source speech-onset timestamps and sink first-presentation timestamps must yield the expected non-zero `first_audio_out_ms` for each turn, including offline decode delay and cloud response delay.
9. Add protocol/lifecycle coverage for Gemini's migrated scaffold and every OpenAI response status, an interleaved old/new response cancellation race, outbound single-writer priority, all four EOF commit races, sent-audio timestamp mapping under mismatched local/provider VAD segmentation, gapless bounded lookahead across adjacent chunks, a capacity-blocked stale schedule racing generation stop in both playback paths, partial heard-audio accounting with device latency, stateful chunked resampling, fully offline adapter construction, and a worker that ignores cooperative shutdown forever.
10. Exercise pure worker request handlers in-process for deterministic branch coverage and keep separate real-`spawn` lifecycle tests. If worker code is measured only in children, enable coverage multiprocessing startup/parallel data and combine files before enforcing 90%; do not silently exclude `model_worker.py` because subprocess coverage was not collected.

**Required verification commands:**

```bash
ruff check .
ruff format --check .
mypy src
pytest -q --cov=interpret_live --cov-report=term-missing --cov-fail-under=90
```

**Manual smoke gates:**

```bash
interpret-live models download --backend offline --from en --to es
interpret-live run --backend offline --from en --to es --voice <piper-voice> --input-device <id> --output-device <id>
# with OPENAI_API_KEY already exported in the environment
interpret-live run --backend cloud --provider openai --from en --to es --voice <openai-voice> --input-device <id> --output-device <id>
```

For each live smoke, verify: audible translated output, no repeated prior segment, visible first-audio latency, barge-in stops old audio, speech resumes after the interrupt, Ctrl-C closes devices promptly, and no traceback/background task remains.

For the offline smoke, also inspect the process table after Ctrl-C and confirm no model-worker children remain. For the OpenAI smoke, verify the configured output voice is audibly/diagnostically selected and no API key or raw authorization header appears in debug logs.

## Release Acceptance Criteria

The plan is complete only when all of the following are true:

- `interpret-live run` executes a live session for offline and OpenAI modes; it no longer exits successfully after printing only hints.
- Internal audio remains mono float32, and all 16/22.05/24/48 kHz boundary conversions are explicit and tested.
- A continuous microphone stream produces multiple finalized Whisper utterances with full-prefix LocalAgreement-compatible hypotheses and bounded memory/work queues.
- Whisper, NLLB, and Piper never perform blocking model work on the event loop; their spawned workers are cooperatively stopped or hard-reaped within a bounded shutdown budget, and stale results are discarded.
- NLLB never returns/re-speaks prior rolling context.
- Piper yields its first audio block before synthesis completes and stops yielding cancelled-utterance blocks.
- Model downloads are visible, atomic, cached, checksum-verified where artifacts are directly managed, and completely disabled by `--offline`.
- Offline and cloud first-audio latency starts at actual source speech onset and ends at the sink's first presented sample for every turn.
- OpenAI uses the existing `S2SBackend`/`S2SPipeline`, maps the official event/status lifecycle, serializes all outbound writes, sends response-ID-scoped cancel plus sink-reported heard-audio truncation when applicable, and continues after interruption.
- The `[openai]` extra installs the SDK's Realtime transport dependency, and the configured OpenAI voice reaches `session.audio.output.voice`.
- Dual mode creates two independent stateful backend/provider stacks.
- Normal PR CI covers live adapter code through mocks, installs optional dependencies in a dedicated contract job, and retains the coverage gate.
- The no-extras deterministic `bench` remains offline, fast, and behaviorally unchanged.

## Finding Coverage

| Reviewed finding | Addressed by |
|---|---|
| Stale adapter baseline | Current Baseline; Tasks 2-4 and 7 describe actual deltas |
| No executable live CLI path | Task 8 |
| Undefined Whisper/VAD utterance lifecycle | Task 2 |
| Blocking inference defeats cancellation | Architecture decisions; Tasks 2-4 |
| Conflicting audio formats/sample rates | Task 1 |
| Wrong cloud abstraction and no post-interrupt recovery | Tasks 6-7 |
| Incomplete OpenAI event/cancel/truncate lifecycle | Tasks 6-7 |
| NLLB repeats rolling context | Task 3 |
| Vague model acquisition/cache behavior | Task 5 |
| Mock-only testing and excluded coverage | Task 9 |
| Stateful dual backends are shared | Task 8 |
| Ctrl-C can hang on non-cooperative executor work | Architecture decision 3; Tasks 2-4 and 8-9 |
| Heard-audio cursor can include queued/device-buffered samples | Tasks 1, 6, and 9 |
| Cancel can race and target a newer response | Tasks 6-7 and 9 |
| S2S/offline first-audio latency starts too late | Tasks 2, 6, and 9 |
| Realtime transport extra and provider voice are unwired | Tasks 7-9 |
| Realtime response statuses/control errors are ambiguous | Tasks 6-7 and 9 |
| Per-block resampling resets phase | Tasks 1-2, 7, and 9 |
| Cached prefetch can still fall back to network | Tasks 5 and 9 |
| Gemini scaffold drifts from the shared S2S protocol | Tasks 6 and 9 |
| Cloud `--offline` is ambiguous | Tasks 5, 8, and 9 |
| In-scope coverage remains hidden by pragmas | Task 9 |
| Waiting for audible chunk completion causes playback gaps | Tasks 1, 6, and 9 |
| Local/provider VAD mismatch corrupts cloud turn mapping | Tasks 6-7 and 9 |
| EOF auto/manual commit race can duplicate responses | Tasks 7 and 9 |
| PortAudio and application clocks are incomparable without calibration | Architecture decision 9; Tasks 1-2 and 6-7 |
| Capacity-blocked schedule can enqueue stale audio after stop | Tasks 1, 6, and 9 |
