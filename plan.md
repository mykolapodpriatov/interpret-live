# Implementation Plan: Wiring Live Adapters & Cloud Providers

## Issues to Address
1. **Incomplete Offline STT Adapter:** The `whisper.py` file exists but needs to integrate `faster-whisper` and yield incremental `Hypothesis` objects.
2. **Incomplete Offline MT Adapter:** The `nllb.py` file needs the actual `transformers` implementation for local machine translation of complete segments.
3. **Incomplete Offline TTS Adapter:** The `piper.py` file needs to be wired up with `piper-tts` to synthesize text into audio streams locally.
4. **Wiring Real Audio I/O:** We have `MicSource` and `SpeakerSink` in `audio_io.py`, but we need to ensure they seamlessly plug into the `Session` alongside the fakes.
5. **Incomplete Cloud S2S Implementation:** The `realtime.py` file exists but needs the actual wrapper for OpenAI Realtime to support the cloud path.

## Proposed Plan

### Task 1: Real Audio I/O (`[audio]`)
- Audit the existing `MicSource` and `SpeakerSink` in `src/interpret_live/audio_io.py`.
- **Audio Configuration:** Verify they expose and enforce parameters for sample rate (defaulting to 16kHz or 24kHz), bit depth (16-bit PCM), and channels (mono), to ensure strict downstream compatibility.
- Ensure graceful shutdown and device enumeration work robustly across platforms.

### Task 2: Offline STT Adapter (`[whisper]`)
- Implement the `faster-whisper` worker thread/process in `src/interpret_live/backends/whisper.py`.
- **VAD Integration:** Since `faster-whisper` expects discrete audio segments, ensure the existing or a lightweight VAD (like Silero VAD) chunks the continuous stream before feeding it to the transcriber.
- Ensure the adapter yields partials as the correct `Hypothesis` type so the `LocalAgreement` stabilizer can consume them.

### Task 3: Offline MT and TTS (`[mt]`, `[piper]`)
- Implement the local MT logic in `src/interpret_live/backends/nllb.py` using `transformers`.
- Implement the local TTS logic in `src/interpret_live/backends/piper.py` using `piper-tts`.
- **TTS Execution Model:** Ensure `piper.py` streams synthesized audio incrementally (chunk-by-chunk) as soon as tokens are ready, rather than blocking until the entire segment is synthesized.
- **Model Downloads:** Implement a graceful pre-flight downloading and caching strategy for `transformers` and `piper-tts` models, so the first run shows a clear progress bar instead of blocking invisibly.

### Task 4: Cloud S2S Backend (`[openai]`)
- Implement the `S2SBackend` wrapper inside `src/interpret_live/backends/realtime.py` to connect directly from the audio source to sink through WebSocket, handling server-side VAD and interrupts.

### Task 5: Testing
- Create or update unit tests for the backend adapters (`whisper.py`, `nllb.py`, `piper.py`, `realtime.py`) with mocked underlying models to avoid slow runs.
- Add hardware-mocking strategies for `MicSource` and `SpeakerSink` so the I/O pipeline can be fully exercised in CI.
