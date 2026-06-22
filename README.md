# interpret-live

> A real-time speech-to-speech translator CLI — you speak one language, the other side hears another, sub-second latency with barge-in, cloud or fully offline.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

A simultaneous-interpretation CLI that streams live STT -> incremental MT -> streaming TTS in the target language, starting before the speaker finishes the sentence. Partial-hypothesis stabilization is pushed into the audio output stage so synthesized speech doesn't stutter as ASR revises words.

## Why

Streaming speech-to-speech is fragmented across text-only tools, research models, and vendor-locked demos. This is one polished CLI that runs the same pipeline on cloud realtime APIs or fully offline.

## Features

- Streaming STT -> incremental MT -> streaming TTS, started before the sentence ends
- Partial-stabilization at the audio stage + barge-in/VAD interrupt handling
- Pluggable backend: cloud S2S (OpenAI Realtime / Gemini Live) **and** fully offline (faster-whisper + local MT + Piper)
- Dual-channel call/meeting mode with independent inbound/outbound voices
- Optional speaker voice-preservation in the target language (ElevenLabs)

## How it works

Run the CLI with source/target languages and a backend. It captures audio, transcribes incrementally, translates on partial hypotheses, and speaks the target language with sub-second latency, halting instantly on barge-in.

## Tech stack

- Python
- faster-whisper
- NLLB / Madlad MT
- Piper TTS
- OpenAI Realtime API
- Gemini Live API
- ElevenLabs
- WebRTC VAD

## Status & roadmap

🚧 **Early development.** This repository is being built in the open; the scaffold and design are in place and the implementation is landing incrementally.

- [ ] Offline pipeline: faster-whisper -> local MT -> Piper with audio-stage stabilization
- [ ] Barge-in/VAD interrupt handling
- [ ] Cloud S2S backend (OpenAI Realtime / Gemini Live) behind one interface
- [ ] Dual-channel meeting mode; optional voice preservation

## Installation

> Coming soon.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
