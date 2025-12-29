# faster-whisper Deep Documentation

**What this is**: A fast Python implementation of OpenAI's Whisper speech recognition model using CTranslate2 for inference.

**Problem solved**: Whisper is accurate but slow. This library wraps Whisper models converted to CTranslate2 format, achieving 2-4x speedup over the original with lower memory usage, while supporting 8-bit quantization for further efficiency.

## Navigation

### Core Concepts
- [Architecture Overview](core-concepts/architecture.md) - High-level structure, key components, data flow
- [Domain Model](core-concepts/domain-model.md) - Core types, relationships, invariants

### Features (Deep Dives)
- [Transcription Pipeline](features/transcription/flow.md) - Complete traced flow from audio to segments
- [Batched Inference](features/batched-inference/overview.md) - How batched transcription works

### Boundaries
- [CTranslate2 Integration](boundaries/ctranslate2.md) - How the library interfaces with the inference engine
- [Audio Decoding](boundaries/audio.md) - PyAV integration for audio processing
- [VAD Integration](boundaries/vad.md) - Silero VAD for speech detection
- [Model Loading](boundaries/model-loading.md) - HuggingFace Hub integration

### Open Questions
- [Open Questions](open-questions.md) - Unresolved areas, potential improvements

## Quick Mental Model

```
Audio File/Array
      │
      ▼
┌─────────────────┐
│  decode_audio   │  (PyAV - FFmpeg bindings)
│  → 16kHz float32│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Silero VAD     │  (optional - removes silence)
│  → speech chunks│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FeatureExtractor│  (Mel spectrogram)
│  → (80, frames) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CTranslate2    │  (Whisper encoder/decoder)
│  Whisper Model  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tokenizer     │  (decode tokens → text)
└────────┬────────┘
         │
         ▼
   Segment(text, start, end, words)
```

## Key Files

| File | Purpose |
|------|---------|
| `faster_whisper/transcribe.py` | Main logic - WhisperModel, BatchedInferencePipeline |
| `faster_whisper/audio.py` | Audio decoding via PyAV |
| `faster_whisper/feature_extractor.py` | Mel spectrogram computation |
| `faster_whisper/tokenizer.py` | Token encoding/decoding wrapper |
| `faster_whisper/vad.py` | Silero VAD integration |
| `faster_whisper/utils.py` | Model downloading, utilities |
