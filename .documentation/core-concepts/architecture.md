# Architecture Overview

faster-whisper is a thin wrapper around CTranslate2's Whisper implementation. The library's value-add is:

1. **Easy model loading** from HuggingFace Hub
2. **Audio preprocessing** (decoding, resampling, feature extraction)
3. **VAD integration** for filtering silent audio
4. **Word-level timestamps** via cross-attention alignment
5. **Batched inference** for throughput-oriented use cases

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Code                                   │
│   model = WhisperModel("large-v3")                                  │
│   segments, info = model.transcribe("audio.mp3")                    │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         transcribe.py                                │
│                                                                      │
│  ┌──────────────────┐  ┌─────────────────────────────┐              │
│  │   WhisperModel   │  │  BatchedInferencePipeline   │              │
│  │  (sequential)    │  │  (batch parallel)           │              │
│  └────────┬─────────┘  └──────────────┬──────────────┘              │
│           │                           │                              │
│           └───────────────┬───────────┘                              │
│                           │                                          │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Core Processing                             │ │
│  │  audio.py → feature_extractor.py → ctranslate2 → tokenizer.py │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      External Dependencies                           │
│                                                                      │
│  ┌─────────────┐  ┌────────────────┐  ┌────────────────────────┐   │
│  │    PyAV     │  │  CTranslate2   │  │     onnxruntime        │   │
│  │(FFmpeg bind)│  │(Whisper model) │  │    (Silero VAD)        │   │
│  └─────────────┘  └────────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Two Transcription Modes

### 1. WhisperModel.transcribe() - Sequential Processing

The original Whisper approach: process 30-second windows sequentially, with autoregressive decoding conditioning on previous output.

**Why use it:**
- When context continuity matters (previous text influences next segment)
- When memory is constrained
- For streaming-like scenarios

**Key flow** (transcribe.py:747-1022):
```
for each 30-sec window:
    encode(features) → encoder_output
    generate_with_fallback(encoder_output, prompt) → tokens
    split_segments_by_timestamps(tokens) → segments
    if word_timestamps: add_word_timestamps(segments)
    yield Segment
```

### 2. BatchedInferencePipeline.transcribe() - Parallel Processing

Newer addition: use VAD to find speech chunks, process multiple chunks in parallel.

**Why use it:**
- 3-5x faster for long audio
- When throughput matters more than strict sequential consistency

**Key flow** (transcribe.py:254-578):
```
VAD → speech_chunks
collect_chunks(audio, speech_chunks) → batches
for each batch:
    forward(features_batch) → segmented_outputs
    yield Segment
restore_speech_timestamps(segments, speech_chunks)
```

## Non-Obvious Architectural Decisions

### 1. Generator-Based Segments

Both `transcribe()` methods return `Iterable[Segment]`, not `List[Segment]`. This is intentional:

- **Lazy evaluation**: Transcription only runs when you iterate
- **Memory efficiency**: Don't hold all segments in memory
- **Streaming potential**: Could process output while transcription continues

**Gotcha** (documented in README):
```python
segments, _ = model.transcribe("audio.mp3")
segments = list(segments)  # Actually runs transcription here
```

### 2. Temperature Fallback

If decoding fails quality thresholds (compression ratio, log probability), the model retries with increasing temperature (transcribe.py:1432-1520).

Default temperatures: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`

This mimics OpenAI's original behavior: deterministic first, increasingly stochastic if that fails.

### 3. No FFmpeg Dependency

Unlike openai-whisper, this library uses PyAV which bundles FFmpeg libraries. This is a deliberate choice to reduce system dependencies.

**Trade-off**: Slightly slower audio loading (see transcribe.py comment about gc.collect() workaround for memory leak).

### 4. Bundled VAD Model

The Silero VAD ONNX model is distributed with the package (`faster_whisper/assets/silero_vad_v6.onnx`). This avoids runtime downloads but increases package size.

## File Responsibilities

| File | LOC | Responsibility |
|------|-----|----------------|
| transcribe.py | ~1940 | Core transcription logic, model management |
| audio.py | ~125 | Audio decoding (PyAV wrapper) |
| feature_extractor.py | ~230 | Mel spectrogram computation |
| tokenizer.py | ~320 | Token encode/decode, special tokens |
| vad.py | ~385 | Speech detection, timestamp restoration |
| utils.py | ~150 | Model download, formatting helpers |
