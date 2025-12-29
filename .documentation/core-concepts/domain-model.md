# Domain Model

## Core Types

### Output Types (what users receive)

```python
@dataclass
class Segment:
    id: int                          # 1-indexed segment number
    seek: int                        # Frame position in audio
    start: float                     # Start time (seconds)
    end: float                       # End time (seconds)
    text: str                        # Transcribed text
    tokens: List[int]                # Token IDs
    avg_logprob: float               # Decoding confidence
    compression_ratio: float         # Text compression ratio (hallucination indicator)
    no_speech_prob: float            # Probability of no speech
    words: Optional[List[Word]]      # Word-level timestamps (if requested)
    temperature: Optional[float]     # Sampling temperature used
```
Location: transcribe.py:47-68

```python
@dataclass
class Word:
    start: float                     # Word start time (seconds)
    end: float                       # Word end time (seconds)
    word: str                        # The word text
    probability: float               # Confidence score
```
Location: transcribe.py:31-44

```python
@dataclass
class TranscriptionInfo:
    language: str                    # Detected/specified language code
    language_probability: float      # Language detection confidence
    duration: float                  # Original audio duration
    duration_after_vad: float        # Duration after VAD filtering
    all_language_probs: Optional[List[Tuple[str, float]]]  # All language scores
    transcription_options: TranscriptionOptions
    vad_options: VadOptions
```
Location: transcribe.py:100-108

### Configuration Types

```python
@dataclass
class TranscriptionOptions:
    beam_size: int                   # Beam search width (default: 5)
    best_of: int                     # Candidates for sampling (default: 5)
    patience: float                  # Beam search patience
    length_penalty: float            # Exponential length penalty
    repetition_penalty: float        # Penalty for repeated tokens
    no_repeat_ngram_size: int        # Prevent ngram repetitions
    log_prob_threshold: Optional[float]  # Quality threshold
    no_speech_threshold: Optional[float] # Silence detection threshold
    compression_ratio_threshold: Optional[float]  # Hallucination detection
    condition_on_previous_text: bool # Use previous output as context
    temperatures: List[float]        # Fallback temperatures
    initial_prompt: Optional[Union[str, Iterable[int]]]  # Prompt text/tokens
    # ... and more
```
Location: transcribe.py:70-97

```python
@dataclass
class VadOptions:
    threshold: float = 0.5           # Speech probability threshold
    neg_threshold: float = None      # Silence threshold (default: threshold - 0.15)
    min_speech_duration_ms: int = 0  # Minimum speech chunk length
    max_speech_duration_s: float = inf  # Maximum speech chunk length
    min_silence_duration_ms: int = 2000  # Silence needed to split
    speech_pad_ms: int = 400         # Padding around speech chunks
```
Location: vad.py:14-48

## Key Invariants

### 1. Segment Ordering

Segments are yielded in temporal order. For sequential transcription, `segment[i].end <= segment[i+1].start`. Tests verify this: test_transcribe.py:247-271

### 2. Word Timestamps Consistency

When `word_timestamps=True`:
- `segment.text == "".join(word.word for word in segment.words)` (test_transcribe.py:40)
- `segment.start == segment.words[0].start` (test_transcribe.py:41)
- `segment.end == segment.words[-1].end` (test_transcribe.py:42)

### 3. Feature Shape Convention

Audio features are `(n_mels, n_frames)` = `(80, frames)` by default. The encoder expects exactly 3000 frames (30 seconds at 16kHz with hop_length=160).

```python
# feature_extractor.py:16-17
self.n_samples = chunk_length * sampling_rate      # 30 * 16000 = 480000
self.nb_max_frames = self.n_samples // hop_length  # 480000 / 160 = 3000
```

### 4. Timestamp Precision

Whisper uses 20ms precision for timestamps (50 tokens per second):
```python
# transcribe.py:721
self.time_precision = 0.02
```

Timestamps are encoded as special tokens starting at `timestamp_begin`:
```python
# tokenizer.py:77-78
@property
def timestamp_begin(self) -> int:
    return self.no_timestamps + 1
```

### 5. Language Detection Before First Segment

If `language=None`, language detection runs on the first 30-second window (or multiple windows with `language_detection_segments`). This happens before any transcription.

## Relationships

```
WhisperModel
    ├── model: ctranslate2.models.Whisper  # Inference engine
    ├── hf_tokenizer: tokenizers.Tokenizer # HuggingFace tokenizer
    ├── feature_extractor: FeatureExtractor # Audio → Mel
    └── logger: logging.Logger

BatchedInferencePipeline
    └── model: WhisperModel  # Delegates to WhisperModel for encoding/decoding

Tokenizer (wrapper)
    └── tokenizer: tokenizers.Tokenizer  # Underlying HF tokenizer

SileroVADModel
    └── session: onnxruntime.InferenceSession  # ONNX runtime session
```

## Token Types

Whisper's tokenizer has special tokens that control behavior:

| Token | ID Property | Purpose |
|-------|------------|---------|
| `<\|startoftranscript\|>` | `sot` | Start of output |
| `<\|startofprev\|>` | `sot_prev` | Indicates previous context follows |
| `<\|startoflm\|>` | `sot_lm` | Language model start |
| `<\|endoftext\|>` | `eot` | End of text |
| `<\|notimestamps\|>` | `no_timestamps` | Disable timestamp tokens |
| `<\|nospeech\|>` | `no_speech` | No speech detected |
| `<\|en\|>`, `<\|fr\|>`, ... | `language` | Language tokens |
| `<\|transcribe\|>`, `<\|translate\|>` | `task` | Task tokens |
| `<\|0.00\|>` ... `<\|30.00\|>` | `timestamp_begin + n` | Timestamp tokens |

The prompt sequence (transcribe.py:1532-1565):
```
[sot_prev, hotwords?, previous_tokens?] + [sot, language?, task?] + [no_timestamps?] + [prefix?]
```

## Data Flow Types

```
Raw Audio (bytes/file)
    ↓ decode_audio()
np.ndarray (float32, 16kHz)
    ↓ feature_extractor()
np.ndarray (float32, shape=(80, frames))
    ↓ pad_or_trim()
np.ndarray (float32, shape=(80, 3000))
    ↓ model.encode()
ctranslate2.StorageView (encoder output)
    ↓ model.generate()
List[int] (token IDs)
    ↓ tokenizer.decode()
str (text)
```
